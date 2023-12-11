import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import os
import glob
from PIL import Image
from torchvision import transforms

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

class SingleImageDataset(Dataset):
    def __init__(self, root_dir, subset, transforms=None):
        """
        Args:
            root_dir (string): Root directory with all the videos organized within subfolders for each subset.
            subset (string): One of 'train', 'val', or 'test' to specify the subset to use.
            transforms (callable, optional): Optional transform to be applied to the sample.
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.transforms = transforms
        self.data = []
        self.labels = []     
        
        # Load the data paths
        for video_folder in sorted(glob.glob(os.path.join(self.root_dir, '*'))):
            frames = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))
            annotations = [f.replace('.jpg', '.txt') for f in frames]
            self.data.extend(frames)
            self.labels.extend(annotations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.data[idx])
        
        # Load annotation
        label, bbox = self._load_annotation(self.labels[idx])
        
        # Apply transformations
        if self.transforms:
            image = self.transforms(image)
        
        # Convert annotations to tensor
        label = torch.tensor(label, dtype=torch.float).squeeze(dim=0)
        bbox = torch.tensor(bbox, dtype=torch.float).squeeze() 
        return (image, label, bbox)

    @staticmethod
    def _load_annotation(annotation_path):
        """
        Load annotation from the .txt file.
        Returns:
            annotation (list): A list of lists, where each sublist contains
                               [class, center_x, center_y, width, height].
        """
        label = []
        bbox = []
        with open(annotation_path) as f:
            for line in f.readlines():
                class_label, cx, cy, w, h = line.strip().split()
                label.append([int(class_label)])
                bbox.append([float(cx), float(cy), float(w), float(h)])
        return label, bbox
    

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the videos organized in subfolders.
            sequence_length (int): Length of the sequence of frames.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.data = []
        self.labels = []
        
        # Go through each subset (training, validation, test)
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(root_dir, subset)
            for video_folder in sorted(glob.glob(os.path.join(subset_path, '*'))):
                frames = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))
                annotations = sorted(glob.glob(os.path.join(video_folder, '*.txt')))
                
                # Only include sequences that have enough frames
                for i in range(len(frames) - sequence_length):
                    self.data.append(frames[i:i + sequence_length])
                    self.labels.append(annotations[i:i + sequence_length])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image sequence and annotations
        image_sequence = [Image.open(frame) for frame in self.data[idx]]
        annotation_sequence = [torch.tensor(self._load_annotation(label))
                               for label in self.labels[idx]]
        
        # Apply transformations
        if self.transforms:
            image_sequence = [self.transforms(image) for image in image_sequence]
        
        # Stack images to create the sequence tensor
        images_tensor = torch.stack(image_sequence)
        
        # Now annotation_sequence is a list of tensors of shape (num_objects, 5)
        # We will concatenate them along the first dimension
        annotations_tensor = torch.cat(annotation_sequence, dim=0)
        
        # Return the image tensor and the corresponding annotations
        return images_tensor, annotations_tensor

    @staticmethod
    def _load_annotation(annotation_path):
        """
        Load annotation from a single .txt file.
        Returns:
            annotation (tensor): Of shape (num_objects, 5) where 5 corresponds to
                                 'class', 'center_x', 'center_y', 'width', 'height'.
        """
        objects = []
        with open(annotation_path) as f:
            for line in f.readlines():
                class_label, cx, cy, w, h = line.strip().split()
                objects.append([int(class_label), float(cx), float(cy), float(w), float(h)])
        return objects if objects else [[-1, 0, 0, 0, 0]]  # If no objects, return a placeholder "no object".

# # Applying transformations to resize the images and convert them to Tensor
# transformations = transforms.Compose([
#     transforms.Resize((384, 640)),
#     transforms.ToTensor()
# ])

# # Create the video frame dataset
# root_dir = '../data'
# sequence_length = 1  # Number of frames in the sequence
# video_dataset = VideoFrameDataset(root_dir, sequence_length, transforms=transformations)

# # Create the DataLoader to batch and shuffle the data
# video_dataloader = DataLoader(video_dataset, batch_size=16, shuffle=False)

# # Visual checking - what the dataloader yields (disable this part for actual training)
# for images, annotations in video_dataloader:
#     print("Batch of Image Tensors (stacked sequences):", images.shape)  # Expected: [batch, sequence, channels, H, W]
#     print("Batch of Annotations:", annotations.shape)  # Expected: [batch, sequence*objects, 5]
#     break  # Just to check the first batch; remove this in actual training loop