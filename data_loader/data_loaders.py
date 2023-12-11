from torchvision import datasets, transforms
from base import BaseDataLoader, VideoFrameDataset, SingleImageDataset

from torch.utils.data import DataLoader
from torchvision import transforms

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class VideoDataLoader(BaseDataLoader):
    """
    Video data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, sequence_length=8, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        
        self.data_dir = data_dir
        self.sequence_length = 8  # Number of frames in the sequence
        self.video_dataset = VideoFrameDataset(self.data_dir, sequence_length, transforms=trsfm)

        super().__init__(self.dataset, batch_size, sequence_length, shuffle, validation_split, num_workers)

class ImageDataLoader(DataLoader):
    """
    Image data loading demo using DataLoader
    """
    def __init__(self, data_dir, batch_size, subset, shuffle=False, num_workers=1, training=True):
        # Define the transformations
        trsfm = transforms.Compose([
            transforms.Resize((384, 640)),
            transforms.ToTensor()
        ])

        # Store parameters
        self.batch_size = batch_size
        self.subset = subset
        self.data_dir = data_dir

        # Create the dataset
        self.dataset = SingleImageDataset(self.data_dir, self.subset, transforms=trsfm)

        # Initialize the DataLoader
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # # Example: Retrieve a batch from the training DataLoader
        # for image_batch, annotation_batch in self.dataset: #image_batch, annotation_batch = next(iter(self.train_loader))
        #     print("Batch of Image Tensors:", image_batch.shape)  # Expected: [batch, channels, H, W]
        #     print("Batch of Annotations:", annotation_batch.shape)  # Expected: [batch, objects, 5]
        #     break