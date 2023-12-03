from torchvision import datasets, transforms
from base import BaseDataLoader, VideoFrameDataset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torchvision.io import read_video



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
            transforms.Resize((384, 640)),
            transforms.ToTensor()
        ])
        
        self.data_dir = data_dir
        self.sequence_length = 8  # Number of frames in the sequence
        self.video_dataset = VideoFrameDataset(self.data_dir, sequence_length, transforms=trsfm)

        super().__init__(self.dataset, batch_size, sequence_length, shuffle, validation_split, num_workers)

# # Create the DataLoader to batch and shuffle the data
# video_dataloader = DataLoader(video_dataset, batch_size=16, shuffle=False)