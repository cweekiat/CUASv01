import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
    
class SimpleSSD(BaseModel):
    def __init__(self, num_classes=1):
        super(SimpleSSD, self).__init__()

        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for regression
        self.fc_regressor = nn.Sequential(
            nn.Linear(128 * 160 * 160, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Assuming 4 values for bounding box coordinates
            nn.Sigmoid()
        )

        # Fully connected layers for classification
        self.fc_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 160 * 160, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        print(x.shape)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Regression
        bbox_output = self.fc_regressor(x)

        # Classification
        class_output = self.fc_classifier(x)

        return bbox_output, class_output 


class SimpleCNN(BaseModel):
    def __init__(self, num_classes, num_coords):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.num_coords = num_coords

        # Convolutional layers for feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers for regression
        self.fc_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240 * 32 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # Assuming 4 values for bounding box coordinates
            nn.Sigmoid()
        )

        # Fully connected layers for classification
        self.fc_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240 * 32 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_classes)
        )

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        bbox = self.fc_regressor(x)
        label = self.fc_classifier(x)


        return bbox, label
