
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Model(nn.Module):
    """CNN model for CIFAR-10 image classification"""
    
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input: 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # After two pooling layers: 32x32 -> 16x16 -> 8x8
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Output: 10 classes
        self.dropout = nn.Dropout(0.25)  # Regularization
        
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AutoRegressionModel(nn.Module):
    """Regression model for UCI Auto MPG prediction"""
    
    def __init__(self, input_dim):
        """
        Initialize regression model
        Args:
            input_dim: Number of input features
        """
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: 1 value (MPG)
        self.dropout = nn.Dropout(0.2)  # Regularization
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MNISTModel(nn.Module):
    """CNN model for MNIST and Fashion-MNIST datasets"""
    
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After two pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classes
        self.dropout = nn.Dropout(0.25)  # Regularization
        
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class IndustrialModel(nn.Module):
    """Model for industrial asset status classification"""
    
    def __init__(self, input_dim, num_classes):
        """
        Initialize classification model
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # Output: status classes
        self.dropout = nn.Dropout(0.2)  # Regularization
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(dataset_name, input_dim=None, num_classes=None):
    """
    Factory function to get appropriate model for dataset
    Args:
        dataset_name: Name of dataset
        input_dim: Input dimension for non-image datasets
        num_classes: Number of classes for classification tasks
    Returns:
        Initialized model
    """
    if dataset_name == "cifar10":
        return CIFAR10Model()
    elif dataset_name == "uci_auto":
        return AutoRegressionModel(input_dim)
    elif dataset_name in ["mnist", "fashion_mnist"]:
        return MNISTModel()
    elif dataset_name == "industrial":
        return IndustrialModel(input_dim, num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
