import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MultiDatasetLoader:
    """Loader for multiple datasets used in federated learning experiments"""
    
    def __init__(self, dataset_name, data_path):
        """
        Initialize dataset loader
        Args:
            dataset_name: Name of dataset (cifar10, uci_auto, mnist, etc.)
            data_path: Path to dataset directory
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        
    def load(self):
        """Load dataset based on name"""
        if self.dataset_name == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name == "uci_auto":
            return self._load_uci_auto()
        elif self.dataset_name == "mnist":
            return self._load_mnist()
        elif self.dataset_name == "fashion_mnist":
            return self._load_fashion_mnist()
        elif self.dataset_name == "industrial":
            return self._load_industrial()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_cifar10(self):
        """Load and preprocess CIFAR-10 image classification dataset"""
        # Define image transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download and load dataset
        train_set = datasets.CIFAR10(
            root=self.data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(
            root=self.data_path, train=False, download=True, transform=transform)
        
        return train_set, test_set, 10  # Return datasets and number of classes
    
    def _load_uci_auto(self):
        """Load and preprocess UCI Auto dataset for regression"""
        # Load data from CSV
        data = pd.read_csv(f"{self.data_path}/auto_data.csv")
        
        # Preprocess data
        data = data.dropna()
        data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
        data = data.dropna()
        
        # Separate features and target
        X = data.drop('mpg', axis=1).values
        y = data['mpg'].values
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        train_set = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        test_set = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        
        return train_set, test_set, 1  # Output dimension = 1 for regression
    
    def _load_mnist(self):
        """Load and preprocess MNIST handwritten digits dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load dataset
        train_set = datasets.MNIST(
            root=self.data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(
            root=self.data_path, train=False, download=True, transform=transform)
        
        return train_set, test_set, 10  # 10 digit classes
    
    def _load_fashion_mnist(self):
        """Load and preprocess Fashion-MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Download and load dataset
        train_set = datasets.FashionMNIST(
            root=self.data_path, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(
            root=self.data_path, train=False, download=True, transform=transform)
        
        return train_set, test_set, 10  # 10 fashion categories
    
    def _load_industrial(self):
        """Load and preprocess industrial asset dataset"""
        # Load data from CSV
        data = pd.read_csv(f"{self.data_path}/sensor_data.csv")
        
        # Extract features and target
        features = data[['temperature', 'vibration', 'location_x', 'location_y']]
        target = data['status']
        
        # Encode categorical labels
        status_mapping = {'normal': 0, 'warning': 1, 'critical': 2}
        target = target.map(status_mapping)
        
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        train_set = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long)
        )
        test_set = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.long)
        )
        
        return train_set, test_set, 3  # 3 status classes