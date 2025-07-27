import torch
import yaml
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from .data_loader import MultiDatasetLoader
from .models import get_model
from .evaluation import evaluate_model
from .visualization import plot_results

class FederatedLearningExperiment:
    """Main class for running federated learning experiments"""
    
    def __init__(self, config_path):
        """
        Initialize experiment with configuration
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Create result directory with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.result_dir = f"results/{self.config['dataset']}_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Save configuration for reference
        with open(f"{self.result_dir}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
    
    def run(self):
        """Run the federated learning experiment"""
        print(f"\n===== Starting Experiment: {self.config['dataset'].upper()} =====")
        
        # Load dataset
        loader = MultiDatasetLoader(
            self.config['dataset'], 
            f"datasets/{self.config['dataset']}"
        )
        train_set, test_set, num_classes = loader.load()
        
        # Initialize model
        input_dim = train_set[0][0].shape[0] if len(train_set[0][0].shape) == 1 else None
        global_model = get_model(self.config['dataset'], input_dim, num_classes)
        
        # Create data loaders
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Configure loss function and optimizer
        if self.config['dataset'] == "uci_auto":
            criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
        else:
            criterion = torch.nn.CrossEntropyLoss()  # Cross Entropy for classification
        
        optimizer = torch.optim.Adam(
            global_model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Initialize tracking variables
        train_losses = []
        val_metrics = []
        
        # Training loop
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            global_model.train()  # Set model to training mode
            running_loss = 0.0
            
            # Batch training
            for inputs, targets in train_loader:
                optimizer.zero_grad()  # Reset gradients
                
                # Forward pass
                outputs = global_model(inputs)
                
                # Calculate loss
                if self.config['dataset'] == "uci_auto":
                    # Regression task: adjust output shape
                    loss = criterion(outputs.squeeze(), targets.float())
                else:
                    loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item() * inputs.size(0)
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(train_set)
            train_losses.append(epoch_loss)
            
            # Evaluate model
            if self.config['dataset'] == "uci_auto":
                # Regression evaluation
                val_mae, val_rmse = evaluate_model(
                    global_model, test_loader, self.config['dataset'])
                val_metrics.append({'mae': val_mae, 'rmse': val_rmse})
                print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | "
                      f"Time: {time.time()-epoch_start:.2f}s")
            else:
                # Classification evaluation
                val_acc, val_loss = evaluate_model(
                    global_model, test_loader, self.config['dataset'], criterion)
                val_metrics.append({'acc': val_acc, 'loss': val_loss})
                print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {time.time()-epoch_start:.2f}s")
        
        # Save final model
        model_path = f"{self.result_dir}/final_model.pt"
        torch.save(global_model.state_dict(), model_path)
        
        # Visualize and save results
        plot_results(train_losses, val_metrics, self.config['dataset'], self.result_dir)
        
        print(f"\nExperiment completed! Results saved to {self.result_dir}")

if __name__ == "__main__":
    """Run experiments for all datasets"""
    datasets = [
        "cifar10",        # Image classification (vehicles/animals)
        "uci_auto",       # Regression (vehicle MPG prediction)
        "mnist",          # Digit recognition
        "fashion_mnist",  # Fashion item classification
        "industrial"      # Industrial asset status prediction
    ]
    
    # Run experiment for each dataset
    for dataset in datasets:
        config_path = f"configs/{dataset}.yaml"
        experiment = FederatedLearningExperiment(config_path)
        experiment.run()