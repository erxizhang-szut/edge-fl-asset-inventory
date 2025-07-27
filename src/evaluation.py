import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

def evaluate_model(model, data_loader, dataset_type, criterion=None):
    """
    Evaluate model performance on test dataset
    Args:
        model: Trained model
        data_loader: DataLoader for test dataset
        dataset_type: Type of dataset (determines evaluation metrics)
        criterion: Loss function (optional)
    Returns:
        Evaluation metrics based on dataset type
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            
            # Collect predictions and targets
            if dataset_type == "uci_auto":
                # Regression task: use raw outputs
                preds = outputs.squeeze().numpy()
                targets_np = targets.numpy()
            else:
                # Classification task: get class predictions
                _, preds = torch.max(outputs, 1)
                preds = preds.numpy()
                targets_np = targets.numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets_np)
            
            # Calculate loss if criterion is provided
            if criterion:
                if dataset_type == "uci_auto":
                    loss = criterion(outputs.squeeze(), targets.float())
                else:
                    loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
    
    # Calculate metrics based on dataset type
    if dataset_type == "uci_auto":
        # Regression metrics
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        avg_loss = running_loss / len(data_loader.dataset) if criterion else None
        return mae, rmse
    else:
        # Classification metrics
        acc = accuracy_score(all_targets, all_preds)
        avg_loss = running_loss / len(data_loader.dataset) if criterion else None
        return acc, avg_loss