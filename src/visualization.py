import matplotlib.pyplot as plt
import numpy as np

def plot_results(train_losses, val_metrics, dataset_name, save_dir):
    """
    Visualize training results and save to file
    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of validation metrics per epoch
        dataset_name: Name of dataset (determines plot type)
        save_dir: Directory to save visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f'{dataset_name.upper()} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation metrics based on dataset type
    plt.subplot(2, 1, 2)
    if dataset_name == "uci_auto":
        # Regression metrics: MAE and RMSE
        mae = [m['mae'] for m in val_metrics]
        rmse = [m['rmse'] for m in val_metrics]
        
        plt.plot(mae, label='Validation MAE', color='blue')
        plt.plot(rmse, label='Validation RMSE', color='red')
        plt.title('Regression Metrics')
        plt.ylabel('Error')
    else:
        # Classification metrics: Accuracy and Loss
        acc = [m['acc'] for m in val_metrics]
        val_loss = [m['loss'] for m in val_metrics]
        
        plt.plot(acc, label='Validation Accuracy', color='green')
        plt.twinx()  # Create secondary y-axis
        plt.plot(val_loss, label='Validation Loss', color='red', linestyle='--')
        plt.title('Classification Metrics')
        plt.ylabel('Accuracy/Loss')
    
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results.png", dpi=300)
    plt.close()
    
    # Save metrics for further analysis
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_metrics.npy", np.array(val_metrics))