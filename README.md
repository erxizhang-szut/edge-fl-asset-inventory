# edge-fl-asset-inventory
asset-inventory system

Experiment Workflow

#1.Prepare Datasets:# Create dataset directories
mkdir -p datasets/{cifar10,uci_auto,mnist,fashion_mnist,industrial} # Run the main experiment (will download standard datasets automatically)

python src/fl_experiment.py
# Generate industrial dataset (if not available)
python -c "from src.data_loader import MultiDatasetLoader; \
           loader = MultiDatasetLoader('industrial', 'datasets/industrial'); \
           loader._load_industrial()"
           
#2.Run Experiments:
python src/fl_experiment.py  # Or run individual dataset experiment

python src/fl_experiment.py --config configs/cifar10.yaml

#3.Analyze Results:

Results are saved in timestamped directories under results/
Each directory contains:
  Training loss curve
  Validation metrics (accuracy for classification, MAE/RMSE for regression)
  Final trained model
  Configuration file
  Raw metrics data for further analysis
