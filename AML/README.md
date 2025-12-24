# AML - Anti Money Laundering Detection

## Installation

```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install numpy pandas scikit-learn imbalanced-learn xgboost pyvis
```

**Note**: Install PyTorch with CUDA support if you have a GPU. Adjust the installation command based on your CUDA version from [PyTorch website](https://pytorch.org/).

## Data Preparation

1. Download the IBM Transactions for Anti-Money Laundering dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

2. Place the CSV file in the `data/raw/` directory:
   ```
   AML/
   └── data/
       └── raw/
           └── your_transaction_file.csv
   ```

3. The script will automatically process the data and create graph structures.

## Usage

### 1. Train GNN Models

**Step 1**: Edit `train.py` line 12 to set your data path:

```python
# Change this line (around line 12)
dataset = AMLtoGraph('/home/zmxu/Desktop/ly/GNN-and-HyperGNN-all/AML/data', use_rf_features=False)
```

**Step 2**: Run training:

```bash
python train.py
```

**Output**: Training progress, metrics (accuracy, AUROC, F1-score, recall), and model checkpoints.

### 2. Train Baseline Models

**Random Forest:**

1. Edit `train_rf.py` line 8 to set data path:
   ```python
   # Change this line (around line 8)
   data_path = '/home/zmxu/Desktop/ly/GNN-and-HyperGNN-all/AML/data/raw/your_file.csv'
   ```

2. Run:
   ```bash
   python train_rf.py
   ```

**XGBoost:**

1. Edit `train_xgb.py` line 8 to set data path:
   ```python
   # Change this line (around line 8)
   data_path = '/home/zmxu/Desktop/ly/GNN-and-HyperGNN-all/AML/data/raw/your_file.csv'
   ```

2. Run:
   ```bash
   python train_xgb.py
   ```

### 3. Visualization

**Step 1**: Edit `visualize.py` line 9 to set data path:

```python
# Change this line (around line 9)
dataset = AMLtoGraph('/home/zmxu/Desktop/ly/GNN-and-HyperGNN-all/AML/data', use_rf_features=False)
```

**Step 2**: Run visualization:

```bash
python visualize.py
```

**Output**: Interactive HTML visualization file in `visualization/` directory showing suspicious transaction subgraphs.

### 4. Data Analysis Scripts

Located in `data/raw/`:

- **PCA Visualization**: `python data/raw/pca.py`
- **Time Series Analysis**: `python data/raw/timeseries.py`
- **Distribution Analysis**: `python data/raw/pie.py`

**Note**: Edit paths in these scripts to point to your data file.

## Configuration

### Model Selection (train.py)

Edit `train.py` to switch between models:

```python
# Default: GAT (around line 50-60)
model = GAT(in_channels=data.num_features, hidden_channels=128, out_channels=1, heads=10)

# Alternative: GATv2 (uncomment to use)
# model = GATv2Convolution(in_channels=data.num_features, hidden_channels=16, 
#                         out_channels=1, heads=8, edge_dim=data.edge_attr.shape[1])

# Alternative: GAS (uncomment to use)
# model = GAS(in_channels=data.num_features, hidden_channels=64)
```

### Hyperparameters (train.py)

Edit these parameters in `train.py`:

```python
epoch = 500              # Number of training epochs
batch_size = 256         # Batch size for training/testing
learning_rate = 0.00001  # Learning rate for SGD optimizer
threshold = 0.6          # Classification threshold for binary predictions
num_neighbors = [30] * 2 # Neighbor sampling configuration for large graphs
```

### Dataset Configuration (dataset.py)

In `dataset.py`, you can modify:

- **Resampling methods**: SMOTE, RandomOverSampler (for imbalanced data)
- **Feature engineering**: Random Forest features (`use_rf_features=True/False`)
- **Train/test split ratios**: Default split configuration
- **Graph construction**: Edge creation logic based on transaction relationships

## Output Files

- **Processed data**: `data/processed/`, `data/processed1/`, etc.
- **Visualizations**: `visualization/suspicious_subgraph.html`
- **Model checkpoints**: Saved during training (check script for save location)
- **Analysis plots**: `data/raw/*.png` (from analysis scripts)

## Scripts

- `train.py` - GNN model training (GAT, GATv2, GAS)
- `train_rf.py` - Random Forest baseline training
- `train_xgb.py` - XGBoost baseline training
- `visualize.py` - Network visualization (interactive HTML)
- `dataset.py` - Graph dataset class (converts CSV to PyTorch Geometric format)
- `dataset_rf.py` - Tabular dataset class (for RF/XGBoost)
- `model.py` - GNN model definitions
- `data/raw/pca.py` - PCA visualization
- `data/raw/timeseries.py` - Time series analysis
- `data/raw/pie.py` - Distribution analysis
