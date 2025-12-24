# AML - Anti Money Laundering Detection

## Installation

```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install numpy pandas scikit-learn imbalanced-learn xgboost pyvis
```

## Usage

### 1. Prepare Data

Place CSV files in the `data/raw/` directory.

### 2. Train GNN Models

Edit line 12 in `train.py` to set data path:

```python
dataset = AMLtoGraph('/path/to/AML/data', use_rf_features=False)
```

Run:

```bash
python train.py
```

### 3. Train Baseline Models

**Random Forest:**
```bash
# Edit the path in train_rf.py line 8
python train_rf.py
```

**XGBoost:**
```bash
# Edit the path in train_xgb.py line 8
python train_xgb.py
```

### 4. Visualization

```bash
# Edit the path in visualize.py line 9
python visualize.py
```

## Configuration

### Model Selection (train.py)

```python
# Default: GAT
model = GAT(in_channels=data.num_features, hidden_channels=128, out_channels=1, heads=10)

# Alternative: GATv2
# model = GATv2Convolution(in_channels=data.num_features, hidden_channels=16, 
#                         out_channels=1, heads=8, edge_dim=data.edge_attr.shape[1])

# Alternative: GAS
# model = GAS(in_channels=data.num_features, hidden_channels=64)
```

### Hyperparameters (train.py)

- `epoch = 500`: Number of training epochs
- `batch_size = 256`: Batch size
- `learning_rate = 0.00001`: Learning rate
- `threshold = 0.6`: Classification threshold
- `num_neighbors = [30] * 2`: Neighbor sampling configuration

## Scripts

- `train.py` - GNN model training
- `train_rf.py` - Random Forest training
- `train_xgb.py` - XGBoost training
- `visualize.py` - Network visualization
- `dataset.py` - Graph dataset class
- `dataset_rf.py` - Tabular dataset class (for RF/XGBoost)
- `model.py` - GNN model definitions
- `pca.py` - PCA visualization (in data/raw/)
- `timeseries.py` - Time series analysis (in data/raw/)
- `pie.py` - Distribution analysis (in data/raw/)
