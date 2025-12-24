# Anti Money Laundering Detection with Graph Attention Network

This repository provides model training of Graph Attention Network in Anti Money Laundering Detection problem. The project implements multiple GNN architectures and baseline models for detecting suspicious financial transactions.

**Dataset**: [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## 📋 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Model Training](#model-training)
- [Data Analysis](#data-analysis)
- [References](#references)

## ✨ Features

- **Multiple GNN Models**: GAT, GAS, and GATv2Convolution implementations
- **Baseline Models**: Random Forest and XGBoost for comparison
- **Graph Construction**: Automatic conversion of transaction data to graph structure
- **Visualization Tools**: Interactive network visualization of suspicious subgraphs
- **Data Preprocessing**: Comprehensive feature engineering and resampling

## 🚀 Getting Started

### Prerequisites

Main dependencies:
- Python >= 3.6
- PyTorch >= 1.8.0
- PyTorch Geometric >= 2.0.0
- NumPy
- Pandas
- scikit-learn
- imbalanced-learn (for SMOTE)
- XGBoost
- pyvis (for visualization)

### Installation

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install numpy pandas scikit-learn imbalanced-learn xgboost pyvis
```

## 📁 Project Structure

```
AML/
├── data/
│   ├── raw/              # Place your CSV files here
│   ├── processed/       # Auto-generated processed data
│   ├── processed1/       # Alternative processed versions
│   ├── processed2/
│   └── processed3/
├── visualization/        # Generated visualization outputs
├── dataset.py           # Main dataset class for GNN
├── dataset_rf.py         # Dataset class for Random Forest/XGBoost
├── model.py              # GNN model definitions (GAT, GAS, GATv2)
├── train.py              # GNN model training script
├── train_rf.py           # Random Forest training script
├── train_xgb.py          # XGBoost training script
├── visualize.py          # Network visualization script
├── anti-money-laundering-detection-with-gnn.ipynb  # Analysis notebook
└── README.md
```

## 📜 Scripts Overview

### Core Scripts

#### `dataset.py` - Graph Dataset Class
- **Purpose**: Converts transaction CSV data into PyTorch Geometric graph format
- **Features**:
  - Implements `AMLtoGraph` class extending `InMemoryDataset`
  - Automatic edge construction based on transaction relationships
  - Feature engineering and normalization
  - Support for Random Forest features (`use_rf_features` parameter)
  - Class weight calculation for imbalanced data
  - Data resampling (SMOTE, RandomOverSampler)

#### `dataset_rf.py` - Tabular Dataset Class
- **Purpose**: Prepares data for traditional ML models (RF, XGBoost)
- **Features**:
  - Feature extraction from transaction data
  - Train/test split functionality
  - Compatible with scikit-learn pipelines

#### `model.py` - GNN Model Definitions
- **Models Available**:
  - **GAT**: Graph Attention Network with multi-head attention
  - **GAS**: Graph Attention with Sampling
  - **GATv2Convolution**: Improved GAT with dynamic attention mechanism
- **Usage**: Import and instantiate models in `train.py`

#### `train.py` - GNN Training Script
- **Purpose**: Main training script for GNN models
- **Features**:
  - Neighbor sampling for large graphs
  - Class-weighted loss function
  - Comprehensive metrics: Accuracy, AUROC, F1-Score, Recall, Confusion Matrix
  - Validation during training
- **Hyperparameters**:
  - Epochs: 500
  - Batch size: 256
  - Learning rate: 0.00001
  - Optimizer: SGD
  - Threshold: 0.6 (for binary classification)

#### `train_rf.py` - Random Forest Training
- **Purpose**: Train Random Forest baseline model
- **Features**:
  - Balanced class weights
  - Comprehensive evaluation metrics
  - Model persistence (saves trained model)

#### `train_xgb.py` - XGBoost Training
- **Purpose**: Train XGBoost baseline model
- **Features**:
  - Binary classification with logistic objective
  - Comprehensive evaluation metrics
  - Model persistence

#### `visualize.py` - Network Visualization
- **Purpose**: Create interactive visualizations of suspicious transaction subgraphs
- **Features**:
  - BFS-based subgraph sampling around suspicious nodes
  - Interactive HTML visualization using pyvis
  - Color-coded nodes (suspicious vs. normal)
  - Configurable sampling size and depth

### Data Analysis Scripts

Located in `data/raw/`:

#### `pca.py` - PCA Visualization
- Performs PCA on transaction features
- 2D scatter plot colored by laundering labels
- Helps visualize feature separability

#### `timeseries.py` - Time Series Analysis
- Analyzes transaction patterns over time
- Stacked area plots by bank
- Monthly aggregation of transaction amounts

#### `pie.py` - Distribution Analysis
- Pie charts for categorical feature distributions

## 💻 Usage

### 1. Prepare Data

Create the directory structure and place your CSV file:

```bash
mkdir -p data/raw
# Place your transaction CSV file in data/raw/
```

The expected directory structure:
```
├── data
│   ├── raw
│       └── your_transaction_file.csv
├── dataset.py
├── model.py
└── train.py
```

### 2. Train GNN Models

Edit `train.py` line 12 to set your data path:

```python
dataset = AMLtoGraph('/path/to/AML/data', use_rf_features=False)
```

Run training:

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

### 4. Generate Visualizations

```bash
# Edit the path in visualize.py line 9
python visualize.py
```

This will generate an interactive HTML file in the `visualization/` directory.

## 🔬 Data Preprocessing

All data preprocessing is done in `dataset.py`:

1. **Graph Construction**: 
   - Nodes represent transactions
   - Edges connect related transactions (based on time windows and relationships)

2. **Feature Engineering**:
   - Transaction amounts, timestamps, bank information
   - Aggregated features from transaction history
   - Optional Random Forest derived features

3. **Handling Imbalanced Data**:
   - Class weight calculation
   - Support for SMOTE and RandomOverSampler

4. **Data Splits**:
   - Automatic train/validation/test splits
   - Configurable split ratios

## 🎯 Model Training

### GNN Training

The main training script (`train.py`) supports:

- **Neighbor Sampling**: Uses `NeighborLoader` for efficient training on large graphs
- **Class Weighting**: Automatically calculates and applies class weights
- **Metrics Tracking**: 
  - Accuracy
  - AUROC (Area Under ROC Curve)
  - F1-Score
  - Recall
  - Confusion Matrix

### Baseline Models

Both Random Forest and XGBoost scripts:
- Use balanced class weights
- Provide comprehensive evaluation metrics
- Save trained models for later use

## 📊 Data Analysis and Visualization

The Jupyter notebook `anti-money-laundering-detection-with-gnn.ipynb` provides:
- Feature engineering explanations
- Data visualization pipeline
- Dataset design details
- Exploratory data analysis

## 🔧 Configuration

### Model Selection

Switch between models in `train.py`:

```python
# Default: GAT
model = GAT(in_channels=data.num_features, hidden_channels=128, out_channels=1, heads=10)

# Alternative: GATv2
# model = GATv2Convolution(in_channels=data.num_features, hidden_channels=16, 
#                         out_channels=1, heads=8, edge_dim=data.edge_attr.shape[1])

# Alternative: GAS
# model = GAS(in_channels=data.num_features, hidden_channels=64)
```

### Hyperparameters

Key hyperparameters in `train.py`:
- `epoch = 500`: Number of training epochs
- `batch_size = 256`: Batch size for training and testing
- `learning_rate = 0.00001`: Learning rate for SGD optimizer
- `threshold = 0.6`: Classification threshold for binary predictions
- `num_neighbors = [30] * 2`: Neighbor sampling configuration

## 📈 Expected Results

The models should achieve:
- **GAT**: Good balance between precision and recall
- **Random Forest**: Fast training, baseline performance
- **XGBoost**: Strong performance on tabular features

Metrics are printed during training and include:
- Accuracy
- AUROC
- F1-Score
- Recall
- Confusion Matrix

## 📚 References

Some of the feature engineering in this repository are referenced to:

1. [Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv preprint arXiv:1908.02591.](https://arxiv.org/pdf/1908.02591.pdf)

2. [Johannessen, F., & Jullum, M. (2023). Finding Money Launderers Using Heterogeneous Graph Neural Networks. arXiv preprint arXiv:2307.13499.](https://arxiv.org/pdf/2307.13499.pdf)

## 🤝 Contributing

This is a research project. For questions or improvements, please open an issue or submit a pull request.

## 📄 License

Please check the license file in the repository root.

---

**Note**: Make sure to download the dataset from Kaggle and place it in the `data/raw/` directory before running the scripts.
