# Anti-money Laundering on Elliptic Dataset with GNN

A comprehensive comparison of multiple Graph Neural Network architectures for detecting illicit transactions in the Elliptic cryptocurrency transaction dataset. This project implements and evaluates GCN, GAT, SAGE, Chebyshev, GATv2, and Custom GAT models.

**Published at**: 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, January 2024

## 📋 Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Visualization Tools](#visualization-tools)

## ✨ Features

- **Multiple GNN Architectures**: Implements and compares GCN, GAT, SAGE, Chebyshev, GATv2, and Custom GAT
- **Comprehensive Evaluation**: Detailed metrics including Precision, Recall, F1-Score, and AUROC
- **Visualization Tools**: Interactive network visualizations, PCA analysis, and performance metrics
- **Flexible Configuration**: YAML-based configuration for easy hyperparameter tuning
- **TensorBoard Integration**: Real-time training monitoring
- **Inference Support**: Pre-trained model inference capabilities

## 🚀 Setup

### Prerequisites

- Python >= 3.6
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.1
- CUDA (optional, for GPU acceleration)

### Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

Additional packages for visualization:
```bash
pip install plotly seaborn matplotlib networkx umap-learn
```

### Download the Data

Download the Elliptic dataset from the following [link](http://dl.dropboxusercontent.com/scl/fi/2j7nx8y3jbyypdm7r100f/dataset.zip?rlkey=veu69cngj0els6emgp549r06u&dl=0).

After downloading, extract the zip file into the `data` folder located at the root of the repository. You can also place the data in a different location and update the path in `config.yaml`.

**Required files in data folder:**
- `edgelist.csv`: Transaction graph edges
- `features.csv`: Node features
- `classes.csv`: Transaction labels (licit/illicit)

## 📁 Project Structure

```
aml-elliptic-gnn/
├── data/                      # Dataset directory (not in repo)
│   ├── edgelist.csv
│   ├── features.csv
│   ├── classes.csv
│   ├── column_distribution/   # Feature distribution plots
│   └── training_metrics/      # Training logs
├── models/
│   ├── models.py              # Standard GNN models
│   ├── models_cp.py           # Model copies/variants
│   └── custom_gat/            # Custom GAT implementation
│       ├── model.py
│       └── layer.py
├── saved_models/              # Trained model checkpoints
├── runs/                      # TensorBoard logs
├── jsonfile/                  # Metrics JSON files
├── metrics/                   # Evaluation metrics
├── config.yaml                # Configuration file
├── main.py                    # Main training script
├── train.py                   # Training utilities
├── loader.py                  # Data loading (with aggregation)
├── loader_sub.py              # Data loading (subgraph)
├── infer.py                   # Inference script
├── visualize.py               # 2D network visualization
├── 3dvisualize.py             # 3D subgraph visualization
├── animation.py               # Training animation
├── board.py                   # TensorBoard utilities
├── metrics.py                 # Metrics calculation
├── utils.py                   # Utility functions
└── requirements.txt
```

## 📜 Scripts Overview

### Core Training Scripts

#### `main.py` - Main Training Script
- **Purpose**: Train and compare multiple GNN models
- **Features**:
  - Loads configuration from `config.yaml`
  - Trains all model variants (with/without aggregation features)
  - Saves models and metrics
  - Comprehensive model comparison
- **Usage**:
  ```bash
  python main.py
  # Or with custom data path:
  python main.py --data /path/to/data
  ```

#### `train.py` - Training Utilities
- **Purpose**: Core training and testing functions
- **Features**:
  - Model training with early stopping
  - Learning rate scheduling
  - TensorBoard logging
  - Comprehensive metrics tracking
  - Model checkpointing

#### `loader.py` - Data Loader (Full Features)
- **Purpose**: Load Elliptic dataset with aggregated features
- **Features**:
  - Loads transaction features (local + aggregate)
  - Constructs PyTorch Geometric Data object
  - Handles train/test splits
  - Feature normalization

#### `loader_sub.py` - Subgraph Data Loader
- **Purpose**: Load data for subgraph analysis
- **Features**:
  - Subgraph extraction
  - Focused analysis on suspicious regions

### Model Scripts

#### `models/models.py` - Standard GNN Models
- **Models**:
  - `GCNConvolution`: Graph Convolutional Network
  - `GINConvolution`: Graph Isomorphism Network
  - `SAGEConvolution`: GraphSAGE
  - `GATConvolution`: Graph Attention Network
  - `GATv2Convolution`: Improved GAT
  - `ChebyshevConvolution`: Chebyshev spectral convolution
  - `GTCConvolution`: Graph Transformer Convolution

#### `models/custom_gat/` - Custom GAT Implementation
- Custom attention mechanism
- Specialized for transaction graphs
- Enhanced feature aggregation

### Inference and Evaluation

#### `infer.py` - Model Inference
- **Purpose**: Load trained models and perform inference
- **Features**:
  - Load saved model checkpoints
  - Batch inference on new data
  - Prediction export
- **Usage**:
  ```bash
  python infer.py
  ```

#### `metrics.py` - Metrics Calculation
- **Purpose**: Comprehensive evaluation metrics
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - AUROC, Confusion Matrix
  - Per-class metrics

### Visualization Scripts

#### `visualize.py` - 2D Network Visualization
- **Purpose**: Create 2D network visualizations
- **Features**:
  - Full graph visualization
  - Subgraph extraction and visualization
  - Node coloring by class
  - Interactive HTML output

#### `3dvisualize.py` - 3D Subgraph Visualization
- **Purpose**: Interactive 3D visualization of transaction subgraphs
- **Features**:
  - 3D network layout using Plotly
  - BFS-based subgraph extraction
  - Color-coded nodes (suspicious/normal)
  - Interactive HTML output
- **Usage**:
  ```bash
  python 3dvisualize.py
  ```

#### `animation.py` - Training Animation
- **Purpose**: Animate training progress over epochs
- **Features**:
  - Accuracy over time visualization
  - Multi-model comparison
  - Animated line plots
  - MP4 video output

#### `board.py` - TensorBoard Utilities
- **Purpose**: TensorBoard integration and visualization
- **Features**:
  - Model graph visualization
  - Parameter histograms
  - Training metrics tracking

### Data Analysis Scripts

Located in `data/`:

#### `distribution.py` - Feature Distribution Analysis
- Generates histograms for all feature columns
- Saves distribution plots to `column_distribution/`
- Helps understand feature characteristics

#### `pca_2d.py` - 2D PCA Visualization
- Performs PCA on transaction features
- 2D scatter plot colored by transaction class
- Helps visualize feature separability

#### `pca_3d.py` - 3D PCA Visualization
- 3D PCA with interactive Plotly visualization
- Interactive HTML output
- Better understanding of feature space

#### `heatmap.py` - Correlation Heatmap
- Feature correlation analysis
- Heatmap visualization
- Identifies highly correlated features

#### `boxplot.py` - Feature Distribution Boxplots
- Boxplot analysis of features
- Comparison across transaction classes
- Log-transformed and original scales

## 💻 Usage

### Training Models

**Train all models:**
```bash
python main.py
```

**Train with custom data path:**
```bash
python main.py --data /path/to/elliptic/data
```

### Individual Model Training

You can also train models individually by modifying `main.py` or using the training functions directly:

```python
from train import train, test
from models import models
from loader import load_data, data_to_pyg

# Load data
features, edges = load_data(data_path)
data = data_to_pyg(features, edges)

# Create model
model = models.GCNConvolution(args, data.num_features, args.hidden_units)

# Train
trained_model = train(args, model, data, 'GCN')
```

### Inference

```bash
python infer.py
```

Make sure to update the model path and data path in `infer.py` before running.

### Visualization

**2D Network Visualization:**
```bash
python visualize.py
```

**3D Subgraph Visualization:**
```bash
python 3dvisualize.py
```

**Training Animation:**
```bash
python animation.py
```

**Feature Analysis:**
```bash
cd data/
python distribution.py    # Feature distributions
python pca_2d.py          # 2D PCA
python pca_3d.py          # 3D PCA
python heatmap.py         # Correlation heatmap
python boxplot.py         # Boxplots
```

## ⚙️ Configuration

Edit `config.yaml` to modify hyperparameters:

```yaml
data_path: /workspace/data
use_cuda: True

hidden_units: 110              # Hidden units for aggregated features
hidden_units_noAgg: 64         # Hidden units for non-aggregated features
epochs: 13000                   # Number of training epochs
num_classes: 2                  # Binary classification
lr: 9e-3                        # Learning rate
weight_decay: 5e-4              # Weight decay (L2 regularization)
```

### Model-Specific Configuration

Models are configured in `main.py`. You can modify:
- Hidden dimensions
- Number of layers
- Dropout rates
- Activation functions

## 📊 Results

Performance comparison on Elliptic dataset:

| Model      | Precision | Recall | F1-Score | F1 Micro AVG |
|------------|-----------|--------|----------|--------------|
| GCN        | 0.832     | 0.457  | 0.59     | 0.94         |
| GAT        | 0.787     | 0.683  | 0.731    | 0.952        |
| SAGE       | 0.931     | 0.788  | 0.853    | 0.974        |
| Cheb       | **0.942** | 0.795  | **0.862**| **0.976**    |
| GATv2      | 0.891     | **0.804** | 0.845  | 0.972        |
| Custom GAT | 0.861     | 0.762  | 0.808    | 0.966        |

**Best Overall**: Chebyshev Convolution achieves the highest F1-Score and F1 Micro Average.

**Best Recall**: GATv2 achieves the highest recall, important for detecting illicit transactions.

## 🎨 Visualization Tools

### Network Visualizations

- **Full Graph**: Visualize the entire transaction network
- **Subgraph**: Focus on suspicious transaction clusters
- **3D Layout**: Interactive 3D visualization for better understanding

### Analysis Visualizations

- **PCA**: Understand feature separability
- **Heatmaps**: Feature correlations
- **Distributions**: Feature characteristics
- **Training Curves**: Model learning progress

### Output Locations

- Network visualizations: `data/interactive_network.html`, `data/subgraph_visualization*.html`
- PCA visualizations: `data/pca_2d.png`, `data/pca_3d.html`
- Training animations: `data/accuracy_animation.mp4`
- Metrics: `jsonfile/*.json`, `data/metrics.csv`

## 📈 Training Monitoring

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir=runs/
```

View:
- Training/validation loss
- Accuracy curves
- Model graph structure
- Parameter distributions

### Metrics Files

- JSON metrics: `jsonfile/metrics_*.json`
- CSV metrics: `data/metrics.csv`, `data/acc_rec.csv`
- Training logs: `data/training_metrics/`

## 🔧 Advanced Usage

### Custom Model Training

To add a new model:

1. Implement model in `models/models.py`
2. Add to `models_to_train` dictionary in `main.py`
3. Run training

### Subgraph Analysis

Use `loader_sub.py` for focused analysis:

```python
from loader_sub import load_data, data_to_pyg

features, edges = load_data(data_path, noAgg=True)
data = data_to_pyg(features, edges)
# Perform subgraph analysis
```

### Feature Engineering

Modify feature processing in `loader.py`:
- Add new features
- Change normalization
- Modify aggregation strategies

## 📚 Publication

This work was presented at **2024 IEEE 21st Consumer Communications & Networking Conference (CCNC)** held in Las Vegas on January 2024.

## 🤝 Contributing

This is a research project. For questions or improvements, please refer to the main repository or contact the authors.

## 📄 License

Please check the license file in the repository root.

---

**Note**: Large data files, model checkpoints, and visualization outputs are excluded from the repository via `.gitignore`. Please download the dataset separately and run the scripts to generate outputs.
