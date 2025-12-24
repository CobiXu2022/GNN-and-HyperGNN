# HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs

[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://nips.cc/) [![Paper](http://img.shields.io/badge/paper-arxiv.1809.02589-B31B1B.svg)](https://arxiv.org/abs/1809.02589)

Source code for [NeurIPS 2019](https://nips.cc/) paper: [**HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs**](https://papers.nips.cc/paper/8430-hypergcn-a-new-method-for-training-graph-convolutional-networks-on-hypergraphs)

![](./hmlap.png)

**Overview of HyperGCN:** *Given a hypergraph and node features, HyperGCN approximates the hypergraph by a graph in which each hyperedge is approximated by a subgraph consisting of an edge between maximally disparate nodes and edges between each of these and every other node (mediator) of the hyperedge. A graph convolutional network (GCN) is then run on the resulting graph approximation.*

## 📋 Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Citation](#citation)

## ✨ Features

- **Hypergraph to Graph Approximation**: Converts hypergraphs to graphs for GCN training
- **Multiple Model Variants**: Support for GCN, GAT, SAGE, and Chebyshev convolutions
- **Mediator Support**: Optional mediator nodes for better hyperedge approximation
- **Fast Mode**: FastHyperGCN for efficient training on large hypergraphs
- **Multiple Datasets**: Support for coauthorship and cocitation datasets
- **Visualization Tools**: Network visualization and metrics analysis

## 🔧 Dependencies

- Compatible with PyTorch 1.0+ and Python 3.x
- NumPy
- SciPy
- NetworkX (for visualization)
- Matplotlib (for visualization)

For data (and/or splits) not used in the paper, please consider tuning hyperparameters such as hidden size, learning rate, seed, etc. on validation data.

## 📁 Project Structure

```
HyperGCN/
├── config/
│   ├── config.py              # Configuration parser
│   └── coauthorship.yml       # Dataset configuration
├── model/
│   ├── model.py               # Model initialization and training
│   ├── networks.py            # HyperGCN network definition
│   └── utils.py               # Hypergraph utilities
├── data/
│   ├── data.py                # Data loading utilities
│   ├── coauthorship/          # Coauthorship datasets
│   ├── cocitation/            # Cocitation datasets
│   └── datasets/              # Additional datasets
├── visualize/                 # Visualization outputs
├── hypergcn.py                # Simple training script
├── main.py                    # Main training script (multiple models)
├── visualize.py               # Network visualization
└── README.md
```

## 📜 Scripts Overview

### Core Training Scripts

#### `hypergcn.py` - Simple Training Script
- **Purpose**: Quick training of a single HyperGCN model
- **Usage**:
  ```bash
  python hypergcn.py
  ```
- **Features**:
  - Single model training
  - Basic configuration
  - Quick testing

#### `main.py` - Comprehensive Training Script
- **Purpose**: Train and compare multiple GNN variants on hypergraphs
- **Features**:
  - Trains multiple models: GCN, GAT, SAGE, Chebyshev
  - Saves metrics to JSON files
  - Comprehensive evaluation
  - Model comparison
- **Usage**:
  ```bash
  python main.py
  ```
- **Output**: 
  - Metrics saved to `*_metrics.json` files
  - Console output with detailed metrics

### Model Scripts

#### `model/model.py` - Model Management
- **Purpose**: Model initialization, training, and testing
- **Functions**:
  - `initialise()`: Initialize HyperGCN model
  - `train()`: Train the model
  - `test()`: Evaluate the model
- **Metrics**: Accuracy, F1-Score (macro/weighted), Precision, Recall, Log Loss, Confusion Matrix

#### `model/networks.py` - Network Definitions
- **Purpose**: HyperGCN network architecture
- **Classes**:
  - `HyperGCN`: Main HyperGCN model
  - `HyperGraphConvolution`: Hypergraph convolution layer
- **Features**:
  - Configurable depth (number of layers)
  - Dropout support
  - Mediator support
  - Fast mode option

#### `model/utils.py` - Hypergraph Utilities
- **Purpose**: Hypergraph processing utilities
- **Functions**:
  - `Laplacian`: Hypergraph Laplacian computation
  - `HyperGraphConvolution`: Convolution operation on hypergraphs

### Data Scripts

#### `data/data.py` - Data Loading
- **Purpose**: Load and preprocess hypergraph datasets
- **Features**:
  - Support for multiple dataset types
  - Train/test split handling
  - Feature and label loading
- **Supported Datasets**:
  - **Coauthorship**: DBLP, ACM, etc.
  - **Cocitation**: Cora, Citeseer, PubMed

### Configuration

#### `config/config.py` - Configuration Parser
- **Purpose**: Parse configuration parameters
- **Parameters**:
  - Dataset selection
  - Model hyperparameters (depth, dropout, epochs)
  - Training parameters (learning rate, weight decay)
  - GPU and seed settings
  - Mediator and fast mode flags

Edit `config/config.py` to modify:
```python
data = "coauthorship"  # or "cocitation"
dataset = "dblp"       # or "cora", "citeseer", "pubmed", etc.

mediators = False      # Use mediators (True) or not (False)
fast = False           # FastHyperGCN (True) or standard (False)
split = 0              # Train-test split number

gpu = 0                # GPU number
cuda = True            # Use CUDA
seed = 1               # Random seed

depth = 3              # Number of hidden layers
dropout = 0.5          # Dropout probability
epochs = 200           # Training epochs

rate = 0.01            # Learning rate
decay = 0.005          # Weight decay
```

### Visualization

#### `visualize.py` - Network Visualization
- **Purpose**: Visualize hypergraph structure and learned embeddings
- **Features**:
  - Network layout visualization
  - Node coloring by class
  - Hyperedge visualization
  - UMAP-based embedding visualization
- **Usage**:
  ```bash
  python visualize.py
  ```
- **Output**: 
  - PNG images: `*_gcn.png`, `*_gat.png`, etc.
  - Metrics JSON files

## 🚀 Getting Started

### Download Datasets

Please download the datasets from the [original HyperGCN repository](https://github.com/malllabiisc/HyperGCN) and copy the `data` directory into this repository.

**Required structure:**
```
data/
├── coauthorship/
│   ├── dblp/
│   ├── acm/
│   └── ...
└── cocitation/
    ├── cora/
    ├── citeseer/
    ├── pubmed/
    └── ...
```

### Installation

```bash
pip install torch numpy scipy networkx matplotlib
```

## 💻 Usage

### Basic Training

**Train a single model:**
```bash
python hypergcn.py --mediators True --split 1 --data coauthorship --dataset dblp
```

**Train multiple models:**
```bash
python main.py
```

### Training Parameters

- `--mediators`: Use mediators (True) or not (False)
- `--split`: Train-test split number (0, 1, 2, ...)
- `--data`: Dataset type (`coauthorship` or `cocitation`)
- `--dataset`: Specific dataset name
  - For coauthorship: `dblp`, `acm`, etc.
  - For cocitation: `cora`, `citeseer`, `pubmed`

### Example Commands

**Coauthorship dataset (DBLP):**
```bash
python hypergcn.py --mediators True --split 1 --data coauthorship --dataset dblp
```

**Cocitation dataset (Cora):**
```bash
python hypergcn.py --mediators False --split 0 --data cocitation --dataset cora
```

**Multiple models comparison:**
```bash
python main.py
```

This will train GCN, GAT, SAGE, and Chebyshev models and save metrics to JSON files.

## ⚙️ Configuration

### Edit `config/config.py`

Key parameters:

```python
# Dataset
data = "coauthorship"      # or "cocitation"
dataset = "dblp"          # Dataset name

# Hypergraph approximation
mediators = False          # Use mediators
fast = False              # FastHyperGCN mode
split = 0                 # Train-test split

# Hardware
gpu = 0                   # GPU ID
cuda = True               # Use CUDA
seed = 1                  # Random seed

# Model architecture
depth = 3                 # Number of layers
dropout = 0.5            # Dropout rate
epochs = 200             # Training epochs

# Optimization
rate = 0.01              # Learning rate
decay = 0.005            # Weight decay
```

### Hyperparameter Tuning

For datasets not in the paper, tune:
- **Hidden size**: Adjust in `model/networks.py` (line 20-24)
- **Learning rate**: Modify `rate` in `config/config.py`
- **Depth**: Change `depth` parameter
- **Dropout**: Adjust `dropout` for regularization

## 📊 Expected Results

The models should achieve competitive results on standard hypergraph datasets:

- **Coauthorship (DBLP)**: ~90%+ accuracy
- **Cocitation (Cora)**: ~70%+ accuracy
- **Cocitation (PubMed)**: ~70%+ accuracy

Metrics saved include:
- Accuracy
- Macro F1-Score
- Weighted F1-Score
- Precision
- Recall
- Log Loss
- Confusion Matrix

## 🎨 Visualization

### Network Visualization

```bash
python visualize.py
```

Generates:
- Network layout images
- Model comparison plots
- Embedding visualizations

### Metrics Analysis

Metrics are saved to JSON files:
- `*_metrics.json`: Per-model metrics
- `actor_metrics.json`, `amazon_metrics.json`, etc.: Dataset-specific metrics

## 📚 Datasets

### Coauthorship Datasets
- **DBLP**: Computer science coauthorship network
- **ACM**: ACM coauthorship network

### Cocitation Datasets
- **Cora**: Paper citation network
- **Citeseer**: Paper citation network
- **PubMed**: Biomedical paper citation network

### Additional Datasets
- **Amazon**: Product co-purchase network
- **Actor**: Actor collaboration network
- **Pokec**: Social network
- **Twitch**: Streaming network

## 🔬 Model Variants

### Standard HyperGCN
- Full hypergraph approximation
- Re-approximates at each layer
- More accurate but slower

### FastHyperGCN
- Pre-computes approximation
- Faster training
- Slightly less accurate

### With/Without Mediators
- **With mediators**: Better hyperedge approximation
- **Without mediators**: Simpler, faster

## 📈 Performance Tips

1. **Use mediators** for better accuracy on complex hypergraphs
2. **Fast mode** for large datasets when speed is important
3. **Tune depth** based on dataset size and complexity
4. **Adjust dropout** to prevent overfitting
5. **Multiple splits** for robust evaluation

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Dataset not found**: Check data directory structure
3. **Low accuracy**: Try tuning hyperparameters or using mediators

## 📖 Citation

If you use this code, please cite the original paper:

```bibtex
@incollection{hypergcn_neurips19,
title = {HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs},
author = {Yadati, Naganand and Nimishakavi, Madhav and Yadav, Prateek and Nitin, Vikram and Louis, Anand and Talukdar, Partha},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 32},
pages = {1509--1520},
year = {2019},
publisher = {Curran Associates, Inc.}
}
```

## 📄 License

Please check the LICENSE file in the repository.

---

**Note**: Large data files and visualization outputs are excluded from the repository. Please download datasets separately and run scripts to generate outputs.
