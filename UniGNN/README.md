# UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks

This repository contains the source code for the paper [_UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks_](https://arxiv.org/abs/2105.00956), accepted by IJCAI 2021.

![](utils/figure.png)

## 📋 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Model Variants](#model-variants)
- [Visualization and Analysis](#visualization-and-analysis)
- [Citation](#citation)

## ✨ Features

- **Unified Framework**: Single framework for both graph and hypergraph neural networks
- **Multiple Model Variants**: UniGCN, UniGAT, UniGIN, UniSAGE, UniGCNII
- **Semi-supervised Learning**: Node classification on graphs and hypergraphs
- **Inductive Learning**: Support for evolving hypergraphs
- **Deep Networks**: Deep-layered architectures with residual connections
- **Comprehensive Tools**: Data conversion, visualization, and analysis utilities

## 🚀 Getting Started

### Prerequisites

- Python >= 3.6
- PyTorch >= 1.8.0
- PyTorch Geometric >= 2.0.0
- NumPy, SciPy
- Additional packages: `path`, `tqdm`, `umap-learn` (for visualization)

### Installation

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torch-geometric

# Install additional packages
pip install scipy path tqdm umap-learn matplotlib
```

### Download Datasets

Please download the datasets from [HyperGCN](https://github.com/malllabiisc/HyperGCN) and copy the `data` directory into this repository.

**Required structure:**
```
data/
├── coauthorship/
│   ├── dblp/
│   └── ...
└── cocitation/
    ├── cora/
    ├── citeseer/
    ├── pubmed/
    └── ...
```

## 📁 Project Structure

```
UniGNN/
├── model/
│   ├── __init__.py
│   ├── UniGNN.py          # Main UniGNN model definitions
│   └── HyperGCN.py         # HyperGCN compatibility
├── utils/                  # Utility functions
├── data/                   # Dataset directory (not in repo)
├── runs/                   # Training outputs (not in repo)
├── labels/                 # Saved labels (not in repo)
├── config.py               # Configuration parser
├── train.py                # Semi-supervised training
├── train_val.py            # Deep-layered training with validation
├── train_evolving.py       # Inductive learning on evolving hypergraphs
├── prepare.py              # Data preparation utilities
├── convert.py              # Data format conversion
├── convert0.py             # Alternative conversion script
├── view.py                 # Feature visualization (UMAP)
├── check.py                # Model checkpoint inspection
├── logger.py               # Logging utilities
└── README.md
```

## 📜 Scripts Overview

### Core Training Scripts

#### `train.py` - Semi-supervised Node Classification
- **Purpose**: Standard semi-supervised learning on static hypergraphs
- **Features**:
  - Multiple runs for statistical significance
  - Early stopping
  - Comprehensive logging
  - Model checkpointing
- **Usage**:
  ```bash
  python train.py --data=coauthorship --dataset=dblp --model-name=UniSAGE
  ```
- **Output**: Average test accuracy across multiple runs

#### `train_val.py` - Deep-layered Training with Validation
- **Purpose**: Train deep networks (e.g., UniGCNII) with validation-based early stopping
- **Features**:
  - Deep network support (up to 32+ layers)
  - Validation-based model selection
  - Normalization and self-loop options
  - Best model checkpointing
- **Usage**:
  ```bash
  python train_val.py --data=cocitation --dataset=cora --use-norm --add-self-loop \
                      --model-name=UniGCNII --nlayer=32 --dropout=0.2 \
                      --patience=150 --epochs=1000 --n-runs=1
  ```
- **Output**: Best validation accuracy and final test accuracy

#### `train_evolving.py` - Inductive Learning on Evolving Hypergraphs
- **Purpose**: Train on seen nodes and test on unseen nodes in evolving hypergraphs
- **Features**:
  - Inductive learning setup
  - Seen/unseen node evaluation
  - Temporal hypergraph support
- **Usage**:
  ```bash
  python train_evolving.py --data=coauthorship --dataset=dblp --model-name=UniGIN
  ```
- **Output**: Accuracy on seen and unseen nodes

### Model Scripts

#### `model/UniGNN.py` - Unified Model Definitions
- **Models**:
  - **UniGCN**: Unified Graph Convolutional Network
  - **UniGAT**: Unified Graph Attention Network
  - **UniGIN**: Unified Graph Isomorphism Network
  - **UniSAGE**: Unified GraphSAGE
  - **UniGCNII**: Deep UniGCN with residual connections
- **Features**:
  - First aggregation: Hyperedge aggregation (max, sum, mean)
  - Second aggregation: Node aggregation (max, sum, mean)
  - Configurable activation functions
  - Multi-head attention support

#### `model/HyperGCN.py` - HyperGCN Compatibility
- Compatibility layer for HyperGCN models
- Allows comparison with baseline methods

### Data Processing Scripts

#### `prepare.py` - Data Preparation
- **Purpose**: Prepare and preprocess datasets
- **Features**:
  - Data normalization
  - Feature extraction
  - Train/test split preparation
- **Usage**:
  ```bash
  python prepare.py
  ```

#### `convert.py` - Data Format Conversion
- **Purpose**: Convert PyTorch tensors to JSON format
- **Features**:
  - Batch conversion of `.pt` files to `.json`
  - Useful for feature analysis
- **Usage**: Edit paths in script and run:
  ```bash
  python convert.py
  ```

#### `convert0.py` - Single File Conversion
- **Purpose**: Convert a single `.pt` file to JSON
- **Usage**: Edit file path in script and run:
  ```bash
  python convert0.py
  ```

### Analysis and Visualization Scripts

#### `view.py` - Feature Visualization
- **Purpose**: Visualize learned node embeddings using UMAP
- **Features**:
  - 2D/3D UMAP visualization
  - Color-coded by class labels
  - Feature analysis
- **Usage**: Edit paths in script:
  ```python
  # Update paths to your feature and label files
  features = torch.load('path/to/features.pt')
  labels = torch.load('path/to/labels.pt')
  python view.py
  ```

#### `check.py` - Model Checkpoint Inspection
- **Purpose**: Inspect saved model checkpoints
- **Features**:
  - View model state dict
  - Check model parameters
  - Verify model structure
- **Usage**: Edit model path and run:
  ```bash
  python check.py
  ```

### Utility Scripts

#### `config.py` - Configuration Parser
- **Purpose**: Parse command-line arguments and configuration
- **Parameters**:
  - Dataset selection
  - Model selection
  - Hyperparameters
  - Training settings

#### `logger.py` - Logging Utilities
- **Purpose**: Comprehensive logging system
- **Features**:
  - File and console logging
  - Training progress tracking
  - Metrics logging

## 💻 Usage

### Semi-supervised Hypernode Classification

**Basic training:**
```bash
python train.py --data=coauthorship --dataset=dblp --model-name=UniSAGE
```

**With custom parameters:**
```bash
python train.py --data=coauthorship --dataset=dblp --model-name=UniGAT \
                --nlayer=3 --nhid=64 --dropout=0.6 --lr=0.01 \
                --epochs=200 --n-runs=10
```

**Expected output:**
```
Average final test accuracy: 88.53697896003723 ± 0.21541351159170083
```

### Deep-layered HyperGraph Neural Networks

**Train UniGCNII (deep network):**
```bash
python train_val.py --data=cocitation --dataset=cora \
                    --use-norm --add-self-loop \
                    --model-name=UniGCNII --nlayer=32 \
                    --dropout=0.2 --patience=150 \
                    --epochs=1000 --n-runs=1
```

**Expected output:**
```
best test accuracy: 72.92, acc(last): 73.16
```

### Inductive Learning on Evolving Hypergraphs

**Train on evolving hypergraph:**
```bash
python train_evolving.py --data=coauthorship --dataset=dblp \
                         --model-name=UniGIN
```

**Expected output:**
```
Average final seen: 89.47043359279633 ± 0.23909984894330719
Average final unseen: 82.99945294857025 ± 0.40471906653645956
```

## 🎯 Model Variants

### UniGCN
- Unified Graph Convolutional Network
- Mean aggregation for hyperedges and nodes
- Good baseline performance

### UniGAT
- Unified Graph Attention Network
- Attention-based aggregation
- Better for complex relationships

### UniGIN
- Unified Graph Isomorphism Network
- MLP-based aggregation
- Strong for inductive learning

### UniSAGE
- Unified GraphSAGE
- Sampling-based aggregation
- Efficient for large graphs

### UniGCNII
- Deep UniGCN with residual connections
- Supports very deep networks (32+ layers)
- Best for deep architectures

## ⚙️ Configuration Options

### Command-line Arguments

```
usage: UniGNN: Unified Graph and Hypergraph Message Passing Model
       [-h] [--data DATA] [--dataset DATASET] [--model-name MODEL_NAME]
       [--first-aggregate FIRST_AGGREGATE]
       [--second-aggregate SECOND_AGGREGATE] [--add-self-loop] [--use-norm]
       [--activation ACTIVATION] [--nlayer NLAYER] [--nhid NHID]
       [--nhead NHEAD] [--dropout DROPOUT] [--input-drop INPUT_DROP]
       [--attn-drop ATTN_DROP] [--lr LR] [--wd WD] [--epochs EPOCHS]
       [--n-runs N_RUNS] [--gpu GPU] [--seed SEED] [--patience PATIENCE]
       [--nostdout] [--split SPLIT] [--out-dir OUT_DIR]
```

### Key Parameters

- `--data`: Dataset type (`coauthorship` or `cocitation`)
- `--dataset`: Specific dataset (`dblp`, `cora`, `citeseer`, `pubmed`)
- `--model-name`: Model variant (`UniGCN`, `UniGAT`, `UniGIN`, `UniSAGE`, `UniGCNII`)
- `--first-aggregate`: Hyperedge aggregation (`max`, `sum`, `mean`)
- `--second-aggregate`: Node aggregation (`max`, `sum`, `mean`)
- `--nlayer`: Number of layers
- `--nhid`: Hidden dimension
- `--dropout`: Dropout rate
- `--use-norm`: Use normalization
- `--add-self-loop`: Add self-loops to hypergraph

## 📊 Expected Results

### Semi-supervised Learning

- **Coauthorship (DBLP)**: ~88-90% accuracy
- **Cocitation (Cora)**: ~80-85% accuracy
- **Cocitation (PubMed)**: ~75-80% accuracy

### Deep Networks (UniGCNII)

- **Cocitation (Cora)**: ~72-73% accuracy (32 layers)
- **Cocitation (PubMed)**: ~75-78% accuracy (32 layers)

### Inductive Learning

- **Seen nodes**: ~89% accuracy
- **Unseen nodes**: ~83% accuracy

## 🎨 Visualization and Analysis

### Feature Visualization

Use `view.py` to visualize learned embeddings:

1. Train a model (features are saved during training)
2. Edit paths in `view.py`:
   ```python
   features = torch.load('runs/.../features_epoch_200_UniGCNII_pubmed.pt')
   labels = torch.load('saved_labels_pubmed.pt')
   ```
3. Run:
   ```bash
   python view.py
   ```

### Model Inspection

Check saved models:
```bash
python check.py
```

Edit the model path in `check.py` to inspect your trained models.

### Data Conversion

Convert features for analysis:
```bash
# Batch conversion
python convert.py

# Single file
python convert0.py
```

## 📈 Training Tips

1. **Use normalization** (`--use-norm`) for deep networks
2. **Add self-loops** (`--add-self-loop`) for better connectivity
3. **Tune dropout** based on dataset size
4. **Multiple runs** (`--n-runs=10`) for statistical significance
5. **Early stopping** (`--patience`) to prevent overfitting

## 🔬 Advanced Usage

### Custom Aggregation

Modify aggregation functions:
```bash
python train.py --first-aggregate=max --second-aggregate=sum
```

### Multi-head Attention

For UniGAT:
```bash
python train.py --model-name=UniGAT --nhead=8
```

### Deep Networks

For very deep networks:
```bash
python train_val.py --model-name=UniGCNII --nlayer=64 --use-norm
```

## 📚 Baselines

Baseline results can be found in:
- [HyperSAGE](https://openreview.net/forum?id=cKnKJcTPRcV)
- [HyperGCN](https://github.com/malllabiisc/HyperGCN)
- [MPNN-R](https://github.com/naganandy/G-MPNN-R)

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{ijcai21-UniGNN,
  title     = {UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks},
  author    = {Huang, Jing and Yang, Jie},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  year      = {2021}
}
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Note**: Large data files, model checkpoints, and training outputs are excluded from the repository. Please download datasets separately and run scripts to generate outputs.
