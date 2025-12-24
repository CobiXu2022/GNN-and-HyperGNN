# HyperGCN

## Installation

```bash
pip install torch numpy scipy networkx matplotlib
pip install configargparse
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html
```

**Note**: The PyTorch Geometric extensions (torch-scatter, torch-sparse, torch-cluster) need to match your PyTorch version. If the above command fails, check your PyTorch version and install from the [official PyG wheels](https://data.pyg.org/whl/).

## Data Preparation

Download datasets from [HyperGCN repository](https://github.com/malllabiisc/HyperGCN) and copy the `data` directory to this repository.

**Required directory structure:**
```
HyperGCN/
├── data/
│   ├── coauthorship/
│   │   ├── dblp/
│   │   │   ├── features.pickle
│   │   │   ├── hypergraph.pickle
│   │   │   ├── labels.pickle
│   │   │   └── splits/
│   │   │       ├── 1.pickle
│   │   │       ├── 2.pickle
│   │   │       └── ...
│   │   └── acm/
│   ├── cocitation/
│   │   ├── cora/
│   │   ├── citeseer/
│   │   └── pubmed/
│   └── datasets/
│       ├── amazon/
│       ├── actor/
│       └── ...
```

## Usage

### Train Single Model

**Basic usage:**
```bash
python hypergcn.py
```

This uses default settings from `config/config.py`:
- Dataset: `datasets/amazon`
- Split: `1`
- GPU: `3`
- Epochs: `200`

**With command line arguments:**
```bash
python hypergcn.py --mediators True --split 1 --data coauthorship --dataset dblp --gpu 0 --epochs 100
```

**Available command line arguments:**
- `--data`: Dataset type (`coauthorship`, `cocitation`, or `datasets`)
- `--dataset`: Specific dataset name
  - For `coauthorship`: `dblp`, `acm`, etc.
  - For `cocitation`: `cora`, `citeseer`, `pubmed`
  - For `datasets`: `amazon`, `actor`, `pokec`, `twitch`
- `--split`: Train-test split number (usually 1-10, check available splits in `data/{data}/{dataset}/splits/`)
- `--mediators`: Use mediators (True/False)
- `--fast`: FastHyperGCN mode (True/False)
- `--gpu`: GPU ID to use
- `--epochs`: Number of training epochs
- `--depth`: Number of hidden layers
- `--dropout`: Dropout rate
- `--rate`: Learning rate
- `--decay`: Weight decay
- `--seed`: Random seed
- `--model`: Model type (`gcn`, `gat`, `sage`, `cheb`)
- `--optimizer`: Optimizer type (`adam`, `sgd`, `adadelta`)

### Train Multiple Models for Comparison

```bash
python main.py
```

This will train GCN, GAT, SAGE, and Chebyshev models sequentially and save metrics to JSON files (e.g., `amazon_metrics.json`).

**Note**: You can modify `main.py` to change which models are trained or add custom evaluation logic.

### Visualization

```bash
python visualize.py
```

**Configuration**: Edit `visualize.py` to specify:
- Dataset path
- Model type
- Output directory
- Visualization parameters

## Configuration

### Edit `config/config.py`

The main configuration file is `config/config.py`. Key parameters:

```python
# Dataset selection
data = "datasets"          # or "coauthorship", "cocitation"
dataset = "amazon"         # Dataset name (must exist in data/{data}/{dataset}/)

# Hypergraph approximation
mediators = False          # Use mediators (True/False)
fast = False              # FastHyperGCN mode (True/False)
split = 1                 # Train-test split number (check available splits!)

# Hardware settings
gpu = 3                   # GPU ID (use 0 if you have only one GPU)
cuda = True               # Use CUDA (set False for CPU-only)
seed = 5                  # Random seed for reproducibility

# Model architecture
depth = 3                 # Number of hidden layers
dropout = 0.5            # Dropout rate (0.0-1.0)
epochs = 200             # Number of training epochs

# Optimization
rate = 0.01              # Learning rate
decay = 0.005            # Weight decay (L2 regularization)
```

**Important Notes:**
- **Split number**: Make sure the split number exists in `data/{data}/{dataset}/splits/`. Available splits are usually 1-10, but check your dataset.
- **GPU ID**: If you have multiple GPUs, specify which one to use. For single GPU systems, use `gpu = 0`.
- **Dataset path**: The data is loaded from `data/{data}/{dataset}/` relative to the `data/` directory.

### Using YAML Configuration Files

You can also create YAML configuration files in `config/` directory (e.g., `config/datasets.yml`) and use them:

```bash
python hypergcn.py -c config/datasets.yml
```

## Troubleshooting

### Common Issues

1. **"split + X does not exist"**
   - Check available splits: `ls data/{data}/{dataset}/splits/`
   - Update `split` in `config/config.py` or use `--split` argument

2. **CUDA out of memory**
   - Reduce batch size (if applicable)
   - Use a smaller dataset
   - Set `cuda = False` in config to use CPU

3. **ModuleNotFoundError: No module named 'torch_scatter'**
   - Install PyTorch Geometric extensions matching your PyTorch version
   - Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Install from matching wheel: `pip install torch-scatter -f https://data.pyg.org/whl/torch-{version}.html`

4. **Dataset not found**
   - Ensure data directory structure matches the required format
   - Check that `data/{data}/{dataset}/` contains `features.pickle`, `hypergraph.pickle`, `labels.pickle`, and `splits/` directory

## Scripts

- `hypergcn.py` - Simple training script (single model)
- `main.py` - Comprehensive training script (multiple models comparison)
- `visualize.py` - Network visualization
- `model/model.py` - Model management (initialization, training, testing)
- `model/networks.py` - Network definitions (HyperGCN architecture)
- `model/utils.py` - Hypergraph utilities
- `data/data.py` - Data loading and parsing
- `config/config.py` - Configuration parser
