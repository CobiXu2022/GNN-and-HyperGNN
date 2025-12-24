# HyperGCN

## Installation

```bash
pip install torch numpy scipy networkx matplotlib
```

## Data Preparation

Download datasets from [HyperGCN repository](https://github.com/malllabiisc/HyperGCN) and copy the `data` directory to this repository.

## Usage

### Train Single Model

```bash
python hypergcn.py
```

### Train Multiple Models for Comparison

```bash
python main.py
```

### Visualization

```bash
python visualize.py
```

## Configuration

Edit `config/config.py`:

```python
# Dataset
data = "coauthorship"      # or "cocitation"
dataset = "dblp"          # Dataset name

# Hypergraph approximation
mediators = False          # Use mediators
fast = False              # FastHyperGCN mode
split = 0                 # Train-test split number

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

### Command Line Arguments

`hypergcn.py` supports command line arguments:

```bash
python hypergcn.py --mediators True --split 1 --data coauthorship --dataset dblp
```

Arguments:
- `--mediators`: Use mediators (True/False)
- `--split`: Train-test split number
- `--data`: Dataset type (`coauthorship` or `cocitation`)
- `--dataset`: Specific dataset name
  - coauthorship: `dblp`, `acm`, etc.
  - cocitation: `cora`, `citeseer`, `pubmed`

## Scripts

- `hypergcn.py` - Simple training script (single model)
- `main.py` - Comprehensive training script (multiple models comparison)
- `visualize.py` - Network visualization
- `model/model.py` - Model management
- `model/networks.py` - Network definitions
- `model/utils.py` - Hypergraph utilities
- `data/data.py` - Data loading
- `config/config.py` - Configuration parser
