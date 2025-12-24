# aml-elliptic-gnn - AML Detection on Elliptic Dataset

## Installation

```bash
pip install -r requirements.txt
pip install plotly seaborn matplotlib networkx umap-learn
```

## Data Preparation

Download the Elliptic dataset and extract it to the `data/` directory.

Required files:
- `edgelist.csv`: Transaction graph edges
- `features.csv`: Node features
- `classes.csv`: Transaction labels

## Usage

### Train All Models

```bash
python main.py
```

### Custom Data Path

```bash
python main.py --data /path/to/elliptic/data
```

### Inference

```bash
python infer.py
```

Edit model path and data path in `infer.py`.

### Visualization

```bash
python visualize.py          # 2D network visualization
python 3dvisualize.py        # 3D subgraph visualization
python animation.py          # Training animation
```

### Data Analysis

```bash
cd data/
python distribution.py    # Feature distributions
python pca_2d.py         # 2D PCA
python pca_3d.py         # 3D PCA
python heatmap.py        # Correlation heatmap
python boxplot.py        # Boxplots
```

## Configuration

Edit `config.yaml`:

```yaml
data_path: /workspace/data
use_cuda: True

hidden_units: 110              # Hidden units for aggregated features
hidden_units_noAgg: 64         # Hidden units for non-aggregated features
epochs: 13000                   # Number of training epochs
num_classes: 2                  # Number of classes
lr: 9e-3                        # Learning rate
weight_decay: 5e-4              # Weight decay
```

## Scripts

- `main.py` - Main training script (trains all models)
- `train.py` - Training utility functions
- `loader.py` - Data loading (full features)
- `loader_sub.py` - Data loading (subgraph)
- `infer.py` - Model inference
- `visualize.py` - 2D network visualization
- `3dvisualize.py` - 3D subgraph visualization
- `animation.py` - Training animation
- `board.py` - TensorBoard utilities
- `metrics.py` - Metrics calculation
- `models/models.py` - Standard GNN models
- `models/models_cp.py` - Model copies/variants
- `models/custom_gat/` - Custom GAT implementation
