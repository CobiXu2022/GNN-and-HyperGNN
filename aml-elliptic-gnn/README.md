# aml-elliptic-gnn - AML Detection on Elliptic Dataset

## Installation

```bash
pip install -r requirements.txt
pip install plotly seaborn matplotlib networkx umap-learn
```

**Note**: If `requirements.txt` is missing, install core dependencies:
```bash
pip install torch torch-geometric numpy pandas scikit-learn
pip install plotly seaborn matplotlib networkx umap-learn
```

## Data Preparation

1. Download the Elliptic dataset from the [official link](http://dl.dropboxusercontent.com/scl/fi/2j7nx8y3jbyypdm7r100f/dataset.zip?rlkey=veu69cngj0els6emgp549r06u&dl=0)

2. Extract the zip file and place the following files in the `data/` directory:
   ```
   aml-elliptic-gnn/
   └── data/
       ├── edgelist.csv    # Transaction graph edges
       ├── features.csv    # Node features
       └── classes.csv     # Transaction labels (licit/illicit)
   ```

3. **Alternative**: You can place data in a different location and update the path in `config.yaml` (see Configuration section).

## Usage

### Train All Models

**Basic usage:**
```bash
python main.py
```

This will train all model variants (GCN, GAT, SAGE, Chebyshev, GATv2, Custom GAT) with and without aggregation features.

**Output**: 
- Trained models saved in `saved_models/`
- Metrics saved in `jsonfile/` and `metrics/`
- TensorBoard logs in `runs/`

### Custom Data Path

If your data is in a different location:

```bash
python main.py --data /path/to/elliptic/data
```

**Note**: The `--data` argument overrides `data_path` in `config.yaml`.

### Inference

**Step 1**: Edit `infer.py` to set paths:

```python
# Edit these lines in infer.py
model_path = '/path/to/saved_models/GCN_Convolution_tx_agg.pt'
data_path = '/path/to/data'  # Directory containing edgelist.csv, features.csv, classes.csv
```

**Step 2**: Run inference:

```bash
python infer.py
```

### Visualization

**2D Network Visualization:**
```bash
python visualize.py
```

**Configuration**: Edit `visualize.py` to specify:
- Data path
- Output file name
- Visualization parameters (node size, colors, layout)

**3D Subgraph Visualization:**
```bash
python 3dvisualize.py
```

**Configuration**: Edit `3dvisualize.py` to specify:
- Data path
- Subgraph extraction parameters
- Output HTML file name

**Training Animation:**
```bash
python animation.py
```

**Configuration**: Edit `animation.py` to specify:
- Metrics JSON file path (from `jsonfile/`)
- Output video file name

### Data Analysis

Navigate to the `data/` directory and run analysis scripts:

```bash
cd data/
python distribution.py    # Feature distributions (saves to column_distribution/)
python pca_2d.py         # 2D PCA visualization (saves pca_2d.png)
python pca_3d.py         # 3D PCA visualization (saves pca_3d.html)
python heatmap.py        # Correlation heatmap (saves heatmap.png)
python boxplot.py        # Boxplots (saves boxplot_*.png)
```

**Note**: Edit paths in these scripts if your data is in a different location.

## Configuration

### Edit `config.yaml`

The main configuration file is `config.yaml`:

```yaml
data_path: /workspace/data          # Path to data directory (or use --data argument)
use_cuda: True                      # Use CUDA if available

hidden_units: 110                   # Hidden units for aggregated features
hidden_units_noAgg: 64              # Hidden units for non-aggregated features
epochs: 13000                       # Number of training epochs
num_classes: 2                      # Number of classes (binary classification)
lr: 9e-3                            # Learning rate
weight_decay: 5e-4                  # Weight decay (L2 regularization)
```

**To change data path:**
- Option 1: Edit `data_path` in `config.yaml`
- Option 2: Use command line: `python main.py --data /path/to/data`

**To modify training parameters:**
- Edit values in `config.yaml`
- Or modify `main.py` to override config values

### Model Configuration

Models are configured in `main.py`. You can:
- Add/remove models from the training list
- Modify model architectures in `models/models.py`
- Adjust training parameters per model

## Output Locations

- **Models**: `saved_models/*.pt`
- **Metrics JSON**: `jsonfile/metrics_*.json`
- **Metrics CSV**: `metrics/*.json`, `data/metrics.csv`
- **TensorBoard logs**: `runs/`
- **Visualizations**: 
  - `data/interactive_network.html` (2D network)
  - `data/subgraph_visualization*.html` (3D subgraphs)
  - `data/accuracy_animation.mp4` (training animation)
  - `data/pca_2d.png`, `data/pca_3d.html` (PCA)
  - `data/heatmap.png` (correlation)
  - `data/boxplot_*.png` (boxplots)
  - `data/column_distribution/*.png` (feature distributions)

## Scripts

- `main.py` - Main training script (trains all models)
- `train.py` - Training utility functions
- `loader.py` - Data loading (full features with aggregation)
- `loader_sub.py` - Data loading (subgraph extraction)
- `infer.py` - Model inference
- `visualize.py` - 2D network visualization
- `3dvisualize.py` - 3D subgraph visualization
- `animation.py` - Training animation
- `board.py` - TensorBoard utilities
- `metrics.py` - Metrics calculation
- `models/models.py` - Standard GNN models (GCN, GAT, SAGE, Cheb, GATv2, GIN, GTC)
- `models/models_cp.py` - Model copies/variants
- `models/custom_gat/` - Custom GAT implementation
- `data/distribution.py` - Feature distribution analysis
- `data/pca_2d.py` - 2D PCA visualization
- `data/pca_3d.py` - 3D PCA visualization
- `data/heatmap.py` - Correlation heatmap
- `data/boxplot.py` - Boxplot visualization
