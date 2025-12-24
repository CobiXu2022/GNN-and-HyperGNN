# UniGNN

## Installation

```bash
pip install torch torch-geometric scipy path tqdm umap-learn matplotlib
```

**Note**: Install PyTorch Geometric extensions if needed:
```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html
```

## Data Preparation

1. Download datasets from [HyperGCN](https://github.com/malllabiisc/HyperGCN)

2. Copy the `data` directory to this repository:
   ```
   UniGNN/
   └── data/
       ├── coauthorship/
       │   ├── dblp/
       │   └── acm/
       └── cocitation/
           ├── cora/
           ├── citeseer/
           └── pubmed/
   ```

3. Each dataset directory should contain:
   - `features.pickle`
   - `hypergraph.pickle`
   - `labels.pickle`
   - `splits/` directory with split files

## Usage

### Semi-supervised Node Classification

**Basic usage:**
```bash
python train.py --data=coauthorship --dataset=dblp --model-name=UniSAGE
```

**With custom parameters:**
```bash
python train.py --data=coauthorship --dataset=dblp --model-name=UniGAT \
                --nlayer=3 --nhid=64 --dropout=0.6 --lr=0.01 \
                --epochs=200 --n-runs=10 --gpu=0
```

**Output**: Average test accuracy across multiple runs, saved to `runs/` directory.

### Deep Network Training (with Validation)

**Train UniGCNII (deep network):**
```bash
python train_val.py --data=cocitation --dataset=cora \
                    --use-norm --add-self-loop \
                    --model-name=UniGCNII --nlayer=32 \
                    --dropout=0.2 --patience=150 \
                    --epochs=1000 --n-runs=1 --gpu=0
```

**Configuration notes:**
- `--use-norm`: Use normalization (recommended for deep networks)
- `--add-self-loop`: Add self-loops to hypergraph
- `--patience`: Early stopping patience (epochs without improvement)
- `--n-runs`: Number of independent runs (for statistical significance)

**Output**: Best validation accuracy and final test accuracy, saved to `runs/` directory.

### Inductive Learning on Evolving Hypergraphs

**Train on evolving hypergraph:**
```bash
python train_evolving.py --data=coauthorship --dataset=dblp \
                         --model-name=UniGIN --gpu=0
```

**Output**: Accuracy on seen and unseen nodes.

### Data Conversion

**Batch conversion (convert multiple .pt files to JSON):**
```bash
python convert.py
```

**Configuration**: Edit `convert.py` to specify:
- Input directory (containing .pt files)
- Output directory (for JSON files)
- File patterns to match

**Single file conversion:**
```bash
python convert0.py
```

**Configuration**: Edit `convert0.py` to specify:
- Input .pt file path
- Output JSON file path

### Visualization

**Feature visualization (UMAP):**
```bash
python view.py
```

**Configuration**: Edit `view.py` to specify:
```python
# Edit these lines in view.py
features = torch.load('runs/.../features_epoch_200_UniGCNII_pubmed.pt')
labels = torch.load('saved_labels_pubmed.pt')
```

**Output**: UMAP visualization showing learned embeddings colored by class labels.

### Model Inspection

**Check model checkpoint:**
```bash
python check.py
```

**Configuration**: Edit `check.py` to specify:
```python
# Edit this line in check.py
model_path = 'runs/.../model.pt'
```

**Output**: Model state dict structure, parameter shapes, and total parameter count.

## Configuration Parameters

### Main Parameters

All parameters can be set via command line arguments:

- `--data`: Dataset type (`coauthorship` or `cocitation`)
- `--dataset`: Specific dataset (`dblp`, `cora`, `citeseer`, `pubmed`)
- `--model-name`: Model variant (`UniGCN`, `UniGAT`, `UniGIN`, `UniSAGE`, `UniGCNII`)
- `--first-aggregate`: Hyperedge aggregation (`max`, `sum`, `mean`)
- `--second-aggregate`: Node aggregation (`max`, `sum`, `mean`)
- `--nlayer`: Number of layers (default: 2)
- `--nhid`: Hidden dimension (default: 8, note: actual hidden size is nhid × nhead)
- `--nhead`: Number of attention heads (default: 8, for UniGAT)
- `--dropout`: Dropout rate (default: 0.6)
- `--input-drop`: Input dropout rate (default: 0.6)
- `--attn-drop`: Attention dropout rate (default: 0.6, for UniGAT)
- `--use-norm`: Use normalization (flag, no value needed)
- `--add-self-loop`: Add self-loops (flag, no value needed)
- `--lr`: Learning rate (default: 0.01)
- `--wd`: Weight decay (default: 5e-4)
- `--epochs`: Number of training epochs (default: 200)
- `--n-runs`: Number of runs (default: 1)
- `--gpu`: GPU ID (default: 0)
- `--seed`: Random seed (default: 1)
- `--patience`: Early stopping patience (default: 200)
- `--split`: Train-test split number (default: 1)
- `--out-dir`: Output directory (default: `runs`)

### Full Parameter List

To see all available parameters:

```bash
python train.py --help
```

### Default Configuration

Default values are set in `config.py`. You can modify them there or override via command line.

## Output Locations

- **Training outputs**: `runs/{model_name}_{nlayer}_{data}_{dataset}/seed_{seed}/`
  - Model checkpoints
  - Feature embeddings (for visualization)
  - Logging files
- **Saved labels**: `labels/saved_labels_{dataset}.pt`
- **Results**: `{dataset}_results/` (for some scripts)

## Scripts

- `train.py` - Semi-supervised node classification
- `train_val.py` - Deep network training (with validation)
- `train_evolving.py` - Inductive learning on evolving hypergraphs
- `prepare.py` - Data preparation utilities
- `convert.py` - Batch data format conversion (.pt to .json)
- `convert0.py` - Single file conversion (.pt to .json)
- `view.py` - Feature visualization (UMAP)
- `check.py` - Model checkpoint inspection
- `logger.py` - Logging utilities
- `config.py` - Configuration parser (command-line arguments)
- `model/UniGNN.py` - Unified model definitions (UniGCN, UniGAT, UniGIN, UniSAGE, UniGCNII)
- `model/HyperGCN.py` - HyperGCN compatibility layer
