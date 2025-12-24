# UniGNN

## Installation

```bash
pip install torch torch-geometric scipy path tqdm umap-learn matplotlib
```

## Data Preparation

Download datasets from [HyperGCN](https://github.com/malllabiisc/HyperGCN) and copy the `data` directory to this repository.

## Usage

### Semi-supervised Node Classification

```bash
python train.py --data=coauthorship --dataset=dblp --model-name=UniSAGE
```

### Deep Network Training (with Validation)

```bash
python train_val.py --data=cocitation --dataset=cora \
                    --use-norm --add-self-loop \
                    --model-name=UniGCNII --nlayer=32 \
                    --dropout=0.2 --patience=150 \
                    --epochs=1000 --n-runs=1
```

### Inductive Learning on Evolving Hypergraphs

```bash
python train_evolving.py --data=coauthorship --dataset=dblp --model-name=UniGIN
```

### Data Conversion

```bash
python convert.py      # Batch conversion
python convert0.py     # Single file conversion
```

### Visualization

```bash
python view.py         # Feature visualization (UMAP)
```

Edit feature and label paths in `view.py`.

### Model Inspection

```bash
python check.py
```

Edit model path in `check.py`.

## Configuration Parameters

### Main Parameters

- `--data`: Dataset type (`coauthorship` or `cocitation`)
- `--dataset`: Specific dataset (`dblp`, `cora`, `citeseer`, `pubmed`)
- `--model-name`: Model variant (`UniGCN`, `UniGAT`, `UniGIN`, `UniSAGE`, `UniGCNII`)
- `--first-aggregate`: Hyperedge aggregation (`max`, `sum`, `mean`)
- `--second-aggregate`: Node aggregation (`max`, `sum`, `mean`)
- `--nlayer`: Number of layers
- `--nhid`: Hidden dimension
- `--dropout`: Dropout rate
- `--use-norm`: Use normalization
- `--add-self-loop`: Add self-loops
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--n-runs`: Number of runs
- `--gpu`: GPU ID
- `--seed`: Random seed
- `--patience`: Early stopping patience

### Full Parameter List

```bash
python train.py --help
```

## Scripts

- `train.py` - Semi-supervised node classification
- `train_val.py` - Deep network training (with validation)
- `train_evolving.py` - Inductive learning on evolving hypergraphs
- `prepare.py` - Data preparation
- `convert.py` - Batch data format conversion
- `convert0.py` - Single file conversion
- `view.py` - Feature visualization
- `check.py` - Model checkpoint inspection
- `logger.py` - Logging utilities
- `config.py` - Configuration parser
- `model/UniGNN.py` - Unified model definitions
- `model/HyperGCN.py` - HyperGCN compatibility layer
