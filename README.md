# GNN and HyperGNN Projects Collection

This repository contains four comprehensive projects on Graph Neural Networks (GNNs) and Hypergraph Neural Networks (HyperGNNs) for various applications, including anti-money laundering detection and node classification tasks.

## 📁 Projects Overview

### 1. [AML](./AML/) - Anti-Money Laundering Detection with GNN
Graph Attention Network (GAT) based solution for detecting money laundering activities in financial transaction networks.

**Key Features:**
- GAT, GAS, and GATv2 model implementations
- Custom dataset processing for transaction graphs
- Support for Random Forest and XGBoost baselines

### 2. [aml-elliptic-gnn](./aml-elliptic-gnn/) - AML Detection on Elliptic Dataset
Comprehensive comparison of multiple GNN architectures (GCN, GAT, SAGE, Cheb, GATv2) on the Elliptic cryptocurrency transaction dataset.

**Key Features:**
- Multiple GNN model implementations
- Extensive visualization and analysis tools
- Performance comparison across different architectures

### 3. [HyperGCN](./HyperGCN/) - Hypergraph Convolutional Networks
Implementation of HyperGCN for training Graph Convolutional Networks on hypergraphs, published at NeurIPS 2019.

**Key Features:**
- Hypergraph to graph approximation
- Support for multiple datasets (coauthorship, cocitation)
- GCN, GAT, SAGE, and Chebyshev convolution variants

### 4. [UniGNN](./UniGNN/) - Unified Graph and Hypergraph Neural Networks
A unified framework for both graph and hypergraph neural networks, published at IJCAI 2021.

**Key Features:**
- Unified message passing framework
- Support for multiple model variants (UniGCN, UniGAT, UniGIN, UniSAGE, UniGCNII)
- Semi-supervised and inductive learning capabilities


### Prerequisites

- Python >= 3.6
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.1
- CUDA (optional, for GPU acceleration)

### Installation

Each project has its own requirements. Please refer to individual project READMEs for specific installation instructions.

General installation:
```bash
pip install torch torch-geometric numpy pandas scikit-learn matplotlib
```

## 📊 Project Details

| Project | Task | Models | Datasets |
|---------|------|--------|----------|
| AML | Money Laundering Detection | GAT, GAS, GATv2 | IBM Transactions |
| aml-elliptic-gnn | AML Detection | GCN, GAT, SAGE, Cheb, GATv2 | Elliptic |
| HyperGCN | Node Classification | GCN, GAT, SAGE, Cheb | Coauthorship, Cocitation |
| UniGNN | Node Classification | UniGCN, UniGAT, UniGIN, UniSAGE | Coauthorship, Cocitation |

## 📝 Usage

Each project contains detailed README files with specific usage instructions:

- [AML README](./AML/README.md)
- [aml-elliptic-gnn README](./aml-elliptic-gnn/README.md)
- [HyperGCN README](./HyperGCN/README.md)
- [UniGNN README](./UniGNN/README.md)


### Datasets

- [IBM Transactions for Anti-Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Hypergraph datasets from [HyperGCN repository](https://github.com/malllabiisc/HyperGCN)



**Note**: Large datasets and model files are excluded from this repository via `.gitignore`. Please download datasets separately as instructed in each project's README.

