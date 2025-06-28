# DLH-Text2Mol

[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-EMNLP%202021-orange.svg)](https://aclanthology.org/2021.emnlp-main.47/)
[![arXiv](https://img.shields.io/badge/arXiv-2108.02713-b31b1b.svg)](https://arxiv.org/abs/2108.02713)

A PyTorch implementation of [Text2Mol: Cross-Modal Molecular Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/) by Carl Edwards, ChengXiang Zhai, and Heng Ji. This project replicates the original paper's methodology for retrieving molecular structures using natural language queries.

## Overview

Text2Mol is a cross-modal retrieval system that bridges the gap between natural language descriptions and molecular structures. The model learns to map textual descriptions to molecular embeddings, enabling efficient retrieval of relevant molecules based on text queries.

## Features

- **Cross-modal Retrieval**: Map natural language queries to molecular structures
- **Multiple Model Architectures**: MLP, GCN, and Attention-based models
- **ChEBI-20 Dataset**: Comprehensive molecular dataset with textual descriptions
- **Embedding Extraction**: Extract and analyze molecular embeddings
- **Ranking System**: Evaluate retrieval performance with various metrics

## Quick Start

### Environment Setup

Create a new conda environment for the project:

```bash
# Create conda environment
conda env create -f code/requirements.yaml

# Activate environment
conda activate text2mol

# Update environment (if needed)
conda env update -f code/requirements.yaml --prune
```

### Training

Train the Text2Mol model:

```bash
python code/main.py --data data --output_path test_output --model MLP --epochs 40 --batch_size 32
```

### Evaluation

Rank embeddings and evaluate performance:

```bash
# Rank single model outputs
python code/ranker.py test_output/embeddings --train --val --test

# Rank ensemble of multiple models
python code/ensemble.py test_output/embeddings GCN_outputs/embeddings --train --val --test
```

### Testing

Run example queries with a trained model:

```bash
python code/test_example.py test_output/embeddings/ data/ test_output/CHECKPOINT.pt
```

## Dataset

The project uses the **ChEBI-20** dataset located in the `data/` directory. The dataset includes:

- **Training/Validation/Test splits**: `training.txt`, `val.txt`, `test.txt`
- **Molecular graphs**: `mol_graphs.zip` containing graph representations
- **Token embeddings**: `token_embedding_dict.npy` for molecular substructure tokens
- **Corpus data**: `ChEBI_defintions_substructure_corpus.cp` with tokenized descriptions

### Data Format

Each data file contains:
- **CID**: PubChem Compound ID
- **Mol2Vec embeddings**: Pre-computed molecular embeddings
- **ChEBI descriptions**: Natural language descriptions of molecules

## Model Architecture

The implementation includes three model variants:

| Model | Description |
|-------|-------------|
| **MLP** | Multi-layer perceptron for embedding projection |
| **GCN** | Graph Convolutional Network for molecular representation |
| **Attention** | Attention-based model for cross-modal learning |

## Code Structure

| File | Purpose |
|------|---------|
| `main.py` | Main training script |
| `models.py` | Model architecture definitions |
| `dataloaders.py` | Data loading and preprocessing |
| `losses.py` | Loss function implementations |
| `ranker.py` | Embedding ranking and evaluation |
| `ensemble.py` | Ensemble model evaluation |
| `extract_embeddings.py` | Embedding extraction utilities |
| `test_example.py` | Interactive testing interface |
| `ranker_threshold.py` | Threshold analysis and visualization |

## Usage Examples

### Extract Embeddings

```bash
python code/extract_embeddings.py \
    --data data \
    --output_path embedding_output_dir \
    --checkpoint test_output/CHECKPOINT.pt \
    --model MLP \
    --batch_size 32
```

### Analyze Threshold Performance

```bash
python code/ranker_threshold.py test_output/embeddings \
    --train --val --test \
    --output_file threshold_analysis.png
```

## Video Presentation

Watch the project presentation: [YouTube Video](https://youtu.be/6A5zjoiE10Y)

## Dependencies

Key dependencies include:
- **PyTorch 1.11.0**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Transformers 4.15.0**: Pre-trained language models
- **NumPy, Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Scikit-learn**: Machine learning utilities

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{edwards2021text2mol,
  title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
  author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={595--607},
  year={2021},
  url = {https://aclanthology.org/2021.emnlp-main.47/}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original Text2Mol implementation: [Text2Mol GitHub Repository](https://github.com/cnedwards/text2mol)
- ChEBI database for molecular data
- PyTorch and PyTorch Geometric communities
