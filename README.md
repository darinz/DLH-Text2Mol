# DLH-Text2Mol

In this project we aim to replicate the paper [Text2Mol: Cross-Modal Molecular Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/) by Carl Edwards, ChengXiang Zhai, and Heng Ji. The original repository for the paper can be found at [Text2Mol Github Repo](https://github.com/cnedwards/text2mol). This repository contains some modifications.

## Conda Environment Setup

Use the first command to create a new independent environment for the project. Or use the other two commands to remove or update the Conda environment.

```shell
# to create conda environment.
conda env create -f code/requirements.yaml

# to remove conda environment.
conda remove --name text2mol --all

# to update conda environment when some new libraries are added.
conda env update -f code/requirements.yaml --prune
```


## Python Code Description

| Python File      | Description |
| ----------- | ----------- |
| main.py      | Train Text2Mol.       |
| main_parallel.py   | A lightly-tested parallel version.        |
| ranker.py   | Rank output embeddings.        |
| ensemble.py   | Rank ensemble of output embeddings.        |
| test_example.py   | Runs a version of the model that you can query with arbitrary inputs for testing.        |
| extract_embeddings.py   | Extract embeddings or rules from a specific checkpoint.        |
| ranker_threshold.py   | Rank output embeddings and plot cosine score vs. ranking.        |
| models.py   | The three model definitions: MLP, GCN, and Attention.        |
| losses.py   | Losses used for training.        |
| dataloaders.py   | Code for loading the data.        |


## Training, Embedding Extraction, Ranking, etc.:

To train the model:

> python code/main.py --data data --output_path test_output --model MLP --epochs 40 --batch_size 32

ranker.py can be used to rank embedding outpoints. ensemble.py ranks the ensemble of multiple embeddings.  

> python code/ranker.py test_output/embeddings --train --val --test

> python code/ensemble.py test_output/embeddings GCN_outputs/embeddings --train --val --test

To run example queries given a model checkpoint for the MLP model:

> python code/test_example.py test_output/embeddings/ data/ test_output/CHECKPOINT.pt

To get embeddings from a specific checkpoint:

> python code/extract_embeddings.py --data data --output_path embedding_output_dir --checkpoint test_output/CHECKPOINT.pt --model MLP --batch_size 32

To plot cosine score vs ranking:

> python code/ranker_threshold.py test_output/embeddings --train --val --test --output_file threshold_image.png


## Citation

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
