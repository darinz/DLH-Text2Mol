# DLH-Text2Mol

In this project we aim to replicate the paper [Text2Mol: Cross-Modal Molecular Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/) by Carl Edwards, ChengXiang Zhai, and Heng Ji. The original repository for the paper can be found at [Text2Mol Github Repo](https://github.com/cnedwards/text2mol).

## Video Presentation

Watch the video presentation on [YouTube](https://youtu.be/6A5zjoiE10Y).

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


## Data: *ChEBI-20*

Data is located in the "data" directory. Files directly used in the dataloaders are "training.txt", "val.txt", and "test.txt". These include the CIDs (pubchem compound IDs), mol2vec embeddings, and ChEBI descriptions. See README within the data directory for more information.

The data directory contain 6 files:

(1,2,3) The mol2vec_ChEBI_20_X.txt files have lines in the following form:
```
CID	mol2vec embedding	Description
```

(4) mol_graphs.zip contain {cid}.graph files. These are formatted first with the edgelist of the graph and then substructure tokens for each node.
For example,
```
edgelist:
0 1
1 0
1 2
2 1
1 3
3 1
```
```
idx to identifier:
0 3537119515
1 2059730245
2 3537119515
3 1248171218
```

(5) ChEBI_defintions_substructure_corpus.cp contains the molecule token "sentences". It is formatted:
```
cid: tokenid1 tokenid2 tokenid3 ... tokenidn
```

(6) token_embedding_dict.npy is a dictionary mapping molecule tokens to their embeddings. It can be loaded with the following code:
```python
import numpy as np
token_embedding_dict = np.load("token_embedding_dict.npy", allow_pickle=True)[()]
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
