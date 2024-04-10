# DLH-Text2Mol

In this project we aim to replicate the paper [Text2Mol: Cross-Modal Molecular Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/) by Carl Edwards, ChengXiang Zhai, and Heng Ji. The original repository for the paper can be found at [Text2Mol Github Repo](https://github.com/cnedwards/text2mol). This repository contains some modifications.

## Conda Environment Setup

Use the first command to create a new independent environment for the project. Or use the other two commands to remove or update the Conda environment.

```shell
# to create conda environment.
conda env create -f ./code/requirements.yaml

# to remove conda environment.
conda remove --name text2mol --all

# to update conda environment when some new libraries are added.
conda env update -f environment.yaml --prune
```

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