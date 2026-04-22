# DPAS-Graph

DPAS-Graph is a dual-graph framework for cross-dataset spatial RNA-to-protein prediction, designed to improve generalization under leave-one-dataset-out evaluation.

## Overview

DPAS-Graph takes spatial transcriptomic profiles and spatial coordinates as input, and predicts paired protein expression for each spatial spot. The main benchmark in this repository follows a leave-one-dataset-out protocol across paired spatial multi-omics datasets.

## Environment

Create the conda environment with:

```bash
conda create -n DPAS python=3.9 -y
conda activate DPAS
conda env create -f environment.yml
```

## Data

The paired spatial multi-omics datasets used for model training in this project are sourced from:

1. **Tonsil Cancer Add**  
   [10x Genomics dataset](https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard)

2. **Tonsil Cancer**  
   [10x Genomics dataset](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard)

3. **Breast Cancer**  
   [10x Genomics dataset](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard)

4. **GSE263617** (`A1_LN`, `A1_TNSL`, `D1_LN`, and `D1_TNSL`)  
   [NCBI GEO accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617)

The external transcriptomics-only datasets used for model application in this project are sourced from:

1. **Human Lung Cancer**  
   [10x Genomics dataset](https://www.10xgenomics.com/datasets/human-lung-cancer-ffpe-2-standard)

2. **Human Tonsil Section** (`esvq52_nluss5`)  
   [Zenodo record](https://zenodo.org/records/8373756)

After downloading the corresponding raw data from the link above, please run the data processing scripts in the `scripts/data_prep` directory in sequence.

## Model Training

Before training, make sure the paired spatial multi-omics datasets have been preprocessed and registered in a JSON file, for example:

```bash
/root/autodl-tmp/pairs_preprocessed_v3.json
```

Model training is primarily carried out and implemented via `scripts/train/run_lodo.py`.

```bash
python scripts/train/run_lodo.py --specs_json /**Your Root**/pairs_preprocessed_v3.json --names "Breast_cancer,Tonsil_cancer,Tonsil_cancer_Add,A1_LN,A1_TNSL,D1_LN,D1_TNSL" --out_root /**Your Root** --epochs ** --lr_scale ** --log_mode compact --seed **
```
## Notes

- The raw datasets are not redistributed in this repository.
- Users should download the original data from the corresponding public sources.
- Preprocessing, training, and evaluation should follow the protocol described in the paper and repository scripts.

## License

This project is released under the MIT License.
