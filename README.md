# DPAS-Graph

DPAS-Graph is a dual-graph framework for cross-dataset spatial RNA-to-protein prediction, designed to improve generalization under leave-one-dataset-out evaluation.

## Overview

DPAS-Graph takes spatial transcriptomic profiles and spatial coordinates as input and predicts paired protein expression for each spatial spot.  
The main benchmark in this repository follows a leave-one-dataset-out protocol across paired spatial multi-omics datasets.

## Environment

Create the conda environment with:

```bash
conda create -n DPAS python=3.9 -y
conda activate DPAS
pip install -r requirements.txt
