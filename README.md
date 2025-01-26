# Gene Regulatory Network-Driven Transcriptome-Wide Association Studies (GRN-TWAS)

This repository contains the implementation of a novel framework that integrates **tissue-specific gene regulatory networks (GRNs)** into **transcriptome-wide association studies (TWAS)** for studying gene-complex disease association. 

## Key Features
- Utilizes **Findr** for GRN reconstruction from genotype and transcriptome data [3].
- Predicts gene expression by incorporating both **cis** and **trans** regulatory components.
- Evaluates gene-disease associations using **GWAS summary statistics**.

## Requirements

This repository is compatible with **Python 3.11**. The required Python packages are listed in `requirements.txt`. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt

