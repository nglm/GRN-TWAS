# Gene Regulatory Network-Driven Transcriptome-Wide Association Studies (GRN-TWAS)

This repository contains the implementation of a novel framework that integrates **tissue-specific gene regulatory networks (GRNs)** into **transcriptome-wide association studies (TWAS)** for studying gene-complex disease associations.

## Key Features
- Utilizes **Findr** for GRN reconstruction from genotype and transcriptome data.
- Predicts gene expression by incorporating both **cis** and **trans** regulatory components.
- Evaluates gene-disease associations using **GWAS summary statistics**.

## Data Sources
This project uses three main data sources:
1. **Genome-wide summary statistics**: GWAS meta-analysis results including **effect sizes**p-values, and **andard errors** oimillions of genetic variants across large-scale case-control studies of coronary artery disease.
2. **eQTL mapping summary statistics**  
   Summary statistics from expression quantitative trait locus (eQTL) mapping that link SNPs to genes based on their regulatory effects. The data includes:
   - **SNP_ID**: Identifier for the genetic variant.
   - **GENE_ID**: Associated gene identifier.
   - **Beta**: Effect size of the SNP on gene expression.

3. **Reference dataset of genotype and gene expression**: Individual-level genotype data and RNA sequencing-based gene expression profiles for the same individuals, sourced from the STARNET dataset.

## Requirements

This repository is compatible with **Python 3.11**. The required Python packages are listed in `requirements.txt`. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
