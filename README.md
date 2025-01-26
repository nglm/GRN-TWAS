# Gene Regulatory Network-Driven Transcriptome-Wide Association Studies (GRN-TWAS)

This repository contains the implementation of a novel framework that integrates **tissue-specific gene regulatory networks (GRNs)** into **transcriptome-wide association studies (TWAS)** for studying gene-complex disease associations.

## Key Features
- Utilizes **Findr** for GRN reconstruction from genotype and transcriptome data.
- Predicts gene expression by incorporating both **cis** and **trans** regulatory components.
- Evaluates gene-disease associations using **GWAS summary statistics**.

## Data Sources
This project uses three main data sources:
1. **Genome-wide summary statistics**  
   GWAS meta-analysis results, which include data on millions of genetic variants with information such as:
   - **snpid** Identifier for genetic variant
   - **logOR** Log odds ratio representing effect sizes, if other statistics used (e.g., beta values), change association code accordingly
   - **se_gc**  The standard error of the estimated effect size
   - p-values and adjusted p-values
2. **eQTL mapping summary statistics**  
   Summary statistics from expression quantitative trait locus (eQTL) mapping that link SNPs to genes based on their regulatory effects. The data includes:
   - **SNP_ID**: Identifier for the genetic variant.
   - **GENE_ID**: Associated gene identifier.
   - **Beta**: Effect size of the SNP on gene expression.

3. **Reference dataset of genotype and gene expression**: Individual-level genotype data and RNA sequencing-based gene expression profiles for the same individuals

## Requirements

This repository is compatible with **Python 3.11**. The required Python packages are listed in `requirements.txt`. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt


## GRN-TWAS Pipeline

This repository includes a Python pipeline that automates the three main stages of our framework:

1. **Network Reconstruction**: 
   - Reconstructs tissue-specific gene regulatory networks (GRNs) using reference genotype and gene expression data.
   - Leverages tools like `Findr` for causal inference to build GRNs.

2. **Model Training**: 
   - Trains a machine learning model (e.g., Ridge regression) to predict gene expression by incorporating cis- and trans-eQTL regulatory effects derived from GRNs.

3. **Association Analysis**: 
   - Integrates GWAS summary statistics with predicted gene expression to evaluate gene-disease associations.

### Running the Pipeline

To execute the full pipeline, ensure the following inputs are prepared:
- **Reference Dataset**: A file containing genotype and gene expression data (`reference_dataset.tsv`).
- **GWAS Summary Statistics**: A file with genome-wide summary statistics (`gwas_summary_statistics.tsv`).

Run the pipeline script:

```bash
python pipeline.py