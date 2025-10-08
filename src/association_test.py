"""
association_test.py
-------------------
Module for gene-disease association analysis using GRN structure and GWAS summary statistics.
Contains functions for loading graphs, extracting genotype data, calculating Z-scores, and processing associations.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import json
import sys
import os
from typing import Any, Dict, List, Optional

def load_graph(graph_file: str) -> Any:
    """
    Load a directed acyclic graph (DAG) from a pickle file.
    Args:
        graph_file (str): Path to the pickle file containing the graph.
    Returns:
        networkx.DiGraph: Loaded directed acyclic graph.
    Raises:
        FileNotFoundError: If graph file doesn't exist.
        RuntimeError: If graph loading fails.
    """
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    try:
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load graph from {graph_file}: {e}")

    print(f"Loaded graph from {graph_file}")
    return graph

def get_x_data(
    data: pd.DataFrame,
    snp_ids: List[str],
    samples: List[str]
) -> np.ndarray:
    """
    Extract genotype data for specific SNPs and samples.
    Args:
        data (pd.DataFrame): Genotype data.
        snp_ids (list): List of SNP identifiers.
        samples (list): List of sample IDs.
    Returns:
        np.ndarray: Genotype values for the SNPs across samples.
    Raises:
        ValueError: If SNPs not found or data extraction fails.
    """
    if not snp_ids:
        raise ValueError("No SNP IDs provided")

    # Check if required columns exist
    if 'rs_id' not in data.columns:
        raise ValueError("Genotype data must contain 'rs_id' column")

    missing_samples = [s for s in samples if s not in data.columns]
    if missing_samples:
        raise ValueError(f"Missing sample columns in genotype data: {missing_samples}")

    # Find which SNPs are actually in the data
    available_snps = data[data['rs_id'].isin(snp_ids)]
    if available_snps.empty:
        raise ValueError(f"None of the requested SNPs found in genotype data: {snp_ids}")

    found_snps = available_snps['rs_id'].tolist()
    missing_snps = [s for s in snp_ids if s not in found_snps]
    if missing_snps:
        print(f"WARNING: {len(missing_snps)} SNPs not found in genotype data")

    try:
        x = available_snps[samples].to_numpy(dtype='float64')
        x = x.T  # Transpose to have samples as rows, SNPs as columns
    except Exception as e:
        raise ValueError(f"Failed to extract genotype data: {e}")

    if x.size == 0:
        raise ValueError("No genotype data extracted")

    return x

def calculate_z_score(
    alpha: np.ndarray,
    rho: np.ndarray,
    se: np.ndarray,
    X: np.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Calculate Z-score for a gene based on genotype and GWAS data.
    Args:
        alpha (np.ndarray): Effect sizes for SNPs.
        rho (np.ndarray): GWAS effect sizes (logOR).
        se (np.ndarray): Standard errors for GWAS effect sizes.
        X (np.ndarray): Genotype matrix for SNPs.
        epsilon (float): Small value to avoid division by zero.
    Returns:
        float: Calculated Z-score for the gene.
    Raises:
        ValueError: If input arrays have incompatible shapes or invalid data.
        RuntimeError: If Z-score calculation fails.
    """
    # ------------------- Validate inputs -------------------
    if alpha.size == 0:
        raise ValueError("Alpha array is empty")
    if rho.size == 0:
        raise ValueError("Rho array is empty")
    if se.size == 0:
        raise ValueError("Standard error array is empty")
    if X.size == 0:
        raise ValueError("Genotype matrix is empty")

    if len(alpha) != len(rho) or len(alpha) != len(se):
        raise ValueError(f"Array length mismatch: alpha={len(alpha)}, rho={len(rho)}, se={len(se)}")

    if X.shape[1] != len(alpha):
        raise ValueError(f"Genotype matrix has {X.shape[1]} SNPs but alpha has {len(alpha)} elements")

    # Check for invalid values
    if np.any(np.isnan(alpha)) or np.any(np.isnan(rho)) or np.any(np.isnan(se)):
        raise ValueError("Input arrays contain NaN values")

    if np.any(se <= 0):
        print("WARNING: Non-positive standard errors found, setting to epsilon")
        se = np.maximum(se, epsilon)

    # ------------------- Calculate Z-score -------------------
    try:
        # Standardize genotype matrix for variance calculations
        X_standardized = StandardScaler().fit_transform(X)

        if X.shape[1] == 1:
            # Single SNP case: variance calculation is simple
            gene_var = np.var(X_standardized, ddof=1) * alpha[0]**2
        else:
            # Multiple SNPs: use covariance matrix and regularize with epsilon
            snp_cov = np.cov(X_standardized, rowvar=False) + np.eye(X.shape[1]) * epsilon
            gene_var = alpha.reshape(1, -1) @ snp_cov @ alpha.reshape(-1, 1)
            gene_var = gene_var[0, 0]

        # Ensure gene_var is positive
        if gene_var <= 0:
            gene_var = epsilon
            print("WARNING: Non-positive gene variance, using epsilon")

        # Calculate ratio and final Z-score
        ratio = np.std(X_standardized, axis=0) / np.sqrt(gene_var + epsilon)
        z_score = np.sum(alpha * ratio * rho / se)

        # Check for invalid result
        if np.isnan(z_score) or np.isinf(z_score):
            print("WARNING: Invalid Z-score calculated, returning 0.0")
            return 0.0

        return float(z_score)

    except Exception as e:
        raise RuntimeError(f"Failed to calculate Z-score: {e}")

def process_association(
    expression_file: str,
    genotype_file: str,
    graph_file: str,
    gwas_file: str,
    output_file: str
) -> None:
    """
    Perform gene-disease association analysis for a single dataset.

    The GWAS summary statistics file must be a CSV-like file, using tabulations as separators and must contain the columns: 'snpid', 'logOR', and 'se_gc'.

    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        graph_file (str): Path to DAG pickle file.
        gwas_file (str): Path to GWAS summary statistics file.
        output_file (str): Path to save association results.
    Returns:
        None
    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If data format is invalid.
        RuntimeError: If association analysis fails.
    """
    print("Starting gene-disease association analysis")

    # Validate input files
    if not os.path.isfile(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.isfile(genotype_file):
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    if not os.path.isfile(gwas_file):
        raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

    # Load all required input data
    try:
        expression_data = pd.read_csv(expression_file)
        if expression_data.empty:
            raise ValueError("Expression data file is empty")
        if 'id' not in expression_data.columns:
            raise ValueError("Expression data must contain 'id' column")
    except Exception as e:
        raise RuntimeError(f"Failed to load expression data: {e}")

    try:
        genotype_data = pd.read_csv(genotype_file)
        if genotype_data.empty:
            raise ValueError("Genotype data file is empty")
        if 'rs_id' not in genotype_data.columns:
            raise ValueError("Genotype data must contain 'rs_id' column")
    except Exception as e:
        raise RuntimeError(f"Failed to load genotype data: {e}")

    try:
        # Try different separators for GWAS file
        try:
            gwas_data = pd.read_csv(gwas_file, sep='\t')
        except:
            gwas_data = pd.read_csv(gwas_file)

        if gwas_data.empty:
            raise ValueError("GWAS data file is empty")

        required_gwas_cols = ['snpid', 'logOR', 'se_gc']
        missing_cols = [col for col in required_gwas_cols if col not in gwas_data.columns]
        if missing_cols:
            available_cols = list(gwas_data.columns)
            raise ValueError(f"GWAS data missing required columns: {missing_cols}. Available columns: {available_cols}")
    except Exception as e:
        raise RuntimeError(f"Failed to load GWAS data: {e}")

    try:
        graph = load_graph(graph_file)
        if not hasattr(graph, 'nodes') or len(graph.nodes) == 0:
            raise ValueError("Graph has no nodes")
    except Exception as e:
        raise RuntimeError(f"Failed to load graph: {e}")

    # Extract sample IDs and validate consistency
    samples = [col for col in expression_data.columns if col != 'id']
    if not samples:
        raise ValueError("No sample columns found in expression data")

    # Get gene list from graph
    genes = list(graph.nodes)
    if not genes:
        raise ValueError("No genes found in graph")

    print(f"Processing {len(genes)} genes with {len(samples)} samples")
    print(f"GWAS data contains {len(gwas_data)} SNPs")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {output_dir}")

    # Store association results for each gene
    results = {}
    successful_genes = 0
    failed_genes = 0

    for gene in genes:
        try:
            # Get cis SNPs for the gene
            cis_snp_ids = list(graph.nodes[gene].get('cis_snps', []))

            if not cis_snp_ids:
                print(f"WARNING: No cis SNPs found for gene {gene}, skipping")
                results[gene] = {'z_score_cis': 0.0, 'z_score_trans': 0.0, 'z_score': 0.0}
                failed_genes += 1
                continue

            # Check if any cis SNPs exist in genotype data
            X_cis = get_x_data(genotype_data, cis_snp_ids, samples)

            # Extract GWAS effect sizes and standard errors for cis SNPs
            cis_gwas = gwas_data[gwas_data['snpid'].isin(cis_snp_ids)]
            if cis_gwas.empty:
                print(f"WARNING: No GWAS data found for cis SNPs of gene {gene}, skipping")
                results[gene] = {'z_score_cis': 0.0, 'z_score_trans': 0.0, 'z_score': 0.0}
                failed_genes += 1
                continue

            rho_cis = cis_gwas.set_index('snpid')['logOR'].to_dict()
            se_cis = cis_gwas.set_index('snpid')['se_gc'].to_dict()

            # Align arrays (only include SNPs with both genotype and GWAS data)
            aligned_snps = [snp for snp in cis_snp_ids if snp in rho_cis and snp in se_cis]
            if not aligned_snps:
                print(f"WARNING: No aligned SNPs found for gene {gene}, skipping")
                results[gene] = {'z_score_cis': 0.0, 'z_score_trans': 0.0, 'z_score': 0.0}
                failed_genes += 1
                continue

            # TODO! Placeholder: random effect sizes for cis SNPs
            # Use the trained weights from the model, see
            # optimize_weights in model_training.py
            alpha_cis = np.random.rand(len(aligned_snps))
            rho_values_cis = np.array([rho_cis[snp] for snp in aligned_snps])
            se_values_cis = np.array([se_cis[snp] for snp in aligned_snps])

            # Get genotype data for aligned SNPs
            X_cis_aligned = get_x_data(genotype_data, aligned_snps, samples)

            # Calculate cis Z-score
            z_score_cis = calculate_z_score(alpha_cis, rho_values_cis, se_values_cis, X_cis_aligned)

            # Handle trans effects: get parent genes and their cis SNPs
            parents = list(graph.predecessors(gene)) if hasattr(graph, 'predecessors') else []
            z_score_trans = 0.0

            if parents:
                trans_snp_ids = [snp for parent in parents for snp in graph.nodes[parent].get('cis_snps', [])]
                if trans_snp_ids:
                    # Check trans SNPs in GWAS data
                    trans_gwas = gwas_data[gwas_data['snpid'].isin(trans_snp_ids)]
                    if not trans_gwas.empty:
                        X_trans = get_x_data(genotype_data, trans_snp_ids, samples)
                        rho_trans = trans_gwas.set_index('snpid')['logOR'].to_dict()
                        se_trans = trans_gwas.set_index('snpid')['se_gc'].to_dict()

                        # Align trans SNPs
                        aligned_trans_snps = [snp for snp in trans_snp_ids if snp in rho_trans and snp in se_trans]
                        if aligned_trans_snps:
                            # TODO! Placeholder: random effect sizes for trans SNPs
                            # Use the trained weights from the model, see
                            # optimize_weights in model_training.py
                            alpha_trans = np.random.rand(len(aligned_trans_snps))
                            rho_values_trans = np.array([rho_trans[snp] for snp in aligned_trans_snps])
                            se_values_trans = np.array([se_trans[snp] for snp in aligned_trans_snps])

                            X_trans_aligned = get_x_data(genotype_data, aligned_trans_snps, samples)
                            z_score_trans = calculate_z_score(alpha_trans, rho_values_trans, se_values_trans, X_trans_aligned)

            # Combine cis and trans Z-scores
            z_score = z_score_cis + z_score_trans

            # Store results for this gene
            results[gene] = {
                'z_score_cis': float(z_score_cis),
                'z_score_trans': float(z_score_trans),
                'z_score': float(z_score)
            }
            successful_genes += 1

        except Exception as e:
            print(f"WARNING: Failed to process gene {gene}: {e}")
            results[gene] = {'z_score_cis': 0.0, 'z_score_trans': 0.0, 'z_score': 0.0}
            failed_genes += 1
            continue

    if not results:
        raise RuntimeError("No genes were successfully processed")

    print(f"Successfully processed {successful_genes} genes, failed for {failed_genes} genes")

    # Save results to output file in JSON format
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")

if __name__ == "__main__":
    """
    Main entry point for gene-disease association analysis.
    Parses command-line arguments and runs the association analysis pipeline.
    """
    # Parse command-line arguments for input/output files
    parser = argparse.ArgumentParser(description="Gene-disease association analysis.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the DAG pickle file.")
    parser.add_argument("--gwas_file", type=str, required=True, help="Path to the GWAS summary statistics file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the association results.")

    args = parser.parse_args()

    try:
        # Run the association analysis pipeline
        process_association(
            expression_file=args.expression_file,
            genotype_file=args.genotype_file,
            graph_file=args.graph_file,
            gwas_file=args.gwas_file,
            output_file=args.output_file
        )

        print("Association analysis completed successfully!")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Invalid value - {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"ERROR: Permission denied - {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: Runtime error - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error occurred - {e}", file=sys.stderr)
        sys.exit(1)

    # Run the association analysis pipeline
    process_association(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=args.graph_file,
        gwas_file=args.gwas_file,
        output_file=args.output_file
    )
