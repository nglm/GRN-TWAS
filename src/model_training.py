"""
model_training.py
-----------------
Module for training Ridge regression models for gene expression prediction using GRN structure.
Contains functions for loading graphs, extracting data, training models, and optimizing weights.
"""
import argparse
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import networkx as nx
import sys
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

def load_graph(graph_file: str) -> nx.DiGraph:
    """
    Load a directed acyclic graph (DAG) from a pickle file.

    The graph to load should be a NetworkX DiGraph where each node represents a
    gene, and edges represent regulatory relationships. Each node should have
    an attribute 'cis_snps' which is a list of SNP identifiers associated with
    that gene.

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

    if not isinstance(graph, nx.DiGraph):
        raise RuntimeError(f"Loaded object is not a NetworkX DiGraph: {type(graph)}")

    print(f"Loaded graph from {graph_file} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def get_y_data(
    data: pd.DataFrame,
    gene_id: str,
    samples: List[str]
) -> np.ndarray:
    """
    Extract gene expression data for a specific gene and samples.

    Args:
        data (pd.DataFrame): Expression data. Must contain a column 'id' for gene IDs.
        gene_id (str): Gene identifier.
        samples (list): List of sample IDs.
    Returns:
        np.ndarray: shape (n_samples,) Expression values for the gene across
        samples.
    Raises:
        ValueError: If gene not found or data extraction fails.
    """
    if gene_id not in data['id'].values:
        raise ValueError(f"Gene ID '{gene_id}' not found in expression data")

    gene_data = data[data['id'] == gene_id]
    if gene_data.empty:
        raise ValueError(f"No data found for gene ID '{gene_id}'")

    # Check if all sample columns exist
    missing_samples = [s for s in samples if s not in data.columns]
    if missing_samples:
        raise ValueError(f"Missing sample columns in expression data: {missing_samples}")

    try:
        y = gene_data[samples].to_numpy(dtype='float64')
        y = y.flatten()
    except Exception as e:
        raise ValueError(f"Failed to extract expression data for gene '{gene_id}': {e}")

    if len(y) == 0:
        raise ValueError(f"No expression values found for gene '{gene_id}'")

    return y

def get_x_data(
    data: pd.DataFrame,
    snp_ids: List[str],
    samples: List[str]
) -> np.ndarray:
    """
    Extract genotype data for specific SNPs and samples.
    Args:
        data (pd.DataFrame): Genotype data. Must contain a column 'rs_id' for
        SNP IDs.
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
        print(f"WARNING: {len(missing_snps)} SNPs not found in genotype data: {missing_snps[:5]}{'...' if len(missing_snps) > 5 else ''}")

    try:
        x = available_snps[samples].to_numpy(dtype='float64')
        x = x.T  # Transpose to have samples as rows, SNPs as columns
    except Exception as e:
        raise ValueError(f"Failed to extract genotype data: {e}")

    if x.size == 0:
        raise ValueError("No genotype data extracted")

    return x

def train_ridge(
    X_cis: np.ndarray,
    X_trans: Optional[np.ndarray],
    y: np.ndarray
) -> Tuple[RidgeCV, Optional[RidgeCV]]:
    """
    Train Ridge regression models for cis and trans components.

    Use RidgeCV from sklearn to train models with built-in cross-validation
    for selecting the regularization parameter alpha. alphas are set to a
    logspace from 0.001 to 1000 with 10 values.

    The first returned model, `cis_model`, is a RidgeCV instance trained on
    `X_cis` fitted to y.

    The second returned model, `trans_model`, is None if `X_trans` is not
    provided and otherwise a RidgeCV instance trained on `X_trans` fitted to
    the residuals, that is `y - cis_model.predict(X_cis)`.

    Args:
        X_cis (np.ndarray): Cis genotype matrix. Shape `(n_samples, n_cis_snps)`.
        X_trans (np.ndarray or None): Trans genotype matrix. Shape `(n_samples, n_trans_snps)`.
        y (np.ndarray): Gene expression vector. Shape `(n_samples,)`.
    Returns:
        tuple: (cis_model, trans_model). cis_model is a RidgeCV instance trained on cis data fitted to y. trans_model is None if X_trans is not provided and otherwise a RidgeCV instance trained on trans data fitted to the residuals, that is y - cis_model.predict(X_cis).
    Raises:
        ValueError: If input arrays have invalid shapes or data.
        RuntimeError: If model training fails.
    """
    # ------------------ Validate inputs ------------------
    if X_cis.size == 0:
        raise ValueError("Cis genotype matrix is empty")
    if y.size == 0:
        raise ValueError("Expression vector is empty")
    if X_cis.shape[0] != y.shape[0]:
        raise ValueError(f"Sample dimension mismatch: X_cis has {X_cis.shape[0]} samples, y has {y.shape[0]} samples")

    if X_trans is not None and X_trans.shape[0] != y.shape[0]:
        raise ValueError(f"Sample dimension mismatch: X_trans has {X_trans.shape[0]} samples, y has {y.shape[0]} samples")

    # ---------- Train Ridge models for cis component -----------
    try:
        start_time = time()
        cis_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_cis, y)
        residuals = y - cis_model.predict(X_cis)
        print(f"Training cis model took {time() - start_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to train cis Ridge model: {e}")

    # ------- Train Ridge regression for trans component if available --------
    trans_model = None
    if X_trans is not None:
        try:
            start_time = time()
            trans_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_trans, residuals)
            print(f"Training trans model took {time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"WARNING: Failed to train trans Ridge model: {e}")
            trans_model = None

    # Return both models
    return cis_model, trans_model

def optimize_weights(
    y: np.ndarray,
    X_cis: np.ndarray,
    X_trans: Optional[np.ndarray],
    cis_model: RidgeCV,
    trans_model: Optional[RidgeCV]
) -> np.ndarray:
    """
    Optimize weights for combining cis and trans predictions.

    Use scipy's minimize function to find weights that minimize the negative
    R^2 score of the combined prediction against the true expression values y.
    The optimization method used is 'L-BFGS-B' with bounds [0, 1] for each weight.

    If no trans model is provided, returns weights [1.0, 0.0] to use only cis predictions.

    Args:
        y (np.ndarray): Gene expression vector. Shape `(n_samples,)`.
        X_cis (np.ndarray): Cis genotype matrix. Shape `(n_samples, n_cis_snps)`.
        X_trans (np.ndarray): Trans genotype matrix. Shape `(n_samples, n_trans_snps)`.
        cis_model: Trained Ridge model for cis component. See `train_ridge` function for details.
        trans_model: Trained Ridge model for trans component. See `train_ridge` function for details.
    Returns:
        np.ndarray: Optimized weights for cis and trans predictions.
    """
    # ------------------ Validate inputs ------------------
    if trans_model is None or X_trans is None:
        # If no trans model is available, use only cis predictions
        return np.array([1.0, 0.0])

    # Objective function for weight optimization
    def objective(weights: Union[list, np.ndarray]) -> float:
        """
        Compute negative R^2 score for given weights.

        First computes the weighted predictions from cis and trans models.
        Then computes the (negative) R^2 score against true expression values y.

        The whole formula to compute the prediction is:
        y_pred = w_cis * cis_model.predict(X_cis)
        + w_trans * trans_model.predict(X_trans)

        Args:
            weights (list): Weights for cis and trans predictions.
        """
        try:
            w_cis, w_trans = weights
            y_pred = w_cis * cis_model.predict(X_cis)
            if trans_model is not None:
                y_pred += w_trans * trans_model.predict(X_trans)
            # Negative R^2 for minimization
            return -r2_score(y, y_pred)
        except Exception:
            return float('inf')  # Return large value if prediction fails

    # Initial guess and bounds for weights
    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]

    # ----------------- Optimize weights -----------------
    try:
        # Use scipy minimize to find optimal weights
        result = minimize(
            objective, initial_weights, bounds=bounds, method='L-BFGS-B'
        )
        if result.success:
            return result.x
        else:
            print("WARNING: Weight optimization failed, using default weights")
            return np.array([0.5, 0.5])
    except Exception as e:
        print(f"WARNING: Weight optimization error: {e}, using default weights")
        return np.array([0.5, 0.5])

def process_dataset(
    expression_file: str,
    genotype_file: str,
    graph_file: str,
    output_folder: str
) -> None:
    """
    Process a single dataset to train Ridge regression models for gene expression prediction.

    The genotype file must be a CSV-like file with a column 'rs_id' for SNP
    identifiers and additional columns for each sample containing genotype
    values (0, 1, 2). The genotype file may contain additional columns such as
    'chromosome', 'position', 'ref', 'alt' which will be ignored. However, it should not contain any other columns that are not sample columns.

    The expression file must be a CSV-like file with a column 'id' for gene
    identifiers and additional columns for each sample containing expression
    values (floats).

    The genotype and expression files must have matching sample columns (same
    names and exactly the same number of samples).

    The graph file must be a pickle file containing a NetworkX DiGraph where
    each node represents a gene and has an attribute 'cis_snps' which is a list
    of SNP identifiers associated with that gene.

    This function will create a file `model_results.pkl` in the output folder containing a dictionary storing information about the trained models and weights for each gene. More precisely, the dictionary will have gene IDs as keys and for each gene, a sub-dictionary with keys:
    - 'cis_model': the trained RidgeCV model for the cis component
    - 'trans_model': the trained RidgeCV model for the trans component (or None if not applicable)
    - 'weights': the optimized weights for combining cis and trans predictions.

    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        graph_file (str): Path to DAG pickle file.
        output_folder (str): Directory to save results.
    Returns:
        None
    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If data format is invalid.
        RuntimeError: If model training fails.
    """
    print(f"Processing dataset: {expression_file}, {genotype_file}")

    # ------------ Validate input files ------------
    if not os.path.isfile(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.isfile(genotype_file):
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    # ------------ Load expression data -----------
    try:
        expression_data = pd.read_csv(expression_file)
        if expression_data.empty:
            raise ValueError("Expression data file is empty")
        if 'id' not in expression_data.columns:
            raise ValueError("Expression data must contain 'id' column")
    except Exception as e:
        raise RuntimeError(f"Failed to load expression data: {e}")

    # ------------ Load genotype data ------------
    try:
        genotype_data = pd.read_csv(genotype_file)
        if genotype_data.empty:
            raise ValueError("Genotype data file is empty")
        if 'rs_id' not in genotype_data.columns:
            raise ValueError("Genotype data must contain 'rs_id' column")
    except Exception as e:
        raise RuntimeError(f"Failed to load genotype data: {e}")

    # ------------ Load graph data ------------
    try:
        graph = load_graph(graph_file)
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")
    except Exception as e:
        raise RuntimeError(f"Failed to load graph: {e}")

    # Extract sample IDs and validate consistency
    exp_samples = [col for col in expression_data.columns if col != 'id']
    gen_samples = [col for col in genotype_data.columns if col not in ['rs_id', 'chromosome', 'position', 'ref', 'alt']]

    if not exp_samples:
        raise ValueError("No sample columns found in expression data")
    if not gen_samples:
        raise ValueError("No sample columns found in genotype data")

    # Use intersection of samples (warn if mismatch)
    samples = list(set(exp_samples) & set(gen_samples))
    if not samples:
        raise ValueError("No common samples found between expression and genotype data")

    if len(samples) < len(exp_samples) or len(samples) < len(gen_samples):
        print(f"WARNING: Using {len(samples)} common samples out of {len(exp_samples)} expression and {len(gen_samples)} genotype samples")

    # Get gene list from graph
    try:
        genes = list(nx.topological_sort(graph))
        if not genes:
            raise ValueError("No genes found in graph")
    except nx.NetworkXError as e:
        raise ValueError(f"Graph is not a valid DAG: {e}")

    print(f"Training models for {len(genes)} genes with {len(samples)} samples")

    # Create output directory
    try:
        os.makedirs(output_folder, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {output_folder}")

    # Store model results for each gene
    results = {}
    successful_genes = 0
    failed_genes = 0

    for gene in genes:
        try:
            # Prepare gene expression data
            y = get_y_data(expression_data, gene, samples)

            # Get cis SNPs for this gene from the graph
            cis_snp_ids = list(graph.nodes[gene].get('cis_snps', []))
            if not cis_snp_ids:
                print(f"WARNING: No cis SNPs found for gene {gene}, skipping")
                failed_genes += 1
                continue

            X_cis = get_x_data(genotype_data, cis_snp_ids, samples)

            # Prepare trans genotype data if parents exist
            parents = list(graph.predecessors(gene))
            X_trans = None
            if parents:
                trans_snp_ids = [snp for parent in parents for snp in graph.nodes[parent].get('cis_snps', [])]
                if trans_snp_ids:
                    X_trans = get_x_data(genotype_data, trans_snp_ids, samples)

            # Train Ridge regression models for cis and trans components
            cis_model, trans_model = train_ridge(X_cis, X_trans, y)

            # Optimize weights for combining cis and trans predictions
            if X_trans is not None and trans_model is not None:
                w_cis, w_trans = optimize_weights(y, X_cis, X_trans, cis_model, trans_model)
            else:
                print("WARNING: No trans model available, using only cis predictions")
                w_cis, w_trans = 1.0, 0.0

            # Store trained models and weights for this gene
            results[gene] = {
                'cis_model': cis_model,
                'trans_model': trans_model,
                'weights': {'w_cis': w_cis, 'w_trans': w_trans},
            }
            successful_genes += 1

        except Exception as e:
            print(f"WARNING: Failed to train model for gene {gene}: {e}")
            failed_genes += 1
            continue

    if not results:
        raise RuntimeError("No models were successfully trained")

    print(f"Successfully trained models for {successful_genes} genes, failed for {failed_genes} genes")

    # ------------- Save all model results to disk -------------
    try:
        output_file = os.path.join(output_folder, "model_results.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model results: {e}")

if __name__ == "__main__":
    """
    Main entry point for model training.
    Parses command-line arguments and runs the model training pipeline.
    """

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train Ridge regression models for gene expression prediction.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the DAG pickle file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the results.")

    args = parser.parse_args()

    try:
        # Ensure output folder exists
        os.makedirs(args.output_folder, exist_ok=True)

        # Run the model training pipeline
        process_dataset(
            expression_file=args.expression_file,
            genotype_file=args.genotype_file,
            graph_file=args.graph_file,
            output_folder=args.output_folder
        )

        print("Model training completed successfully!")

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