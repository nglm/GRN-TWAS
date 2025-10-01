"""
model_training.py
-----------------
Module for training Ridge regression models for gene expression prediction using GRN structure.
Contains functions for loading graphs, extracting data, training models, and optimizing weights.
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

def load_graph(graph_file: str) -> nx.DiGraph:
    """
    Load a directed acyclic graph (DAG) from a pickle file.
    Args:
        graph_file (str): Path to the pickle file containing the graph.
    Returns:
        networkx.DiGraph: Loaded directed acyclic graph.
    """
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"Loaded graph from {graph_file}")
    return graph

def get_y_data(
    data: pd.DataFrame,
    gene_id: str,
    samples: List[str]
) -> np.ndarray:
    """
    Extract gene expression data for a specific gene and samples.
    Args:
        data (pd.DataFrame): Expression data.
        gene_id (str): Gene identifier.
        samples (list): List of sample IDs.
    Returns:
        np.ndarray: Expression values for the gene across samples.
    """
    y = data[data['id'] == gene_id][samples].to_numpy(dtype='float64')
    return y.flatten()

def get_x_data(
    # Extract genotype values for a list of SNPs and samples
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
    """
    x = data[data['rs_id'].isin(snp_ids)][samples].to_numpy(dtype='float64')
    return x.T

def train_ridge(
    X_cis: np.ndarray,
    X_trans: Optional[np.ndarray],
    y: np.ndarray
) -> Tuple[RidgeCV, Optional[RidgeCV]]:
    """
    Train Ridge regression models for cis and trans components.
    Args:
        X_cis (np.ndarray): Cis genotype matrix.
        X_trans (np.ndarray or None): Trans genotype matrix.
        y (np.ndarray): Gene expression vector.
    Returns:
        tuple: (cis_model, trans_model)
    """
    # Train Ridge regression for cis component
    cis_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_cis, y)
    residuals = y - cis_model.predict(X_cis)

    # Train Ridge regression for trans component if available
    if X_trans is not None:
        trans_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_trans, residuals)
    else:
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
    Args:
        y (np.ndarray): Gene expression vector.
        X_cis (np.ndarray): Cis genotype matrix.
        X_trans (np.ndarray): Trans genotype matrix.
        cis_model: Trained Ridge model for cis component.
        trans_model: Trained Ridge model for trans component.
    Returns:
        np.ndarray: Optimized weights for cis and trans predictions.
    """
    # Objective function for weight optimization
    def objective(weights):
        w_cis, w_trans = weights
        y_pred = w_cis * cis_model.predict(X_cis)
        if trans_model is not None:
            y_pred += w_trans * trans_model.predict(X_trans)
        # Negative R^2 for minimization
        return -r2_score(y, y_pred)

    # Initial guess and bounds for weights
    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]
    result = minimize(objective, initial_weights, bounds=bounds)
    return result.x

def process_dataset(
    expression_file: str,
    genotype_file: str,
    graph_file: str,
    output_folder: str
) -> None:
    """
    Process a single dataset to train Ridge regression models for gene expression prediction.
    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        graph_file (str): Path to DAG pickle file.
        output_folder (str): Directory to save results.
    Returns:
        None
    """
    print(f"Processing dataset: {expression_file}, {genotype_file}")

    # Load expression, genotype, and graph data
    expression_data = pd.read_csv(expression_file)
    genotype_data = pd.read_csv(genotype_file)
    graph = load_graph(graph_file)

    # Extract sample IDs and gene list
    samples = [col for col in expression_data.columns if col != 'id']
    genes = list(nx.topological_sort(graph))

    # Store model results for each gene
    results = {}
    for gene in genes:
        # Prepare gene expression and cis genotype data
        y = get_y_data(expression_data, gene, samples)
        cis_snp_ids = list(graph.nodes[gene].get('cis_snps', []))
        X_cis = get_x_data(genotype_data, cis_snp_ids, samples)

        # Prepare trans genotype data if parents exist
        parents = list(graph.predecessors(gene))
        if parents:
            trans_snp_ids = [snp for parent in parents for snp in graph.nodes[parent].get('cis_snps', [])]
            X_trans = get_x_data(genotype_data, trans_snp_ids, samples)
        else:
            X_trans = None

        # Train Ridge regression models for cis and trans components
        cis_model, trans_model = train_ridge(X_cis, X_trans, y)

        # Optimize weights for combining cis and trans predictions
        if X_trans is not None:
            w_cis, w_trans = optimize_weights(y, X_cis, X_trans, cis_model, trans_model)
        else:
            w_cis, w_trans = 1.0, 0.0

        # Store trained models and weights for this gene
        results[gene] = {
            'cis_model': cis_model,
            'trans_model': trans_model,
            'weights': {'w_cis': w_cis, 'w_trans': w_trans},
        }

    # Save all model results to disk
    output_file = os.path.join(output_folder, "model_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    """
    Main entry point for model training.
    Parses command-line arguments and runs the model training pipeline.
    """
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train Ridge regression models for gene expression prediction.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the DAG pickle file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the results.")

    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the model training pipeline
    process_dataset(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=args.graph_file,
        output_folder=args.output_folder
    )
