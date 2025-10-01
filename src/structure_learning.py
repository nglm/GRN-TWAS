import os
"""
structure_learning.py
---------------------
Module for reconstructing gene regulatory networks (GRNs) using Findr.
Contains functions for calculating posterior probabilities and building GRNs from expression and genotype data.
"""
import pandas as pd
import numpy as np
import findr
import networkx as nx
import json
from typing import Any, Dict, Optional

def calculate_p_values(
    expression_A: np.ndarray,
    expression_ALL: np.ndarray,
    genotype: np.ndarray,
    method: Any,
    n: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate posterior probabilities using Findr for given expression and genotype data.
    Args:
        expression_A (np.ndarray): Expression matrix for target genes.
        expression_ALL (np.ndarray): Expression matrix for all genes.
        genotype (np.ndarray): Genotype matrix.
        method: Findr library object.
        n (int, optional): Number of genes to consider. If None, use all.
    Returns:
        dict: Dictionary of posterior probabilities and intermediate results.
    """
    # Calculate p0: null model posterior probabilities
    p0_results = method.pij_rank(dt=expression_A, dt2=expression_ALL, nodiag=True)
    p0 = p0_results['p'][:, :n] if n else p0_results['p']

    # Calculate other posteriors using genotype and expression data
    p_other_results = method.pijs_gassist(dg=genotype, dt=expression_A, dt2=expression_ALL, nodiag=True)
    p2 = p_other_results['p2'][:, :n] if n else p_other_results['p2']
    p3 = p_other_results['p3'][:, :n] if n else p_other_results['p3']
    p4 = p_other_results['p4'][:, :n] if n else p_other_results['p4']
    p5 = p_other_results['p5'][:, :n] if n else p_other_results['p5']

    # Combine posteriors for downstream analysis
    p2p3 = p2 * p3
    p2p5 = p2 * p5
    p = 0.5 * (p2p5 + p4)

    # Return all relevant posterior probabilities
    return {
        'p0': p0,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'p5': p5,
        'p2p3': p2p3,
        'p2p5': p2p5,
        'p': p,
    }

def reconstruct_grn(
    input_file: str,
    output_folder: str,
    findr_path: str,
    posterior_threshold: float = 0.75
) -> None:
    """
    Reconstruct a gene regulatory network (GRN) for a single tissue dataset using Findr.
    Args:
        input_file (str): Path to input dataset (CSV).
        output_folder (str): Directory to save GRN results.
        findr_path (str): Path to Findr library.
        posterior_threshold (float): Threshold for posterior probabilities.
    Returns:
        None
    """
    print(f"Reconstructing GRN for dataset: {input_file}")

    # Load the input dataset (expression and genotype data)
    data = pd.read_csv(input_file, compression='gzip')

    # Extract sample IDs and build expression/genotype matrices
    sample_ids = [col for col in data.columns if col != 'id']
    expression_A = data[sample_ids].to_numpy(dtype=np.float64)
    expression_ALL = expression_A  # For simplicity, use the same expression matrix
    genotype = data[sample_ids].to_numpy(dtype=np.float64)

    # Initialize Findr library for GRN inference
    findr_lib = findr.lib(path=findr_path, loglv=6, rs=0, nth=0)

    # Calculate p-values and posterior probabilities for network edges
    posteriors = calculate_p_values(expression_A, expression_ALL, genotype, findr_lib)

    # Filter edges by posterior probability threshold
    filtered_posteriors = np.where(posteriors['p'] >= posterior_threshold, posteriors['p'], 0)

    # Save filtered posterior matrix to disk
    output_file = os.path.join(output_folder, "grn_posteriors.csv.gz")
    pd.DataFrame(filtered_posteriors, columns=sample_ids).to_csv(output_file, compression='gzip', index=False)
    print(f"Filtered posteriors saved to {output_file}")

    # Build adjacency matrix for NetworkX graph construction
    adjacency_matrix = pd.DataFrame(filtered_posteriors, columns=sample_ids, index=sample_ids)

    # Create directed graph from adjacency matrix
    graph = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph)

    # Save graph object to disk
    graph_file = os.path.join(output_folder, "grn_graph.gpickle")
    nx.write_gpickle(graph, graph_file)
    print(f"Graph saved to {graph_file}")

    # Save graph summary statistics (nodes/edges)
    graph_info = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
    }
    graph_info_file = os.path.join(output_folder, "graph_info.json")
    with open(graph_info_file, "w") as f:
        json.dump(graph_info, f, indent=4)
    print(f"Graph info saved to {graph_info_file}")

if __name__ == "__main__":
    """
    Main entry point for GRN reconstruction using Findr.
    Parses command-line arguments and runs the reconstruction pipeline.
    """
    import argparse

    # Parse command-line arguments for input/output files
    parser = argparse.ArgumentParser(description="Step 1: Reconstruct GRNs using Findr.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save GRN reconstruction results.")
    parser.add_argument("--findr_path", type=str, required=True, help="Path to the Findr library.")
    parser.add_argument("--posterior_threshold", type=float, default=0.75, help="Threshold for posterior probabilities.")

    args = parser.parse_args()

    # Run the GRN reconstruction pipeline
    reconstruct_grn(
        input_file=args.input_file,
        output_folder=args.output_folder,
        findr_path=args.findr_path,
        posterior_threshold=args.posterior_threshold
    )
