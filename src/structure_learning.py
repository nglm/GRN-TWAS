"""
structure_learning.py
---------------------
Module for reconstructing gene regulatory networks (GRNs) using Findr.
Contains functions for calculating posterior probabilities and building GRNs from expression and genotype data.
"""
import argparse
import os
import pandas as pd
import numpy as np
import findr
import networkx as nx
import json
import sys
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
    Raises:
        ValueError: If input arrays have incompatible shapes or invalid data.
        RuntimeError: If Findr calculations fail.
    """
    # Validate input arrays
    if expression_A.size == 0:
        raise ValueError("Expression matrix A is empty")
    if expression_ALL.size == 0:
        raise ValueError("Expression matrix ALL is empty")
    if genotype.size == 0:
        raise ValueError("Genotype matrix is empty")

    if expression_A.shape[1] != expression_ALL.shape[1]:
        raise ValueError(f"Sample dimension mismatch: expression_A has {expression_A.shape[1]} samples, "
                        f"expression_ALL has {expression_ALL.shape[1]} samples")

    if expression_A.shape[1] != genotype.shape[1]:
        raise ValueError(f"Sample dimension mismatch: expression has {expression_A.shape[1]} samples, "
                        f"genotype has {genotype.shape[1]} samples")

    try:
        # Calculate p0: null model posterior probabilities
        #
        # See section 5.2.1 of the Findr documentation
        # "Inference of pairwise regulation posterior probabilities"
        #
        # The `pij_rank` functions is used for pairwise regulation probability
        # inference when only pairwise expression data is available.
        #
        # dt parameter:
        # -----------------
        # Input matrix of expression levels of A. Element [i,j] is the
        # expression level of gene i of sample j. The matrix has
        # dimension (n_features, n_samples).
        #
        # dt2 parameter:
        # -----------------
        # Input matrix of expression levels of B. Element [i,j] is the
        # expression level of gene i of sample j. The matrix has
        # dimension (n_features2, n_samples).
        #
        # nodiag parameter:
        # -----------------
        # When A (dt param) and B (dt2 param) are the same, log likelihood
        # ratio between alternative and null hypotheses gives infinity.
        # To avoid its contamination in the conversion from log
        # likelihood ratios into probabilities, users need to arrange
        # data accordingly, when A and B are the same or when A is a subset of
        # B. The top submatrix of B’s expression data must be identical with A,
        # and nodiag must be set to True. Otherwise, in the default
        # configuration, A and B should not have any intersection and
        # nodiag = False.
        p0_results = method.pij_rank(dt=expression_A, dt2=expression_ALL, nodiag=True)
        p0 = p0_results['p'][:, :n] if n else p0_results['p']
    except Exception as e:
        raise RuntimeError(f"Failed to calculate null model posteriors: {e}")

    try:
        # Calculate other posteriors using genotype and expression data
        #
        # See section 5.2.1 of the Findr documentation
        # "Inference of pairwise regulation posterior probabilities"
        #
        # The `pijs_gassist` functions is used for pairwise regulation
        # probability inference when discrete causal anchor data available.
        # Findr performs 5 tests for causal inference A → B. The 5 p-values
        # then allow arbitrary combination by the user.
        #
        # `dg` parameter:
        # -----------------
        # Input matrix of best eQTL genotype data E(A), each row of which is the best
        # eQTL of the corresponding row of dt. Element [i,j] is the genotype value
        # of the best eQTL of gene i of sample j, and should be among values
        # 0, 1, . . . , na. The matrix has dimension (n_features, n_samples).
        p_other_results = method.pijs_gassist(dg=genotype, dt=expression_A, dt2=expression_ALL, nodiag=True)
        p2 = p_other_results['p2'][:, :n] if n else p_other_results['p2']
        p3 = p_other_results['p3'][:, :n] if n else p_other_results['p3']
        p4 = p_other_results['p4'][:, :n] if n else p_other_results['p4']
        p5 = p_other_results['p5'][:, :n] if n else p_other_results['p5']
    except Exception as e:
        raise RuntimeError(f"Failed to calculate genotype-assisted posteriors: {e}")

    # Combine posteriors for downstream analysis
    p2p3 = p2 * p3
    p2p5 = p2 * p5

    # See "4.3 Subtest combination" in Findr tutorial
    #
    # Findr computes the final probability of regulation by combining the
    # subtests as:
    #                          p = 0.5 * (p2 * p5 + p4)
    # By combining the secondary linkage and controlled tests, the first term
    # verifies that the correlation between g_i and g_j is not entirely due to
    # pleiotropy. By replacing the conditional independence test in [9] with
    # the controlled test, this combination is robust against hidden confounders
    # and technical variations. On the other hand, the relevance test in the
    # second term can identify interactions that arise from the indirect effect
    # e_i → g_i → g_j but are too weak to be detected by the secondary linkage
    # test. However, in such cases the direction of regulation cannot be
    # determined. The coefficient 0.5 simply assigns half of the probability
    # to each direction.
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
    Reconstruct a gene regulatory network (GRN) for a dataset using Findr.

    To reconstruct a GRN, posterior probabilities for potential regulatory edges are calculated using expression and genotype data. Edges are then filtered based on the specified posterior probabiblity threshold.

    The input dataset must be a CSV-like file, optionally gzipped, with the first column as 'id' (gene/SNP IDs) and subsequent columns as samples. No other columns should be present.

    The Findr library must be properly installed and accessible at the specified path.

    The resulting network is saved as a NetworkX graph object in the output folder along with summary statistics.
    - `grn_posteriors.csv.gz`: Filtered posterior probability matrix.
    - `grn_graph.gpickle`: Serialized NetworkX graph object.
    - `graph_info.json`: Summary statistics of the graph. Available keys are:
        - `total_nodes`: Total number of nodes in the graph.
        - `total_edges`: Total number of edges in the graph.
        - `posterior_threshold`: Posterior probability threshold used for filtering edges.
        - `input_file`: Path to the input dataset used.

    Args:
        input_file (str): Path to input dataset (CSV).
        output_folder (str): Directory to save GRN results.
        findr_path (str): Path to Findr library.
        posterior_threshold (float): Threshold for posterior probabilities.
    Returns:
        None
    Raises:
        FileNotFoundError: If input file or Findr library not found.
        ValueError: If data format is invalid or threshold out of range.
        RuntimeError: If GRN reconstruction fails.
    """
    print(f"Reconstructing GRN for dataset: {input_file}")

    # Validate inputs
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input dataset file not found: {input_file}")

    if not os.path.exists(findr_path):
        raise FileNotFoundError(f"Findr library path not found: {findr_path}")

    if not (0.0 <= posterior_threshold <= 1.0):
        raise ValueError(f"Posterior threshold must be between 0 and 1, got: {posterior_threshold}")

    # Load the input dataset (expression and genotype data)
    try:
        if input_file.endswith('.gz'):
            data = pd.read_csv(input_file, compression='gzip')
        else:
            data = pd.read_csv(input_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load input dataset: {e}")

    # Validate data format
    if data.empty:
        raise ValueError("Input dataset is empty")

    if 'id' not in data.columns:
        raise ValueError("Input dataset must contain 'id' column")

    # Extract sample IDs and build expression/genotype matrices
    sample_ids = [col for col in data.columns if col != 'id']
    if not sample_ids:
        raise ValueError("No sample columns found in dataset")

    try:
        expression_A = data[sample_ids].to_numpy(dtype=np.float64)
        expression_ALL = expression_A  # For simplicity, use the same expression matrix
        genotype = data[sample_ids].to_numpy(dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Failed to convert data to numeric arrays: {e}")

    # Check for missing values
    if np.any(np.isnan(expression_A)):
        print("WARNING: Found NaN values in expression data")
    if np.any(np.isnan(genotype)):
        print("WARNING: Found NaN values in genotype data")

    # Initialize Findr library for GRN inference
    try:
        findr_lib = findr.lib(path=findr_path, loglv=6, rs=0, nth=0)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Findr library: {e}")

    # Calculate p-values and posterior probabilities for network edges
    try:
        posteriors = calculate_p_values(expression_A, expression_ALL, genotype, findr_lib)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate posterior probabilities: {e}")

    # Filter edges by posterior probability threshold
    filtered_posteriors = np.where(posteriors['p'] >= posterior_threshold, posteriors['p'], 0)

    print(f"Filtered {np.sum(filtered_posteriors > 0)} edges with threshold {posterior_threshold}")

    # Create output directory
    try:
        os.makedirs(output_folder, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {output_folder}")

    # Save filtered posterior matrix to disk
    try:
        output_file = os.path.join(output_folder, "grn_posteriors.csv.gz")
        pd.DataFrame(filtered_posteriors, columns=sample_ids).to_csv(output_file, compression='gzip', index=False)
        print(f"Filtered posteriors saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save posterior matrix: {e}")

    # Build adjacency matrix for NetworkX graph construction
    try:
        adjacency_matrix = pd.DataFrame(filtered_posteriors, columns=sample_ids, index=sample_ids)
        # Create directed graph from adjacency matrix
        graph = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph)
    except Exception as e:
        raise RuntimeError(f"Failed to create graph from adjacency matrix: {e}")

    # Save graph object to disk
    try:
        graph_file = os.path.join(output_folder, "grn_graph.gpickle")
        nx.write_gpickle(graph, graph_file)
        print(f"Graph saved to {graph_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save graph: {e}")

    # Save graph summary statistics (nodes/edges)
    try:
        graph_info = {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "posterior_threshold": posterior_threshold,
            "input_file": input_file
        }
        graph_info_file = os.path.join(output_folder, "graph_info.json")
        with open(graph_info_file, "w") as f:
            json.dump(graph_info, f, indent=4)
        print(f"Graph info saved to {graph_info_file}")
        print(f"Graph created with {graph_info['total_nodes']} nodes and {graph_info['total_edges']} edges")
    except Exception as e:
        raise RuntimeError(f"Failed to save graph info: {e}")

if __name__ == "__main__":
    """
    Main entry point for GRN reconstruction using Findr.
    Parses command-line arguments and runs the reconstruction pipeline.
    """

    # Parse command-line arguments for input/output files
    parser = argparse.ArgumentParser(description="Step 1: Reconstruct GRNs using Findr.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save GRN reconstruction results.")
    parser.add_argument("--findr_path", type=str, required=True, help="Path to the Findr library.")
    parser.add_argument("--posterior_threshold", type=float, default=0.75, help="Threshold for posterior probabilities.")

    args = parser.parse_args()

    try:
        # Validate command line arguments
        if args.posterior_threshold < 0.0 or args.posterior_threshold > 1.0:
            raise ValueError(f"Posterior threshold must be between 0 and 1, got: {args.posterior_threshold}")

        # Run the GRN reconstruction pipeline
        reconstruct_grn(
            input_file=args.input_file,
            output_folder=args.output_folder,
            findr_path=args.findr_path,
            posterior_threshold=args.posterior_threshold
        )

        print("GRN reconstruction completed successfully!")

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
