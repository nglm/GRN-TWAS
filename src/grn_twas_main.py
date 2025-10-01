"""
grn_twas_main.py
----------------
Main pipeline script for GRN-based TWAS analysis.
Coordinates structure learning, model training, and association testing steps.
"""
import os
import argparse
from subprocess import run

def run_structure_learning(
    expression_file: str,
    genotype_file: str,
    output_folder: str
) -> str:
    """
    Run the structure learning step of the pipeline.
    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        output_folder (str): Directory to save structure learning outputs.
    Returns:
        str: Path to saved graph file.
    """
    print("Starting structure learning...")
    structure_output = os.path.join(output_folder, "structure")
    os.makedirs(structure_output, exist_ok=True)
    graph_file = os.path.join(structure_output, "graph.pkl")
    run([
        "python", "structure_learning.py",
        "--expression_file", expression_file,
        "--genotype_file", genotype_file,
        "--output_folder", structure_output
    ], check=True)
    return graph_file

def run_model_training(
    expression_file: str,
    genotype_file: str,
    graph_file: str,
    output_folder: str
) -> str:
    """
    Run the model training step of the pipeline.
    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        graph_file (str): Path to graph file.
        output_folder (str): Directory to save model training outputs.
    Returns:
        str: Path to trained model file.
    """
    print("Starting model training...")
    training_output = os.path.join(output_folder, "model_training")
    os.makedirs(training_output, exist_ok=True)
    trained_model_file = os.path.join(training_output, "trained_model.pkl")
    run([
        "python", "model_training.py",
        "--expression_file", expression_file,
        "--genotype_file", genotype_file,
        "--graph_file", graph_file,
        "--output_folder", training_output
    ], check=True)
    return trained_model_file

def run_association_test(
    expression_file: str,
    genotype_file: str,
    graph_file: str,
    gwas_file: str,
    output_folder: str
) -> str:
    """
    Run the association testing step of the pipeline.
    Args:
        expression_file (str): Path to gene expression file.
        genotype_file (str): Path to genotype file.
        graph_file (str): Path to graph file.
        gwas_file (str): Path to GWAS summary statistics file.
        output_folder (str): Directory to save association results.
    Returns:
        str: Path to association results file.
    """
    print("Starting association testing...")
    association_output = os.path.join(output_folder, "association")
    os.makedirs(association_output, exist_ok=True)
    association_file = os.path.join(association_output, "association_results.json")
    run([
        "python", "association_test.py",
        "--expression_file", expression_file,
        "--genotype_file", genotype_file,
        "--graph_file", graph_file,
        "--gwas_file", gwas_file,
        "--output_file", association_file
    ], check=True)
    return association_file

def main() -> None:
    """
    Main entry point for the GRN-based TWAS pipeline.
    Parses command-line arguments and runs all pipeline steps.
    """
    parser = argparse.ArgumentParser(description="Run the integrated pipeline for GRN-based TWAS.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--gwas_file", type=str, required=True, help="Path to the GWAS summary statistics file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save all pipeline outputs.")

    args = parser.parse_args()

    # Create the output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Step 1: Structure Learning
    graph_file = run_structure_learning(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        output_folder=args.output_folder
    )

    # Step 2: Model Training
    run_model_training(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=graph_file,
        output_folder=args.output_folder
    )

    # Step 3: Association Testing
    association_file = run_association_test(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=graph_file,
        gwas_file=args.gwas_file,
        output_folder=args.output_folder
    )

    print(f"Pipeline completed successfully. Association results saved at: {association_file}")

if __name__ == "__main__":
    """
    Main script execution for GRN-based TWAS pipeline.
    """
    main()
