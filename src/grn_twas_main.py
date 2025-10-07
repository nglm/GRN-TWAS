"""
grn_twas_main.py
----------------
Main pipeline script for GRN-based TWAS analysis.
Coordinates structure learning, model training, and association testing steps.
"""
import os
import argparse
from subprocess import run, CalledProcessError
import sys

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
    Raises:
        FileNotFoundError: If input files don't exist.
        CalledProcessError: If structure learning script fails.
    """
    print("Starting structure learning...")

    # Validate input files exist
    if not os.path.isfile(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.isfile(genotype_file):
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")

    # Create output directory
    structure_output = os.path.join(output_folder, "structure")
    try:
        os.makedirs(structure_output, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {structure_output}")

    graph_file = os.path.join(structure_output, "graph.pkl")

    # Run structure learning script as a subprocess
    try:
        run([
            "python", "structure_learning.py",
            "--expression_file", expression_file,
            "--genotype_file", genotype_file,
            "--output_folder", structure_output
        ], check=True)
    except CalledProcessError as e:
        raise CalledProcessError(
            e.returncode,
            e.cmd,
            f"Structure learning failed with return code {e.returncode}"
        )

    # Verify output file was created
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Expected output file not created: {graph_file}")

    print(f"Structure learning completed. Graph saved to: {graph_file}")
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
    Raises:
        FileNotFoundError: If input files don't exist.
        CalledProcessError: If model training script fails.
    """
    print("Starting model training...")

    # Validate input files exist
    if not os.path.isfile(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.isfile(genotype_file):
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    # Create output directory
    training_output = os.path.join(output_folder, "model_training")
    try:
        os.makedirs(training_output, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {training_output}")

    trained_model_file = os.path.join(training_output, "trained_model.pkl")

    # Run model training script as a subprocess
    try:
        run([
            "python", "model_training.py",
            "--expression_file", expression_file,
            "--genotype_file", genotype_file,
            "--graph_file", graph_file,
            "--output_folder", training_output
        ], check=True)
    except CalledProcessError as e:
        raise CalledProcessError(
            e.returncode,
            e.cmd,
            f"Model training failed with return code {e.returncode}"
        )

    # Verify output file was created
    if not os.path.isfile(trained_model_file):
        raise FileNotFoundError(f"Expected output file not created: {trained_model_file}")

    print(f"Model training completed. Model saved to: {trained_model_file}")
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
    Raises:
        FileNotFoundError: If input files don't exist.
        CalledProcessError: If association testing script fails.
    """
    print("Starting association testing...")

    # Validate input files exist
    if not os.path.isfile(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.isfile(genotype_file):
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    if not os.path.isfile(gwas_file):
        raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

    # Create output directory
    association_output = os.path.join(output_folder, "association")
    try:
        os.makedirs(association_output, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {association_output}")

    association_file = os.path.join(association_output, "association_results.json")

    # Run association test script as a subprocess
    try:
        run([
            "python", "association_test.py",
            "--expression_file", expression_file,
            "--genotype_file", genotype_file,
            "--graph_file", graph_file,
            "--gwas_file", gwas_file,
            "--output_file", association_file
        ], check=True)
    except CalledProcessError as e:
        raise CalledProcessError(
            e.returncode,
            e.cmd,
            f"Association testing failed with return code {e.returncode}"
        )

    # Verify output file was created
    if not os.path.isfile(association_file):
        raise FileNotFoundError(f"Expected output file not created: {association_file}")

    print(f"Association testing completed. Results saved to: {association_file}")
    return association_file

def main() -> None:
    """
    Main entry point for the GRN-based TWAS pipeline.
    Parses command-line arguments and runs all pipeline steps.
    """
    # Parse command-line arguments for pipeline inputs/outputs
    parser = argparse.ArgumentParser(description="Run the integrated pipeline for GRN-based TWAS.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--gwas_file", type=str, required=True, help="Path to the GWAS summary statistics file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save all pipeline outputs.")

    args = parser.parse_args()

    try:
        # Validate all input files exist before starting
        if not os.path.isfile(args.expression_file):
            raise FileNotFoundError(f"Expression file not found: {args.expression_file}")
        if not os.path.isfile(args.genotype_file):
            raise FileNotFoundError(f"Genotype file not found: {args.genotype_file}")
        if not os.path.isfile(args.gwas_file):
            raise FileNotFoundError(f"GWAS file not found: {args.gwas_file}")

        # Ensure output folder exists
        try:
            os.makedirs(args.output_folder, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {args.output_folder}")

        print("=== Starting GRN-TWAS Pipeline ===")
        print(f"Expression file: {args.expression_file}")
        print(f"Genotype file: {args.genotype_file}")
        print(f"GWAS file: {args.gwas_file}")
        print(f"Output folder: {args.output_folder}")
        print()

        # Step 1: Run structure learning and get graph file
        graph_file = run_structure_learning(
            expression_file=args.expression_file,
            genotype_file=args.genotype_file,
            output_folder=args.output_folder
        )

        # Step 2: Run model training using the learned graph
        model_file = run_model_training(
            expression_file=args.expression_file,
            genotype_file=args.genotype_file,
            graph_file=graph_file,
            output_folder=args.output_folder
        )

        # Step 3: Run association testing using trained model and graph
        association_file = run_association_test(
            expression_file=args.expression_file,
            genotype_file=args.genotype_file,
            graph_file=graph_file,
            gwas_file=args.gwas_file,
            output_folder=args.output_folder
        )

        print("=== Pipeline Completed Successfully ===")
        print(f"Graph file: {graph_file}")
        print(f"Model file: {model_file}")
        print(f"Association results: {association_file}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"ERROR: Permission denied - {e}", file=sys.stderr)
        sys.exit(1)
    except CalledProcessError as e:
        print(f"ERROR: Pipeline step failed - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error occurred - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    """
    Main script execution for GRN-based TWAS pipeline.
    """
    main()
