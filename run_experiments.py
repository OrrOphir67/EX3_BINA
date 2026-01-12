"""
Main script to run Perceptron experiments for digit classification.

This script implements Question 5 of the assignment:
- Trains two variants of Multi-Class Perceptron
- Reports training errors per epoch
- Reports test errors
- Generates confusion matrices
"""

import numpy as np
from perceptron import MultiClassPerceptron
from data_loader import (
    load_training_data,
    load_test_data,
    print_data_summary
)
from evaluate import (
    print_evaluation_results,
    compare_variants,
    save_confusion_matrix_to_file
)


def run_experiment(X_train, y_train, X_test, y_test,
                   learning_rate=1.0, decay_rate=None,
                   n_epochs=4, variant_name="Variant"):
    """
    Run a single perceptron experiment.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        learning_rate: Initial learning rate
        decay_rate: Learning rate decay factor (None for fixed)
        n_epochs: Number of training epochs
        variant_name: Name for this experiment

    Returns:
        tuple: (model, training_history, test_results)
    """
    print("\n" + "=" * 80)
    print(f"Running: {variant_name}")
    print("=" * 80)

    # Create model
    model = MultiClassPerceptron(
        n_classes=10,
        learning_rate=learning_rate,
        decay_rate=decay_rate
    )

    # Train
    history = model.fit(X_train, y_train, n_epochs=n_epochs, verbose=True)

    # Evaluate on test set
    predictions, n_errors, accuracy, conf_matrix = print_evaluation_results(
        model, X_test, y_test, dataset_name="Test Set"
    )

    results = {
        'predictions': predictions,
        'n_errors': n_errors,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix
    }

    return model, history, results


def main():
    """
    Main function to run all experiments as specified in the assignment.
    """
    print("\n" + "=" * 80)
    print("Multi-Class Perceptron for Digit Classification")
    print("Assignment Question 5")
    print("=" * 80)

    # ===================================================================
    # Step 1: Load Data
    # ===================================================================
    print("\n### STEP 1: Loading Data ###\n")

    try:
        X_train, y_train = load_training_data(data_dir='digitdata')
        X_test, y_test = load_test_data(data_dir='digitdata')
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure the 'digitdata' directory exists with the following files:")
        print("  - trainingimages")
        print("  - traininglabels")
        print("  - testimages")
        print("  - testlabels")
        print("\nAlternatively, specify the correct path to your data directory.")
        return

    # Print data summaries
    print_data_summary(X_train, y_train, "Training Set")
    print_data_summary(X_test, y_test, "Test Set")

    # ===================================================================
    # Step 2: Variant 1 - Fixed Learning Rate (α = 1)
    # ===================================================================
    print("\n### STEP 2: Variant 1 - Fixed Learning Rate ###\n")

    model1, history1, results1 = run_experiment(
        X_train, y_train, X_test, y_test,
        learning_rate=1.0,
        decay_rate=None,  # Fixed learning rate
        n_epochs=4,
        variant_name="Variant 1: Fixed Learning Rate (α = 1)"
    )

    # ===================================================================
    # Step 3: Variant 2 - Decreasing Learning Rate
    # ===================================================================
    print("\n### STEP 3: Variant 2 - Decreasing Learning Rate ###\n")

    model2, history2, results2 = run_experiment(
        X_train, y_train, X_test, y_test,
        learning_rate=1.0,
        decay_rate=0.8,  # Multiply by 0.8 after each epoch
        n_epochs=4,
        variant_name="Variant 2: Decreasing Learning Rate (α starts at 1, decays by 0.8)"
    )

    # ===================================================================
    # Step 4: Compare Results
    # ===================================================================
    print("\n### STEP 4: Comparison of Variants ###\n")

    compare_variants(
        history1, history2,
        name1="Fixed α=1",
        name2="Decreasing α (0.8 decay)"
    )

    # ===================================================================
    # Step 5: Summary Report
    # ===================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - Assignment Requirements")
    print("=" * 80)

    print("\n1. Training Errors per Epoch:")
    print("-" * 80)

    print("\nVariant 1: Fixed Learning Rate (α = 1)")
    for i, (epoch, errors) in enumerate(zip(history1['epoch'], history1['train_errors'])):
        print(f"  Epoch {epoch}: {errors} errors")

    print("\nVariant 2: Decreasing Learning Rate")
    for i, (epoch, errors, lr) in enumerate(zip(history2['epoch'],
                                                  history2['train_errors'],
                                                  history2['learning_rates'])):
        print(f"  Epoch {epoch}: {errors} errors (α = {lr:.4f})")

    print("\n" + "-" * 80)
    print("\n2. Test Set Errors:")
    print("-" * 80)
    print(f"  Variant 1: {results1['n_errors']} errors out of {len(y_test)} "
          f"(Accuracy: {100*results1['accuracy']:.2f}%)")
    print(f"  Variant 2: {results2['n_errors']} errors out of {len(y_test)} "
          f"(Accuracy: {100*results2['accuracy']:.2f}%)")

    print("\n" + "-" * 80)
    print("\n3. Confusion Matrix (shown above for each variant)")
    print("-" * 80)

    # Save confusion matrices to files
    save_confusion_matrix_to_file(
        results1['conf_matrix'],
        'confusion_matrix_variant1.txt'
    )
    save_confusion_matrix_to_file(
        results2['conf_matrix'],
        'confusion_matrix_variant2.txt'
    )

    print("\n" + "=" * 80)
    print("Experiments completed successfully!")
    print("=" * 80)

    # Return results for further analysis if needed
    return {
        'variant1': {'model': model1, 'history': history1, 'results': results1},
        'variant2': {'model': model2, 'history': history2, 'results': results2}
    }


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    np.random.seed(42)

    # Run all experiments
    results = main()

    print("\n\nTo run this script, execute:")
    print("  python run_experiments.py")
    print("\nMake sure the 'digitdata' directory is in the same folder.")
