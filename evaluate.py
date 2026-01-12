"""
Evaluation utilities for Multi-Class Perceptron.
"""

import numpy as np


def print_confusion_matrix(conf_matrix, class_names=None):
    """
    Print confusion matrix in a formatted way.

    Args:
        conf_matrix (np.ndarray): Confusion matrix of shape (n_classes, n_classes)
        class_names (list): Optional list of class names (default: 0-9 for digits)
    """
    n_classes = conf_matrix.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Print header
    print("\nConfusion Matrix:")
    print("=" * 80)
    print("Rows = True label, Columns = Predicted label")
    print("Matrix shows PERCENTAGES (sum = 1.0 = 100%)")
    print("=" * 80)

    # Column header
    print("\n      ", end="")
    for j in range(n_classes):
        print(f"   {class_names[j]}  ", end="")
    print()

    print("      " + "-" * (7 * n_classes))

    # Print rows
    for i in range(n_classes):
        print(f"  {class_names[i]} |", end="")
        for j in range(n_classes):
            value = conf_matrix[i, j]
            # Highlight diagonal (correct predictions)
            if i == j:
                print(f" {value:5.3f}", end=" ")
            else:
                print(f" {value:5.3f}", end=" ")
        print()

    # Print sum (should be 1.0)
    total = conf_matrix.sum()
    print("\n" + "=" * 80)
    print(f"Matrix sum: {total:.6f} (should be 1.0)")
    print("=" * 80)


def print_confusion_matrix_detailed(conf_matrix, class_names=None):
    """
    Print detailed confusion matrix analysis with error rates per class.

    Args:
        conf_matrix (np.ndarray): Normalized confusion matrix (percentages)
        class_names (list): Optional list of class names
    """
    n_classes = conf_matrix.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    print("\n\nDetailed Error Analysis:")
    print("=" * 80)

    # Calculate accuracy (diagonal sum)
    accuracy = np.trace(conf_matrix)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print(f"Overall Error Rate: {1-accuracy:.4f} ({100*(1-accuracy):.2f}%)")

    print("\n" + "-" * 80)
    print("Per-Class Statistics:")
    print("-" * 80)

    for i in range(n_classes):
        correct_rate = conf_matrix[i, i]
        error_rate = 1.0 - correct_rate / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0

        print(f"\nDigit {class_names[i]}:")
        print(f"  Correct: {100*correct_rate:.2f}%")

        # Find most common misclassifications
        errors = [(j, conf_matrix[i, j]) for j in range(n_classes) if i != j and conf_matrix[i, j] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)

        if errors:
            print(f"  Most common errors:")
            for j, rate in errors[:3]:  # Top 3 errors
                if rate > 0.001:  # Only show if > 0.1%
                    print(f"    Misclassified as {class_names[j]}: {100*rate:.2f}%")

    print("\n" + "=" * 80)


def print_evaluation_results(model, X_test, y_test, dataset_name="Test Set"):
    """
    Print comprehensive evaluation results.

    Args:
        model: Trained perceptron model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        dataset_name (str): Name of the dataset for display
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*80}")

    # Get predictions and errors
    predictions, n_errors, accuracy = model.evaluate(X_test, y_test)

    print(f"\nResults:")
    print(f"  Total samples: {len(y_test)}")
    print(f"  Errors: {n_errors}")
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print(f"  Error rate: {1-accuracy:.4f} ({100*(1-accuracy):.2f}%)")

    # Compute and print confusion matrix
    conf_matrix = model.confusion_matrix(X_test, y_test, normalize=True)

    print_confusion_matrix(conf_matrix)
    print_confusion_matrix_detailed(conf_matrix)

    return predictions, n_errors, accuracy, conf_matrix


def compare_variants(history1, history2, name1="Variant 1", name2="Variant 2"):
    """
    Compare training history of two variants.

    Args:
        history1 (dict): Training history of first variant
        history2 (dict): Training history of second variant
        name1 (str): Name of first variant
        name2 (str): Name of second variant
    """
    print("\n" + "=" * 80)
    print("Comparison of Training Variants")
    print("=" * 80)

    print(f"\n{'Epoch':<8} {name1:<25} {name2:<25}")
    print("-" * 80)

    for i in range(len(history1['epoch'])):
        epoch = history1['epoch'][i]
        errors1 = history1['train_errors'][i]
        errors2 = history2['train_errors'][i]
        lr1 = history1['learning_rates'][i]
        lr2 = history2['learning_rates'][i]

        print(f"{epoch:<8} Errors: {errors1:<6} α: {lr1:<7.4f}   "
              f"Errors: {errors2:<6} α: {lr2:<7.4f}")

    print("=" * 80)


def save_confusion_matrix_to_file(conf_matrix, filepath, class_names=None):
    """
    Save confusion matrix to a text file.

    Args:
        conf_matrix (np.ndarray): Confusion matrix
        filepath (str): Output file path
        class_names (list): Optional class names
    """
    n_classes = conf_matrix.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    with open(filepath, 'w') as f:
        # Write header
        f.write("Confusion Matrix (Percentages)\n")
        f.write("=" * 80 + "\n")
        f.write("Rows = True label, Columns = Predicted label\n")
        f.write("=" * 80 + "\n\n")

        # Write column headers
        f.write("      ")
        for j in range(n_classes):
            f.write(f"   {class_names[j]}  ")
        f.write("\n")

        f.write("      " + "-" * (7 * n_classes) + "\n")

        # Write data rows
        for i in range(n_classes):
            f.write(f"  {class_names[i]} |")
            for j in range(n_classes):
                value = conf_matrix[i, j]
                f.write(f" {value:5.3f} ")
            f.write("\n")

        # Write sum
        total = conf_matrix.sum()
        f.write(f"\n\nMatrix sum: {total:.6f}\n")

    print(f"Confusion matrix saved to: {filepath}")
