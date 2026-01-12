"""
Multi-class Perceptron for Handwritten Digit Classification
מבוא לבינה מלאכותית - תרגיל בית 3, שאלה 5
אוניברסיטת חיפה

Algorithm from lecture:
- For each class c, maintain weight vector Wc
- Decision: ĉ = argmax_c(Wc · X)
- Update rule (on error):
    Wc = Wc + αX  (correct class)
    Wc' = Wc' - αX (wrong class)
"""

import numpy as np


def load_images(filepath):
    """
    Load images from file.
    Each image is 28x28 pixels.
    Pixel values: ' ' (space) = 0, '+' or '#' = 1
    """
    images = []
    current_image = []

    with open(filepath, 'r') as f:
        for line in f:
            row = line.rstrip('\n')
            row = row.ljust(28)  # Pad to 28 chars
            # Convert: space=0, anything else=1
            pixel_row = [1 if c != ' ' else 0 for c in row]
            current_image.extend(pixel_row)

            if len(current_image) == 28 * 28:
                images.append(np.array(current_image, dtype=np.float64))
                current_image = []

    return np.array(images)


def load_labels(filepath):
    """Load digit labels (0-9) from file."""
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return np.array(labels)


def predict(weights, x):
    """
    Predict class for a single sample.
    Returns predicted class and all scores.
    """
    scores = np.dot(weights, x)  # Wc · X for each class
    return np.argmax(scores), scores


def evaluate(weights, X, y):
    """
    Evaluate model on dataset.
    Returns number of errors and predictions.
    """
    errors = 0
    predictions = []

    for i in range(len(X)):
        pred, _ = predict(weights, X[i])
        predictions.append(pred)
        if pred != y[i]:
            errors += 1

    return errors, np.array(predictions)


def train_perceptron(X_train, y_train, X_test, y_test, num_epochs=4, decay_lr=False):
    """
    Train multi-class perceptron.

    Update rules (from lecture):
    - If error:
        W_correct = W_correct + α * X
        W_wrong = W_wrong - α * X
    - If tie at beginning (all weights=0): update ALL wrong classes

    Parameters:
    -----------
    decay_lr : bool
        If True: α starts at 1, decreases by 20% after each epoch
        If False: α = 1 (constant)

    Returns:
    --------
    weights, train_errors_per_epoch, test_errors
    """
    n_samples, n_features = X_train.shape
    num_classes = 10

    # Initialize all weights to 0
    weights = np.zeros((num_classes, n_features))

    alpha = 1.0
    train_errors_per_epoch = []

    for epoch in range(num_epochs):
        # Training pass
        for i in range(n_samples):
            x = X_train[i]
            y_true = y_train[i]

            # Compute Wc · X for each class
            scores = np.dot(weights, x)

            # Find winner(s) - classes with maximum score
            max_score = np.max(scores)
            winners = np.where(scores == max_score)[0]

            # Check if prediction is correct
            # Correct only if true class is the UNIQUE winner
            if len(winners) == 1 and winners[0] == y_true:
                continue  # No error, no update

            # Error case: update weights
            # Rule from lecture: W_c = W_c + α*X (correct class up)
            weights[y_true] += alpha * x

            # Rule from lecture: W_c' = W_c' - α*X (wrong classes down)
            # Note: In case of tie, update ALL wrong tied classes
            for wrong_class in winners:
                if wrong_class != y_true:
                    weights[wrong_class] -= alpha * x

        # Evaluate on training set AFTER epoch
        train_errors, _ = evaluate(weights, X_train, y_train)
        train_errors_per_epoch.append(train_errors)

        print(f"Epoch {epoch + 1}: Training errors = {train_errors}, α = {alpha}")

        # Decay learning rate after epoch (if enabled)
        if decay_lr:
            alpha *= 0.8

    # Final evaluation on test set
    test_errors, test_predictions = evaluate(weights, X_test, y_test)

    return weights, train_errors_per_epoch, test_errors, test_predictions


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix.
    Matrix[i][j] = percentage of cases where true class i was classified as j.
    Sum of entire matrix = 1.0 (100%)
    """
    n_samples = len(y_true)
    matrix = np.zeros((num_classes, num_classes))

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    # Convert to percentages (sum = 1)
    matrix = matrix / n_samples

    return matrix


def print_confusion_matrix(matrix):
    """Print confusion matrix in readable format."""
    print("\nConfusion Matrix (מטריצת בלבול):")
    print("Rows = True class (סוג אמיתי), Columns = Predicted class (סוג חזוי)")
    print("Values are percentages, sum = 1.0 (100%)")
    print()

    # Header
    print("      ", end="")
    for j in range(10):
        print(f"   {j}    ", end="")
    print()
    print("      " + "-" * 80)

    # Rows
    for i in range(10):
        print(f"  {i}  |", end="")
        for j in range(10):
            val = matrix[i][j] * 100
            print(f" {val:5.2f}% ", end="")
        print()

    print()
    print(f"Sum of matrix: {np.sum(matrix) * 100:.2f}%")
    print(f"Diagonal (correct): {np.trace(matrix) * 100:.2f}%")


def main():
    print("=" * 70)
    print("שאלה 5: סיווג ספרות עם פרספטרון רב-מחלקתי")
    print("Multi-class Perceptron for Digit Classification")
    print("=" * 70)

    # Load data from same directory as script
    print("\nטוען נתונים / Loading data...")
    X_train = load_images('trainingimages')
    y_train = load_labels('traininglabels')
    X_test = load_images('testimages')
    y_test = load_labels('testlabels')

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features (pixels): {X_train.shape[1]} (28x28)")

    # ============================================================
    # VERSION 1: Constant Learning Rate (α = 1)
    # גרסה 1: קצב למידה קבוע
    # ============================================================
    print("\n" + "=" * 70)
    print("גרסה 1: קצב למידה קבוע (α = 1)")
    print("VERSION 1: Constant Learning Rate (α = 1)")
    print("=" * 70)

    weights1, train_errors1, test_errors1, test_pred1 = train_perceptron(
        X_train, y_train, X_test, y_test,
        num_epochs=4, decay_lr=False
    )

    print(f"\n2. Test errors (שגיאות מבחן): {test_errors1} / {len(y_test)}")
    print(f"   Test accuracy: {(1 - test_errors1/len(y_test))*100:.2f}%")

    print("\n3. Confusion Matrix:")
    confusion1 = compute_confusion_matrix(y_test, test_pred1)
    print_confusion_matrix(confusion1)

    # ============================================================
    # VERSION 2: Decaying Learning Rate
    # גרסה 2: קצב למידה יורד
    # ============================================================
    print("\n" + "=" * 70)
    print("גרסה 2: קצב למידה יורד (α מתחיל ב-1, יורד ב-20% אחרי כל epoch)")
    print("VERSION 2: Decaying Learning Rate (α starts at 1, -20% per epoch)")
    print("=" * 70)

    weights2, train_errors2, test_errors2, test_pred2 = train_perceptron(
        X_train, y_train, X_test, y_test,
        num_epochs=4, decay_lr=True
    )

    print(f"\n2. Test errors (שגיאות מבחן): {test_errors2} / {len(y_test)}")
    print(f"   Test accuracy: {(1 - test_errors2/len(y_test))*100:.2f}%")

    print("\n3. Confusion Matrix:")
    confusion2 = compute_confusion_matrix(y_test, test_pred2)
    print_confusion_matrix(confusion2)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("סיכום / SUMMARY")
    print("=" * 70)

    print("\n1. מספר השגיאות עבור דוגמאות האימון אחרי כל epoch:")
    print("   Training errors after each epoch:")
    print("-" * 50)
    print(f"{'Epoch':<8} {'α קבוע':<20} {'α יורד':<20}")
    print(f"{'':8} {'(Constant α)':<20} {'(Decaying α)':<20}")
    print("-" * 50)
    for i in range(4):
        print(f"{i+1:<8} {train_errors1[i]:<20} {train_errors2[i]:<20}")

    print("\n2. מספר השגיאות עבור דוגמאות המבחן בסוף הלמידה:")
    print("   Test errors at end of training:")
    print("-" * 50)
    print(f"Constant α: {test_errors1} errors ({(1-test_errors1/len(y_test))*100:.2f}% accuracy)")
    print(f"Decaying α: {test_errors2} errors ({(1-test_errors2/len(y_test))*100:.2f}% accuracy)")

    # Save results to file
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("שאלה 5: סיווג ספרות עם פרספטרון רב-מחלקתי\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. מספר השגיאות עבור דוגמאות האימון אחרי כל epoch:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Epoch':<8} {'α קבוע':<20} {'α יורד':<20}\n")
        f.write("-" * 50 + "\n")
        for i in range(4):
            f.write(f"{i+1:<8} {train_errors1[i]:<20} {train_errors2[i]:<20}\n")

        f.write("\n2. מספר השגיאות עבור דוגמאות המבחן בסוף הלמידה:\n")
        f.write(f"Constant α: {test_errors1} errors\n")
        f.write(f"Decaying α: {test_errors2} errors\n")

        f.write("\n3. Confusion Matrix (Constant α):\n")
        f.write("   " + "".join(f"   {j}    " for j in range(10)) + "\n")
        for i in range(10):
            f.write(f"{i} |")
            for j in range(10):
                f.write(f" {confusion1[i][j]*100:5.2f}% ")
            f.write("\n")

        f.write("\n3. Confusion Matrix (Decaying α):\n")
        f.write("   " + "".join(f"   {j}    " for j in range(10)) + "\n")
        for i in range(10):
            f.write(f"{i} |")
            for j in range(10):
                f.write(f" {confusion2[i][j]*100:5.2f}% ")
            f.write("\n")

    print("\nResults saved to results.txt")


if __name__ == "__main__":
    main()