"""
Multi-class Perceptron for Handwritten Digit Classification
"""

import numpy as np

def load_data(images_file, labels_file):
    """
    טוען את הנתונים מהקבצים
    פורמט: 28 שורות לכל תמונה, רווח=לבן, +/#=שחור
    """
    # קריאת תוויות
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    # קריאת תמונות
    with open(images_file, 'r') as f:
        lines = f.readlines()

    n_samples = len(labels)
    rows_per_image = 28
    cols_per_image = 28
    n_features = rows_per_image * cols_per_image

    images = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        start_line = i * rows_per_image
        pixels = []
        for row in range(rows_per_image):
            line = lines[start_line + row]
            # וידוא שיש מספיק תווים
            line = line.ljust(cols_per_image)
            for col in range(cols_per_image):
                char = line[col] if col < len(line) else ' '
                # + או # = שחור/אפור (1), רווח = לבן (0)
                if char in ['+', '#']:
                    pixels.append(1)
                else:
                    pixels.append(0)
        images[i] = pixels

    return images, np.array(labels)


class MultiClassPerceptron:
    def __init__(self, n_features, n_classes=10):
        """
        אתחול הפרספטרון
        """
        self.n_classes = n_classes
        self.n_features = n_features
        # אתחול משקולות לאפס
        self.W = np.zeros((n_classes, n_features))

    def predict(self, X):
        """
        חיזוי עבור דוגמה אחת או יותר
        """
        scores = np.dot(self.W, X.T)
        return np.argmax(scores, axis=0)

    def train_epoch(self, X_train, y_train, alpha):
        """
        אימון epoch אחד
        מחזיר את מספר השגיאות
        """
        n_errors = 0

        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]

            # חישוב הסכומים הממושקלים
            scores = np.dot(self.W, x)

            # מציאת כל המחלקות עם הסכום המקסימלי (לטיפול בתיקו)
            max_score = np.max(scores)
            predicted_classes = np.where(scores == max_score)[0]

            # בדיקה אם יש שגיאה (כולל תיקו)
            if y_true not in predicted_classes or len(predicted_classes) > 1:
                n_errors += 1

                # עדכון משקולות המחלקה הנכונה כלפי מעלה
                self.W[y_true] += alpha * x

                # עדכון משקולות כל המחלקות השגויות כלפי מטה
                for c_wrong in predicted_classes:
                    if c_wrong != y_true:
                        self.W[c_wrong] -= alpha * x

        return n_errors

    def evaluate(self, X_test, y_test):
        """
        הערכת המודל על סט בדיקה
        """
        predictions = self.predict(X_test)
        n_errors = np.sum(predictions != y_test)

        # מטריצת בלבול
        confusion = np.zeros((self.n_classes, self.n_classes))
        for true_label, pred_label in zip(y_test, predictions):
            confusion[true_label, pred_label] += 1

        # המרה לאחוזים (נרמול לפי סה"כ הדוגמאות)
        confusion_percent = confusion / len(y_test)

        return n_errors, confusion_percent, confusion


def run_experiment(X_train, y_train, X_test, y_test, decay_alpha=False):
    """
    הרצת הניסוי עם או בלי דעיכת קצב למידה
    """
    n_features = X_train.shape[1]
    perceptron = MultiClassPerceptron(n_features)

    alpha = 1.0
    n_epochs = 4

    print(f"\n{'='*60}")
    if decay_alpha:
        print("קצב למידה יורד (מתחיל ב-1, יורד 20% אחרי כל epoch)")
    else:
        print("קצב למידה קבוע (alpha = 1)")
    print(f"{'='*60}")

    train_errors_per_epoch = []

    for epoch in range(n_epochs):
        # אימון epoch
        n_train_errors = perceptron.train_epoch(X_train, y_train, alpha)
        train_errors_per_epoch.append(n_train_errors)

        print(f"Epoch {epoch + 1}: שגיאות אימון = {n_train_errors} (מתוך {len(X_train)}), alpha = {alpha:.4f}")

        # עדכון קצב למידה אם צריך
        if decay_alpha:
            alpha *= 0.8

    # הערכה על סט המבחן
    n_test_errors, confusion_percent, confusion_counts = perceptron.evaluate(X_test, y_test)

    print(f"\nשגיאות על סט המבחן: {n_test_errors} (מתוך {len(y_test)})")
    print(f"דיוק על סט המבחן: {100 * (1 - n_test_errors / len(y_test)):.2f}%")

    return train_errors_per_epoch, n_test_errors, confusion_percent, confusion_counts


def print_confusion_matrix(confusion_matrix, title=""):
    """
    הדפסת מטריצת הבלבול באופן נאה
    """
    print(f"\n{title}")
    print("שורות = מחלקה אמיתית (i), עמודות = מחלקה חזויה (j)")
    print("האלכסון = סיווגים נכונים")
    print()

    # כותרות עמודות
    print("      ", end="")
    for j in range(10):
        print(f"    {j}   ", end="")
    print()
    print("      " + "-" * 80)

    for i in range(10):
        print(f"  {i}  |", end="")
        for j in range(10):
            val = confusion_matrix[i, j] * 100
            if i == j:
                print(f" [{val:5.2f}%]", end="")
            else:
                print(f"  {val:5.2f}% ", end="")
        print()

    print(f"\nסכום כל האחוזים במטריצה: {confusion_matrix.sum()*100:.2f}%")


def main():
    """
    הרצה ראשית
    """
    data_dir = "digitdata"

    print("טוען נתונים...")
    X_train, y_train = load_data(
        f"{data_dir}/trainingimages",
        f"{data_dir}/traininglabels"
    )
    X_test, y_test = load_data(
        f"{data_dir}/testimages",
        f"{data_dir}/testlabels"
    )

    print(f"נטענו {len(X_train)} דוגמאות אימון ו-{len(X_test)} דוגמאות בדיקה")
    print(f"מספר תכונות (פיקסלים): {X_train.shape[1]}")

    # ספירת דוגמאות לכל מחלקה
    print("\nהתפלגות דוגמאות אימון:")
    for digit in range(10):
        count = np.sum(y_train == digit)
        print(f"  ספרה {digit}: {count} דוגמאות")

    # ניסוי 1: קצב למידה קבוע
    train_errors_const, test_errors_const, confusion_const, counts_const = run_experiment(
        X_train, y_train, X_test, y_test, decay_alpha=False
    )

    # ניסוי 2: קצב למידה יורד
    train_errors_decay, test_errors_decay, confusion_decay, counts_decay = run_experiment(
        X_train, y_train, X_test, y_test, decay_alpha=True
    )

    # סיכום
    print("\n" + "="*70)
    print("סיכום התוצאות")
    print("="*70)

    print("\n1. שגיאות אימון אחרי כל epoch:")
    print(f"{'Epoch':<10} {'קצב קבוע':<20} {'קצב יורד':<20}")
    print("-" * 50)
    for i in range(4):
        print(f"{i+1:<10} {train_errors_const[i]:<20} {train_errors_decay[i]:<20}")

    print(f"\n2. שגיאות על סט המבחן בסוף הלמידה:")
    print(f"   קצב קבוע: {test_errors_const} שגיאות ({100*test_errors_const/len(y_test):.2f}%)")
    print(f"   קצב יורד: {test_errors_decay} שגיאות ({100*test_errors_decay/len(y_test):.2f}%)")

    print("\n" + "="*70)
    print("3. מטריצת בלבול - קצב למידה קבוע")
    print("="*70)
    print_confusion_matrix(confusion_const, "אחוז המקרים שבהם דוגמא מסוג i סווגה כספרה מסוג j:")

    print("\n" + "="*70)
    print("מטריצת בלבול - קצב למידה יורד")
    print("="*70)
    print_confusion_matrix(confusion_decay, "אחוז המקרים שבהם דוגמא מסוג i סווגה כספרה מסוג j:")


if __name__ == "__main__":
    main()
