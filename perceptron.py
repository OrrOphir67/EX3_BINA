"""
Multi-Class Perceptron Implementation for Digit Classification
"""

import numpy as np


class MultiClassPerceptron:
    """
    Multi-class Perceptron classifier for digit recognition (0-9).

    Implements the weighted voting perceptron algorithm where each class
    has its own weight vector W_c.
    """

    def __init__(self, n_classes=10, learning_rate=1.0, decay_rate=None):
        """
        Initialize the Multi-Class Perceptron.

        Args:
            n_classes (int): Number of classes (default: 10 for digits 0-9)
            learning_rate (float): Initial learning rate α (default: 1.0)
            decay_rate (float): Learning rate decay factor after each epoch.
                               If None, learning rate stays fixed.
                               Example: 0.8 means α *= 0.8 after each epoch
        """
        self.n_classes = n_classes
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.weights = None  # Will be initialized in fit()
        self.n_features = None

    def _initialize_weights(self, n_features):
        """
        Initialize weight vectors to zeros.

        Args:
            n_features (int): Number of features in input
        """
        self.n_features = n_features
        # Create weight matrix: n_classes x n_features
        # Each row is W_c for class c
        self.weights = np.zeros((self.n_classes, n_features))

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
                           or (n_features,) for single sample

        Returns:
            np.ndarray: Predicted class labels
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Compute W_c · X for all classes: (n_samples, n_classes)
        scores = np.dot(X, self.weights.T)

        # Find class with maximum score: argmax_c(W_c · X)
        predictions = np.argmax(scores, axis=1)

        return predictions

    def fit(self, X, y, n_epochs=4, verbose=True):
        """
        Train the perceptron for multiple epochs.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
            n_epochs (int): Number of training epochs (default: 4)
            verbose (bool): If True, print training progress

        Returns:
            dict: Training history with errors per epoch
        """
        n_samples, n_features = X.shape

        # Initialize weights if first time
        if self.weights is None:
            self._initialize_weights(n_features)

        # Reset learning rate to initial value
        self.learning_rate = self.initial_learning_rate

        # Track errors per epoch
        history = {
            'epoch': [],
            'train_errors': [],
            'learning_rates': []
        }

        if verbose:
            print(f"\nTraining Multi-Class Perceptron")
            print(f"{'='*60}")
            print(f"Initial learning rate: {self.learning_rate}")
            if self.decay_rate:
                print(f"Decay rate: {self.decay_rate} (multiply after each epoch)")
            else:
                print(f"Learning rate: Fixed")
            print(f"Number of epochs: {n_epochs}")
            print(f"Training samples: {n_samples}")
            print(f"{'='*60}\n")

        for epoch in range(n_epochs):
            errors = 0

            # Train on each example
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # Predict: ĉ = argmax_c(W_c · X)
                scores = np.dot(self.weights, x_i)
                predicted = np.argmax(scores)

                # Update if prediction is wrong
                if predicted != y_i:
                    errors += 1

                    # Check for tie (all scores equal) - update all wrong classes
                    max_score = scores[predicted]
                    tied_classes = np.where(scores == max_score)[0]

                    if len(tied_classes) > 1:
                        # Tie case: update all classes that are not the correct one
                        for c in range(self.n_classes):
                            if c == y_i:
                                # Update correct class: W_c ← W_c + αX
                                self.weights[y_i] += self.learning_rate * x_i
                            else:
                                # Update all wrong classes: W_c' ← W_c' - αX
                                self.weights[c] -= self.learning_rate * x_i
                    else:
                        # Normal case: update correct and predicted classes only
                        # Update correct class: W_c ← W_c + αX
                        self.weights[y_i] += self.learning_rate * x_i

                        # Update predicted (wrong) class: W_c' ← W_c' - αX
                        self.weights[predicted] -= self.learning_rate * x_i

            # Record history
            history['epoch'].append(epoch + 1)
            history['train_errors'].append(errors)
            history['learning_rates'].append(self.learning_rate)

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                      f"Errors = {errors}/{n_samples} "
                      f"({100*errors/n_samples:.2f}%), "
                      f"α = {self.learning_rate:.4f}")

            # Decay learning rate after epoch (if decay_rate is set)
            if self.decay_rate is not None:
                self.learning_rate *= self.decay_rate

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"{'='*60}\n")

        return history

    def evaluate(self, X, y):
        """
        Evaluate the perceptron on test data.

        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): Test labels of shape (n_samples,)

        Returns:
            tuple: (predictions, n_errors, accuracy)
        """
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        accuracy = 1.0 - (errors / len(y))

        return predictions, errors, accuracy

    def confusion_matrix(self, X, y, normalize=True):
        """
        Compute confusion matrix.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True labels
            normalize (bool): If True, return percentages (sum = 1.0)

        Returns:
            np.ndarray: Confusion matrix of shape (n_classes, n_classes)
                       Entry (i, j) = count/percentage of true class i predicted as j
        """
        predictions = self.predict(X)

        # Initialize confusion matrix
        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        # Fill confusion matrix
        for true_label, pred_label in zip(y, predictions):
            conf_matrix[true_label, pred_label] += 1

        # Normalize if requested
        if normalize:
            total = conf_matrix.sum()
            if total > 0:
                conf_matrix = conf_matrix / total

        return conf_matrix
