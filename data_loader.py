"""
Data loading utilities for digit image dataset.
"""

import numpy as np
import os


# Image dimensions (standard for digit recognition datasets)
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


def load_images(filepath):
    """
    Load digit images from file.

    The file format contains ASCII art representations of digits where:
    - ' ' (space) = white/background
    - '+' or '#' = gray/black (foreground pixels)

    Each image is IMAGE_HEIGHT lines tall.

    Args:
        filepath (str): Path to the images file

    Returns:
        np.ndarray: Array of shape (n_samples, n_features) where
                   n_features = IMAGE_HEIGHT * IMAGE_WIDTH
                   Pixel values are binary: 1 for foreground, 0 for background
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    images = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Process images in chunks of IMAGE_HEIGHT lines
    n_images = len(lines) // IMAGE_HEIGHT

    for i in range(n_images):
        # Extract IMAGE_HEIGHT lines for this image
        image_lines = lines[i * IMAGE_HEIGHT:(i + 1) * IMAGE_HEIGHT]

        # Convert to binary features
        pixels = []
        for line in image_lines:
            # Ensure line has correct width (pad or truncate)
            line = line.rstrip('\n')
            if len(line) < IMAGE_WIDTH:
                line = line + ' ' * (IMAGE_WIDTH - len(line))
            else:
                line = line[:IMAGE_WIDTH]

            # Convert characters to binary: foreground (+ or #) = 1, background ( ) = 0
            row = [1 if char in ['+', '#'] else 0 for char in line]
            pixels.extend(row)

        images.append(pixels)

    # Convert to numpy array and add bias term
    X = np.array(images, dtype=np.float32)

    # Add bias feature (constant 1) to each sample
    bias = np.ones((X.shape[0], 1), dtype=np.float32)
    X = np.hstack([X, bias])

    return X


def load_labels(filepath):
    """
    Load digit labels from file.

    Each line contains a single digit label (0-9).

    Args:
        filepath (str): Path to the labels file

    Returns:
        np.ndarray: Array of shape (n_samples,) containing integer labels
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")

    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                labels.append(int(line))

    return np.array(labels, dtype=np.int32)


def load_data(images_path, labels_path):
    """
    Load both images and labels.

    Args:
        images_path (str): Path to images file
        labels_path (str): Path to labels file

    Returns:
        tuple: (X, y) where
               X is array of shape (n_samples, n_features)
               y is array of shape (n_samples,)
    """
    print(f"Loading images from: {images_path}")
    X = load_images(images_path)
    print(f"  Loaded {X.shape[0]} images with {X.shape[1]} features each")

    print(f"Loading labels from: {labels_path}")
    y = load_labels(labels_path)
    print(f"  Loaded {y.shape[0]} labels")

    # Verify same number of images and labels
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: {X.shape[0]} images but {y.shape[0]} labels")

    return X, y


def load_training_data(data_dir='digitdata'):
    """
    Load training data from default directory structure.

    Args:
        data_dir (str): Directory containing data files

    Returns:
        tuple: (X_train, y_train)
    """
    images_path = os.path.join(data_dir, 'trainingimages')
    labels_path = os.path.join(data_dir, 'traininglabels')
    return load_data(images_path, labels_path)


def load_test_data(data_dir='digitdata'):
    """
    Load test data from default directory structure.

    Args:
        data_dir (str): Directory containing data files

    Returns:
        tuple: (X_test, y_test)
    """
    images_path = os.path.join(data_dir, 'testimages')
    labels_path = os.path.join(data_dir, 'testlabels')
    return load_data(images_path, labels_path)


def load_validation_data(data_dir='digitdata'):
    """
    Load validation data from default directory structure.

    Args:
        data_dir (str): Directory containing data files

    Returns:
        tuple: (X_val, y_val)
    """
    images_path = os.path.join(data_dir, 'validationimages')
    labels_path = os.path.join(data_dir, 'validationlabels')
    return load_data(images_path, labels_path)


def get_class_distribution(y, n_classes=10):
    """
    Get the distribution of examples per class.

    Args:
        y (np.ndarray): Array of labels
        n_classes (int): Number of classes

    Returns:
        dict: Dictionary mapping class label to count
    """
    distribution = {}
    for c in range(n_classes):
        count = np.sum(y == c)
        distribution[c] = count

    return distribution


def print_data_summary(X, y, dataset_name="Dataset"):
    """
    Print summary statistics about a dataset.

    Args:
        X (np.ndarray): Feature array
        y (np.ndarray): Label array
        dataset_name (str): Name of the dataset for display
    """
    print(f"\n{dataset_name} Summary:")
    print(f"{'='*60}")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH} + 1 bias")
    print(f"\n  Class distribution:")

    dist = get_class_distribution(y)
    for label in sorted(dist.keys()):
        count = dist[label]
        percentage = 100 * count / len(y)
        print(f"    Digit {label}: {count} samples ({percentage:.1f}%)")

    print(f"{'='*60}\n")
