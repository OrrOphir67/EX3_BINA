# EX3_BINA - Multi-Class Perceptron Classification

This project implements a multi-class Perceptron classifier for digit recognition using the MNIST-like dataset.

## Project Overview

Implementation of the Perceptron algorithm for multi-class classification (digits 0-9) with the following features:

- Training on 5000 examples (500 per class)
- Testing on approximately 1000 examples (100 per class)
- Two training variations: fixed learning rate (α=1) and decreasing learning rate
- Evaluation over 4 epochs
- Comprehensive error analysis and confusion matrix generation

## Requirements

### System Requirements
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Dependencies

See `requirements.txt` for the complete list. Core dependencies:
- numpy (1.26.3) - Numerical computations
- matplotlib (3.8.2) - Visualization

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd EX3_BINA
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import numpy; import matplotlib; print('Dependencies installed successfully!')"
```

## Project Structure

```
EX3_BINA/
├── README.md                     # This file
├── DEPENDENCY_AUDIT_REPORT.md    # Comprehensive dependency analysis
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
├── perceptron.py                 # (To be implemented) Core Perceptron class
├── train.py                      # (To be implemented) Training script
├── evaluate.py                   # (To be implemented) Evaluation script
└── utils.py                      # (To be implemented) Helper functions
```

## Assignment Requirements

### Task 5: Multi-Class Perceptron with Percentron

Implement Perceptron algorithm for 10-class classification (digits 0-9) with weighted voting.

**Key Components:**
1. **Training Data:** 5000 examples (500 per class)
2. **Test Data:** ~1000 examples (~100 per class)
3. **Algorithm:** Multi-class Perceptron with per-class weight vectors (W_c)
4. **Learning Rule:**
   - Predicted class: ĉ = argmax_c(W_c · X)
   - Update for correct class c: W_c ← W_c + αX
   - Update for all other classes c': W_c' ← W_c' - αX
5. **Epochs:** 4 training epochs for both variants

### Two Implementation Variants

#### Variant 1: Fixed Learning Rate
- Learning rate: α = 1 (constant)

#### Variant 2: Decreasing Learning Rate
- Initial: α = 1
- After each epoch: α = α × 0.8 (20% reduction)
- Epochs 1-4: α = 1.0, 0.8, 0.64, 0.512

### Required Outputs

1. **Training Errors per Epoch**
   - Report number of misclassifications after each epoch
   - For both variants

2. **Test Set Errors**
   - Final error count on test set after training
   - For both variants

3. **Confusion Matrix**
   - 10×10 matrix showing percentage of misclassifications
   - Rows: True labels (i)
   - Columns: Predicted labels (j)
   - Diagonal: Correct classifications
   - Matrix sum: 1.0 (100%)

## Usage

### Training
```bash
# Train with fixed learning rate
python train.py --variant fixed --epochs 4 --alpha 1.0

# Train with decreasing learning rate
python train.py --variant decreasing --epochs 4 --alpha 1.0 --decay 0.8
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py --model-path models/perceptron.pkl

# Generate confusion matrix
python evaluate.py --model-path models/perceptron.pkl --confusion-matrix
```

## Implementation Notes

### Algorithm Pseudocode
```
For each epoch:
    For each training example (X, true_label):
        # Predict
        predicted_label = argmax_c(W_c · X)

        # Update if wrong
        if predicted_label != true_label:
            W_true_label += α * X
            W_predicted_label -= α * X

    # For decreasing variant only
    α = α * 0.8
```

### Key Implementation Details

1. **Feature Vector (X):**
   - Include bias term (add 1 to feature vector)
   - Normalize pixel values (0-1 range)

2. **Weight Initialization:**
   - Initialize all W_c to zeros
   - Shape: (10 classes, num_features)

3. **Training Loop:**
   - Shuffle data between epochs
   - Track errors for each epoch

4. **Evaluation:**
   - Use argmax for final predictions
   - Calculate confusion matrix percentages

## Dependency Management

### Security and Updates

This project follows minimal dependency principles. For detailed analysis, see `DEPENDENCY_AUDIT_REPORT.md`.

**Check for updates:**
```bash
pip list --outdated
```

**Security audit:**
```bash
pip install pip-audit
pip-audit
```

### Adding Dependencies

Before adding new dependencies:
1. Check if functionality can be implemented without it
2. Review security status and maintenance
3. Consider package size and sub-dependencies
4. Update `requirements.txt` with pinned version

## Development

### Running Tests
```bash
# Install testing dependencies
pip install pytest

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Code Quality
```bash
# Format code
pip install black
black .

# Lint code
pip install flake8
flake8 .
```

## Data Format

Expected data format for training and testing:

**File format:** Text file with space-separated values
- First column: Label (0-9)
- Remaining columns: Pixel values (features)
- Example: `readme.txt` (provided in assignment)

## Results

Results will be documented here after implementation:

### Variant 1: Fixed Learning Rate (α=1)
- Epoch 1 errors: TBD
- Epoch 2 errors: TBD
- Epoch 3 errors: TBD
- Epoch 4 errors: TBD
- Test set errors: TBD

### Variant 2: Decreasing Learning Rate
- Epoch 1 errors (α=1.0): TBD
- Epoch 2 errors (α=0.8): TBD
- Epoch 3 errors (α=0.64): TBD
- Epoch 4 errors (α=0.512): TBD
- Test set errors: TBD

### Confusion Matrix
TBD - 10×10 matrix with classification percentages

## Troubleshooting

### Common Issues

**Import Error: No module named 'numpy'**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**Memory Error**
```bash
# Use float32 instead of float64 to reduce memory usage
# Modify code to use: dtype=np.float32
```

**Slow Training**
```bash
# Ensure using vectorized numpy operations
# Avoid Python loops over data
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -m "Description"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

## License

Academic project for educational purposes.

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated:** 2026-01-12
**Status:** Dependencies configured, implementation pending
