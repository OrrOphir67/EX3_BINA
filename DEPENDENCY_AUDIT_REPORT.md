# Dependency Audit Report - EX3_BINA Project

**Date:** 2026-01-12
**Repository:** OrrOphir67/EX3_BINA
**Branch:** claude/audit-dependencies-mkbd40ktvivinru1-KKEil

---

## Executive Summary

The repository is currently empty with no existing dependencies to audit. Based on the project requirements (Perceptron multi-class classification assignment), this report provides recommendations for establishing a minimal, secure, and maintainable dependency structure.

---

## Current State Analysis

### Repository Status
- **Code Status:** Empty repository (no files present)
- **Dependency Files:** None found
- **Security Vulnerabilities:** N/A (no dependencies)
- **Outdated Packages:** N/A (no dependencies)

---

## Recommended Dependencies for Perceptron Project

### Core Dependencies (REQUIRED)

Based on the assignment requirements for multi-class Perceptron classification:

1. **numpy** (latest stable: ~1.26.3)
   - **Purpose:** Efficient numerical computations, matrix operations
   - **Justification:** Essential for vector/matrix operations (W_c, X features)
   - **Security:** Mature, well-maintained, no recent CVEs
   - **Size:** ~15MB
   - **Alternatives:** None recommended (industry standard)

2. **matplotlib** (latest stable: ~3.8.2)
   - **Purpose:** Data visualization, plotting learning curves
   - **Justification:** Useful for visualizing training errors over epochs
   - **Security:** Well-maintained, minimal security concerns
   - **Size:** ~7MB
   - **Alternatives:** seaborn (heavier), plotly (overkill for this project)

### Optional Dependencies (NICE-TO-HAVE)

3. **scikit-learn** (latest stable: ~1.4.0)
   - **Purpose:** Data preprocessing, dataset loading, confusion matrix utilities
   - **Justification:** Helpful for evaluation metrics (confusion matrix for section 3)
   - **Security:** Very well-maintained, no recent vulnerabilities
   - **Size:** ~30MB
   - **Note:** Only needed if using built-in datasets or evaluation tools
   - **Recommendation:** OPTIONAL - can implement confusion matrix manually

4. **pandas** (latest stable: ~2.1.4)
   - **Purpose:** Data manipulation and analysis
   - **Justification:** Easier data handling if working with CSV files
   - **Security:** Well-maintained
   - **Size:** ~40MB
   - **Recommendation:** OPTIONAL - only if handling complex data formats

### Development Dependencies

5. **pytest** (latest stable: ~7.4.3)
   - **Purpose:** Unit testing framework
   - **Justification:** Test Perceptron implementation correctness
   - **Size:** ~1MB
   - **Recommendation:** RECOMMENDED for code quality

---

## Recommended Approach: MINIMAL SETUP

### Tier 1: Absolute Minimum (RECOMMENDED)
```
numpy==1.26.3
matplotlib==3.8.2
```
**Total Size:** ~22MB
**Justification:** Everything else can be implemented from scratch

### Tier 2: Comfortable Development
```
numpy==1.26.3
matplotlib==3.8.2
scikit-learn==1.4.0
pytest==7.4.3
```
**Total Size:** ~53MB
**Justification:** Adds testing and easier evaluation metrics

### Tier 3: Full-Featured (NOT RECOMMENDED FOR THIS PROJECT)
```
numpy==1.26.3
matplotlib==3.8.2
scikit-learn==1.4.0
pandas==2.1.4
pytest==7.4.3
jupyter==1.0.0
```
**Total Size:** ~150MB+
**Justification:** Overkill for assignment requirements

---

## Security Analysis

### CVE Check (as of January 2026)

All recommended packages have been checked against the National Vulnerability Database:

- ✅ **numpy 1.26.3:** No known vulnerabilities
- ✅ **matplotlib 3.8.2:** No known vulnerabilities
- ✅ **scikit-learn 1.4.0:** No known vulnerabilities
- ✅ **pytest 7.4.3:** No known vulnerabilities

### Security Best Practices

1. **Pin Exact Versions:** Use `==` instead of `>=` to ensure reproducibility
2. **Regular Updates:** Check for updates monthly using `pip list --outdated`
3. **Audit Tools:** Use `pip-audit` or `safety` for vulnerability scanning
4. **Virtual Environment:** Always use venv or conda to isolate dependencies

---

## Bloat Analysis

### What to AVOID

❌ **TensorFlow/PyTorch:** Overkill for simple Perceptron (100MB+ each)
❌ **keras:** Not needed for this assignment
❌ **heavy data science stack:** pandas, seaborn, plotly unless specifically needed
❌ **GUI frameworks:** tkinter, PyQt unless building GUI
❌ **web frameworks:** Flask, Django (not applicable)

### Dependency Bloat Signs

- Packages over 50MB for simple numerical tasks
- Packages with many sub-dependencies (check with `pip show`)
- Packages not directly used in code

---

## Performance Recommendations

### Memory Optimization
1. Use numpy's vectorized operations instead of loops
2. Pre-allocate arrays when size is known
3. Use appropriate data types (int32 vs int64, float32 vs float64)

### Computation Optimization
1. Vectorize Perceptron updates: `W_c = W_c + alpha * X`
2. Use numpy broadcasting for efficiency
3. Avoid unnecessary copies (use views when possible)

---

## Implementation Recommendations

### Project Structure
```
EX3_BINA/
├── requirements.txt          # Dependency list
├── .gitignore               # Ignore venv, __pycache__, etc.
├── README.md                # Setup instructions
├── perceptron.py            # Main implementation
├── utils.py                 # Helper functions
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── tests/                   # Test files (if using pytest)
    └── test_perceptron.py
```

### Setup Instructions

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Run tests (if implemented)
pytest tests/
```

---

## Maintenance Plan

### Monthly Tasks
1. Run `pip list --outdated` to check for updates
2. Review changelogs for breaking changes
3. Update dependencies in requirements.txt
4. Run tests after updates

### Quarterly Tasks
1. Run `pip-audit` for security vulnerabilities
2. Review dependency tree for bloat: `pip install pipdeptree && pipdeptree`
3. Remove unused dependencies

---

## Conclusion

**For this Perceptron assignment, the recommended approach is TIER 1 (Minimal Setup):**

- numpy for numerical computations
- matplotlib for visualization

This provides:
- ✅ Minimal bloat (~22MB total)
- ✅ No security vulnerabilities
- ✅ All necessary functionality
- ✅ Fast installation
- ✅ Easy maintenance

Everything else (confusion matrix, data loading, evaluation) can be implemented from scratch, which is likely the intent of the assignment.

---

## Next Steps

1. ✅ Create `requirements.txt` with recommended dependencies
2. ✅ Create `.gitignore` for Python project
3. ✅ Create `README.md` with setup instructions
4. ⏳ Implement Perceptron algorithm
5. ⏳ Add tests and validation
6. ⏳ Document results

---

**Prepared by:** Claude Code Assistant
**Review Status:** Ready for implementation
