# Logistic Regression with Gradient Descent (From Scratch)

A simple, from-scratch implementation of batch gradient descent for logistic regression, benchmarked against scikit-learn's `LogisticRegression`.

## What it does

- Generates a synthetic, linearly separable 2D binary classification dataset (`make_classification`).
- Fits `sklearn.linear_model.LogisticRegression` (no penalty, `sag` solver) as a baseline.
- Implements logistic regression manually using:
  - A `sigmoid` activation function
  - Batch gradient descent (`lr=0.5`, 5000 iterations) to learn weights and bias
- Computes the decision boundary line for both models.
- Plots both decision boundaries over the dataset for visual comparison.

## Result

The hand-written gradient descent boundary closely matches sklearn's boundary, validating the manual implementation.

## Requirements

```
numpy
matplotlib
scikit-learn
```

## Usage

Run all cells top to bottom in Jupyter. The final plot overlays:
- 🔴 Red line — sklearn's decision boundary
- ⚫ Black line — custom gradient descent decision boundary
