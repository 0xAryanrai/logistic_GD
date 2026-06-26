# Logistic Regression from Scratch with Gradient Descent

A from-scratch implementation of **binary logistic regression** trained with **batch gradient descent**, benchmarked against scikit-learn's `LogisticRegression`. The project visually compares the decision boundary learned manually against the one learned by sklearn on the same synthetic dataset.

## Overview

This notebook builds logistic regression from first principles — no `fit()` calls, just the math:

1. Generate a linearly separable 2D binary classification dataset
2. Fit scikit-learn's `LogisticRegression` as a reference/baseline model
3. Implement the **sigmoid function** and a **gradient descent training loop** manually using NumPy
4. Derive the decision boundary line from the learned weights for both models
5. Plot both decision boundaries on the same scatter plot to compare them visually

## Why this project

Most ML workflows treat logistic regression as a black box: import, fit, predict. This notebook instead opens that black box, showing exactly how the weights are updated each iteration and how closely a simple manual gradient descent implementation can match a production-grade library implementation.

## Dataset

A synthetic dataset generated with `sklearn.datasets.make_classification`:
- 100 samples, 2 informative features (easy to visualize in 2D)
- 2 classes, 1 cluster per class
- High class separation (`class_sep=20`) for a clean, clearly separable visualization

## Approach

**Baseline model**
- `LogisticRegression(penalty=None, solver='sag')` from scikit-learn, fit on the raw data
- Coefficients and intercept extracted to compute its decision boundary line

**From-scratch model**
- Custom `sigmoid(z)` function implementing $\sigma(z) = \frac{1}{1+e^{-z}}$
- Custom `gd(X, y)` function that:
  - Adds a bias column to the feature matrix
  - Initializes all weights to 1
  - Runs **5,000 iterations** of batch gradient descent with a learning rate of **0.5**
  - Updates weights each iteration using the gradient of the log-loss: `weights += lr * ((y - y_hat) @ X) / n_samples`

**Comparison**
- Both models' weights are converted into a slope-intercept line (`y = mx + b`) for the 2D decision boundary
- Both lines are plotted over the same scatter of data points to visually compare how closely the manual implementation converges to sklearn's result

## Results

The manually implemented gradient descent model converges to a decision boundary nearly identical to scikit-learn's solver, visually confirming that the from-scratch implementation correctly learns the same separating hyperplane.

## Tech Stack

- **Python 3**
- **NumPy** — vectorized math and the gradient descent loop
- **scikit-learn** — synthetic data generation and baseline `LogisticRegression`
- **Matplotlib** — data and decision boundary visualization

## Getting Started

### Prerequisites
```bash
pip install numpy scikit-learn matplotlib jupyter
```

### Run
```bash
jupyter notebook Gradient_descent_logistic.ipynb
```
Run all cells in order — the dataset is regenerated with a fixed `random_state`, so results are reproducible.

## Project Structure
```
.
├── Gradient_descent_logistic.ipynb   # Main notebook
└── README.md
```

## Key Takeaways

- Logistic regression's decision boundary is fundamentally linear, derived directly from the learned weight vector and bias
- Batch gradient descent on the log-loss gradient converges to essentially the same solution as scikit-learn's optimized `sag` solver, given enough iterations
- Implementing the algorithm manually builds intuition for *why* logistic regression works, not just *how* to call it

## Future Improvements

- [ ] Add a loss-vs-iteration plot to visualize convergence
- [ ] Experiment with different learning rates and iteration counts
- [ ] Extend to mini-batch / stochastic gradient descent
- [ ] Generalize to multi-class classification (softmax regression)
- [ ] Add L2 regularization and compare against sklearn's regularized models

## License

This project is open source and available under the [MIT License](LICENSE).
