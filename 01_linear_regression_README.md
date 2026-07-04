# 01 - Linear Regression (+ Polynomial, Ridge, Lasso, ElasticNet)

## Overview
This notebook applies **Linear Regression** and its regularized/non-linear variants (Polynomial, Ridge, Lasso, ElasticNet) to predict median house value using the California Housing dataset. As the first model in this portfolio, it establishes the baseline workflow (outlier capping, correlation-based multicollinearity check, scaling, target transformation) reused across later notebooks.

## Dataset
**Source:** California Housing dataset (1990 US Census data, loaded via `sklearn.datasets.fetch_california_housing`)
**Size:** 20,640 rows, 8 features.
**Target:** `target` - median house value for California districts (in $100,000s).

### Column Descriptions
| Column | Description |
|---|---|
| `MedInc` | Median income in the block group (in $10,000s) — strongest predictor, corr=0.69 with target |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household — *dropped, see below* |
| `Population` | Block group population — outlier-capped (max was 35,682) |
| `AveOccup` | Average number of household members — outlier-capped (max was 1,243) |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `target` | **Target** - median house value for the block group |

## Preprocessing Decisions

- **Outlier capping (IQR, upper bound only)** applied to `Population` and `AveOccup` — both showed extreme maximum values (35,682 and 1,243 respectively) far beyond the rest of the distribution. Values above `Q3 + 1.5×IQR` were clipped rather than removing rows, to preserve sample size.
- **`AveBedrms` dropped** based on correlation analysis: 0.82 correlation with `AveRooms` (redundant information) while correlation with the target itself was only -0.05 (weak predictive value) — a formal VIF check was not run in this notebook, the correlation heatmap was sufficient to justify the drop.
- **Feature scaling (StandardScaler)** applied, fit only on the training set and applied (transform only) to the test set to avoid data leakage.
- **Target distribution check:** histogram + Q-Q plot showed the target was right-skewed.
- **Yeo-Johnson transformation applied to the target** (not the features) using `PowerTransformer` — chosen over Box-Cox because Yeo-Johnson supports zero/negative values. This improved R² from 0.6432 (untransformed target) to 0.6619.

## Model & Math

**Linear Regression** fits a hyperplane minimizing the sum of squared residuals:
y = β0 + β1x1 + β2x2 + ... + βnxn + ε
Cost = (1/n) Σ(y_actual - y_predicted)²
**Polynomial Regression** expands the feature set with polynomial/interaction terms (e.g. `x²`, `x1×x2`) before fitting a linear model, allowing it to capture non-linear relationships.

**Ridge Regression** adds an L2 penalty:
Cost = MSE + α Σβᵢ²
Shrinks coefficients toward zero without eliminating them — helps with multicollinearity and overfitting.

**Lasso Regression** adds an L1 penalty:
Cost = MSE + α Σ|βᵢ|


Can shrink coefficients to exactly zero, performing automatic feature selection.

**ElasticNet** combines both penalties, useful when correlated features exist and neither pure Ridge nor pure Lasso is clearly preferable.

**When to use:** Linear Regression when the relationship is approximately linear and interpretability matters. Polynomial when curvature/non-linearity is present. Ridge/Lasso/ElasticNet when overfitting or feature selection is a concern — in this dataset, none of the three improved on plain Linear Regression, because the core limitation here was non-linearity, not overfitting.

## Results

| Model | MAE | RMSE | Test R² | Train R² | Overfit Gap |
|---|---|---|---|---|---|
| Linear Regression (raw target) | 0.505 | 0.684 | 0.643 | — | — |
| Linear Regression (Yeo-Johnson target) | 0.443 | 0.581 | 0.662 | 0.676 | 0.014 |
| Polynomial (degree=2) | 0.397 | 0.538 | **0.710 (best)** | 0.729 | 0.020 |
| Polynomial (degree=3) | — | — | 0.700 | — | 0.057 |
| Ridge (best α=0.01) | — | — | ≈0.662 | — | ≈0.014 |
| Lasso (best α=0.001) | 0.444 | 0.666 | 0.662 | 0.676 | 0.014 |
| ElasticNet (α=0.001, l1_ratio=0.1) | 0.443 | 0.581 | 0.662 | — | 0.015 |

**Cross-validation (5-fold, on Yeo-Johnson-transformed target):** Mean R² stable, std=0.0054 — model generalizes consistently across folds.

**Polynomial degree comparison:** degree=2 expanded features from 7 to 35 and gave the best result. degree=3 expanded to 119 features and **overfit increased** (gap grew from 0.020 to 0.057) while test R² actually dropped slightly (0.710 → 0.700) — confirming degree=2 as the optimal complexity for this dataset.

**Lasso behavior at higher alpha:** at α=0.001 no features were eliminated; increasing α to 0.1 eliminated 3 features (R² dropped to 0.497) and at α=1 it eliminated 7 features (R² collapsed to ≈0). This confirms all 7 remaining features carry genuine signal — Lasso's feature-selection strength isn't needed here since there are no irrelevant features to prune.

**Why Ridge/Lasso/ElasticNet didn't beat plain Linear Regression:** all three converged to best alpha values near zero, meaning minimal regularization was preferred. This tells us the dataset's limitation was never overfitting — it was that the true relationship between features and price has non-linear structure (confirmed by Polynomial Regression's clear improvement), which regularization alone cannot fix.

## Winner
**Polynomial Regression (degree=2), R² = 0.710** — the only approach that meaningfully improved over plain Linear Regression, by capturing non-linear feature interactions.

## Limitations
- Even the best model leaves ~29% of variance unexplained — likely due to factors not in this dataset (school quality, crime rate, local amenities).
- Only degree 2 and 3 polynomial expansions were tested; higher degrees were not explored given the clear overfitting trend already visible at degree=3.
- Multicollinearity check was based on a correlation heatmap rather than a formal VIF calculation.

## Techniques Used
IQR-based outlier capping, correlation-based multicollinearity check, StandardScaler, Yeo-Johnson target transformation (PowerTransformer), polynomial feature expansion, Ridge/Lasso/ElasticNet regularization (GridSearchCV), 5-fold cross-validation, residual plot analysis, overfit gap tracking.