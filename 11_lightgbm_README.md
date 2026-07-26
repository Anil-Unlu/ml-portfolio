# 11 - LightGBM: Diamond Price Prediction

## Dataset
Diamonds dataset (seaborn), 53,940 rows → 53,920 after cleaning, 9 features → price (regression).

## Preprocessing
- Dropped 20 rows with x/y/z = 0 (physically impossible, sentinel values)
- Ordinal encoding for `cut`, `color`, `clarity` (industry-standard quality order, not alphabetical) — kept as integer-coded categoricals, **not one-hot encoded**
- No scaling (tree-based, scale-invariant)
- No SMOTE (regression problem)

## Model
- LightGBM with native `categorical_feature` support for cut/color/clarity
- Early stopping (50 rounds) on a held-out validation split (90/10 of train) → best iteration: 440
- `num_iteration=model.best_iteration` used at predict time to avoid using post-stopping trees

## Results
| Metric | Value |
|---|---|
| Test RMSE | 524.11 |
| Test MAE | 269.08 |
| Test R² | 0.9829 |
| Train RMSE | 425.46 |
| Overfit gap (RMSE) | 98.64 |
| CV RMSE (5-fold) | 536.95 ± 15.07 |

## Feature Importance (Gain)
`y` (width) and `carat` dominant, `clarity` third. `x`, `depth`, `table`, `cut` near-zero — masked by multicollinearity with `y`/`carat` (same size signal), same effect seen with atemp/temp in Gradient Boosting and MonthlyCharges/TotalCharges in Random Forest.

## LightGBM - XGBoost — When to Prefer Which

- **Categorical features:** LightGBM handles categorical features natively (`categorical_feature` parameter with integer-coded categories), avoiding the need for one-hot encoding. XGBoost is commonly used with one-hot encoding for categorical variables, which can substantially increase feature dimensionality when dealing with high-cardinality categories.

- **Tree growth:** LightGBM grows trees leaf-wise (best-first), splitting the leaf with the largest loss reduction regardless of depth. This often achieves higher accuracy with fewer trees but can increase overfitting risk on small datasets, requiring careful tuning of parameters such as `num_leaves` and `min_data_in_leaf`. XGBoost grows trees level-wise, producing more balanced trees and generally providing more conservative and stable behavior.

- **Speed on large data:** LightGBM is often faster and more memory-efficient on large datasets due to its histogram-based algorithm, leaf-wise growth strategy, and optimizations such as Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).

- **Small datasets:** LightGBM's leaf-wise growth strategy can overfit more easily on small datasets (e.g., a few thousand rows), especially without proper regularization. XGBoost's level-wise tree growth often provides a safer default with more robust generalization in these scenarios.

- **Rule of thumb:** Prefer LightGBM when working with large datasets and/or categorical features where avoiding one-hot encoding is beneficial. Prefer XGBoost for smaller datasets or when more conservative and stable tree growth is preferred.

## Early Stopping vs GridSearchCV

Used LightGBM's native `early_stopping` on a validation set to automatically determine the optimal number of boosting iterations (`n_estimators`). This prevents unnecessary training once validation performance stops improving and is computationally cheaper than exhaustively searching over boosting rounds with GridSearchCV. Other hyperparameters can still be tuned separately using methods such as GridSearchCV, RandomizedSearchCV, or Bayesian optimization.


## Limitations
- No formal multicollinearity check (VIF/correlation) performed — carat, x, y, z are visibly related (all measure diamond size), likely explaining why `x` shows near-zero importance despite being a size feature (model captures the size signal mainly through `y`/`carat`)
- No hyperparameter grid search performed — early stopping + default learning_rate=0.05 was sufficient given already strong baseline (R²=0.983); left as a possible future improvement