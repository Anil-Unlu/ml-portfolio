# 05 - Gradient Boosting Regressor

## Overview
This notebook applies **Gradient Boosting Regression** to predict hourly bike rental demand using the UCI Bike Sharing Dataset. Unlike previous models in this portfolio, this is the first **regression** task since Linear Regression (01), giving a direct comparison point for how boosting handles non-linear relationships.

## Dataset
**Source:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
**Size:** 17,379 rows, hourly bike rental records from a bike-sharing system spanning 2011-2012.
**Target:** `cnt` - total number of bike rentals in a given hour (continuous).

### Column Descriptions
| Column | Description |
|---|---|
| `season` | Season (1=winter, 2=spring, 3=summer, 4=fall) — *dropped, see below* |
| `yr` | Year (0=2011, 1=2012) |
| `mnth` | Month (1-12) |
| `hr` | Hour of day (0-23) |
| `holiday` | Whether the day is a public holiday (0/1) |
| `weekday` | Day of week (0=Sunday ... 6=Saturday) |
| `workingday` | 1 if day is neither weekend nor holiday |
| `weathersit` | Weather category (1=clear, 2=mist/cloudy, 3=light rain/snow, 4=heavy rain/storm) |
| `temp` | Normalized temperature (0-1 scale, actual value / 41°C) |
| `atemp` | Normalized "feels-like" temperature — *dropped, see below* |
| `hum` | Normalized humidity (0-1 scale) |
| `windspeed` | Normalized wind speed (0-1 scale) |
| `casual` | Rentals by casual (non-registered) users — *dropped, see below* |
| `registered` | Rentals by registered users — *dropped, see below* |
| `cnt` | **Target** - total rentals (`casual + registered`) |

## Preprocessing Decisions

- **`casual` and `registered` dropped** — confirmed via `casual + registered == cnt` (verified True on the full dataset). These columns are direct components of the target and would cause data leakage if kept.
- **`instant` and `dteday` dropped** — `instant` is a row index with no predictive value; `dteday` is redundant since `yr`, `mnth`, `hr`, `weekday` are already extracted from it.
- **`atemp` dropped** — near-perfect correlation with `temp` (r=0.99, VIF=354 for atemp, 320 for temp). Kept `temp` since `hum` and `windspeed` already capture the humidity/wind component of "feels-like" temperature separately.
- **`season` dropped** — high correlation with `mnth` (r=0.83, VIF=21 for season, 15 for mnth). Kept `mnth` for finer granularity (12 categories vs 4).
- **No outlier removal** — `cnt` and `windspeed` showed skew (1.28 and 0.58 respectively) and ~2-3% IQR-flagged outliers, but these represent genuine high-demand hours (e.g. summer evening rush), not data errors. Removing them would discard the exact signal the model needs to learn. Tree-based models like GBM are also not sensitive to this kind of skew the way distance-based models are.
- **`weathersit` one-hot encoded** (nominal, not truly ordinal) — `mnth`, `hr`, `weekday` kept as numeric since they carry meaningful ordinal/cyclical signal that GBM can split on naturally.
- **Rare category note:** `weathersit=4` (heavy rain/storm) has only 3 observations in the entire dataset. Train/test columns were aligned (`align()`) to avoid mismatched dummy columns if this rare category didn't appear in both splits.

## Model & Math

**Gradient Boosting** builds an ensemble of shallow decision trees **sequentially**, where each new tree learns to correct the residual errors of the combined previous trees, rather than being trained independently (as in Random Forest / bagging).

Simplified update rule:F_m(x) = F_{m-1}(x) + learning_rate × h_m(x)


where `h_m(x)` is the new tree fit on the residuals of `F_{m-1}(x)`.

**Key hyperparameters:**
- `n_estimators`: number of sequential trees
- `learning_rate`: shrinkage factor applied to each tree's contribution (lower = more conservative, more trees needed)
- `max_depth`: depth of each individual tree (controls interaction complexity)
- `min_samples_leaf`: minimum samples required in a leaf node (controls overfitting)

**Intuition:** instead of averaging many independent "opinions" (Random Forest), Gradient Boosting builds a chain of "correctors" — each tree specializes in fixing what previous trees got wrong. This makes it powerful but more prone to overfitting if not regularized (via `learning_rate`, `max_depth`, `min_samples_leaf`).

**When to use:** structured/tabular data with non-linear relationships and feature interactions, when predictive accuracy matters more than interpretability, and when enough compute/time is available for sequential training (harder to parallelize than Random Forest).

## Hyperparameter Tuning
GridSearchCV (5-fold CV, scoring=`neg_root_mean_squared_error`), 81 combinations tested.

**Best parameters:** `learning_rate=0.2, max_depth=4, min_samples_leaf=10, n_estimators=300`
**Best CV RMSE:** 46.057 ± 1.444

## Results

| Metric | Baseline (default params) | Tuned |
|---|---|---|
| Train RMSE | 71.46 | 39.60 |
| Test RMSE | 69.25 | 44.01 |
| Train R² | 0.846 | 0.953 |
| Test R² | 0.849 | 0.939 |
| Train MAE | 48.44 | 25.57 |
| Test MAE | 46.95 | 27.93 |

**Overfit check:** Train/Test R² gap is only 0.014 — no meaningful overfitting despite the tuned model choosing parameters at the edge of the search grid.

**Residual analysis:** Residual mean ≈ -0.58 (no systematic bias), residual std ≈ 44.0 (consistent with RMSE). Residual distribution is symmetric and centered at zero. However, residual variance increases at higher predicted values (mild heteroscedasticity) — the model is noticeably less precise during peak-demand hours, likely because high-demand observations are underrepresented in the data (right-skewed target, skew=1.28).

## Feature Importance (MDI)

| Feature | Importance |
|---|---|
| `hr` | 0.606 |
| `temp` | 0.133 |
| `workingday` | 0.099 |
| `yr` | 0.089 |
| `mnth` | 0.024 |
| `hum` | 0.022 |
| `weathersit_3` | 0.013 |
| `weekday` | 0.009 |
| `windspeed` | 0.003 |
| `holiday` | 0.001 |
| `weathersit_2` | 0.001 |
| `weathersit_4` | 0.000 |

Hour of day (`hr`) dominates the model's decisions (60.6% of total importance), consistent with the strong commute-pattern seasonality visible in EDA. The top 4 features (`hr`, `temp`, `workingday`, `yr`) account for ~93% of total importance. `weathersit_4` shows zero importance, expected given it represents only 3 observations in the dataset.

## Limitations
- Model accuracy degrades somewhat during peak-demand hours (heteroscedasticity in residuals).
- The rare `weathersit=4` category is not meaningfully learned due to insufficient samples.
- No external/event-based features (e.g. holidays beyond the binary flag, local events) — some high-demand spikes may be driven by factors not present in this dataset.

## Techniques Used
IQR-based outlier inspection, VIF multicollinearity check, one-hot encoding (nominal features), GridSearchCV, 5-fold cross-validation, MDI feature importance, residual analysis, heteroscedasticity check.
