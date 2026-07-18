# 09 - K-Nearest Neighbors (KNN)

## Dataset: Mobile Price Classification

Source: Kaggle - Mobile Price Classification (2000 rows, 20 features + target)

A multiclass classification problem predicting a mobile phone's price segment based on its technical specifications.

### Column Descriptions

| Column | Description |
|---|---|
| battery_power | Battery capacity (mAh) |
| blue | Has Bluetooth (0/1) |
| clock_speed | Processor speed (GHz) |
| dual_sim | Has dual SIM support (0/1) |
| fc | Front camera resolution (MP) |
| four_g | Has 4G support (0/1) |
| int_memory | Internal memory (GB) |
| m_dep | Mobile depth/thickness (cm) |
| mobile_wt | Weight of mobile phone (grams) |
| n_cores | Number of processor cores |
| pc | Primary (rear) camera resolution (MP) |
| px_height | Pixel resolution height |
| px_width | Pixel resolution width |
| ram | RAM (MB) |
| sc_h | Screen height (cm) |
| sc_w | Screen width (cm) |
| talk_time | Longest battery single charge talk time (hours) |
| three_g | Has 3G support (0/1) |
| touch_screen | Has touch screen (0/1) |
| wifi | Has Wifi (0/1) |
| **price_range** | **Target** — 0: low cost, 1: medium cost, 2: high cost, 3: very high cost |

---

## Math & Intuition

KNN is a **distance-based, instance-based (lazy) learning algorithm**. It has no explicit training phase — it simply memorizes the training data. To classify a new point, it:

1. Computes the distance between the new point and every training point
2. Selects the `k` closest points (neighbors)
3. Assigns the class by majority vote among those neighbors (or a distance-weighted vote)

**Distance metric used (Manhattan / L1):**

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

Unlike Euclidean distance (straight-line distance), Manhattan distance sums the absolute differences along each axis — a "grid-like" distance. This metric was selected by GridSearchCV over Euclidean and Minkowski, likely because it performs more robustly on a feature space that mixes continuous variables with several binary (0/1) flags.

**Distance-weighted voting:**

$$w_i = \frac{1}{d(x, x_i)}$$

With `weights='distance'`, closer neighbors get a proportionally larger vote than farther ones, instead of every neighbor counting equally (`weights='uniform'`).

### Why Scaling Is Critical for KNN

KNN relies entirely on distance calculations. Without scaling, features with larger numeric ranges (e.g. `ram`: 256–3998) dominate the distance formula, while features with small ranges (e.g. `clock_speed`: 0.5–3.0) contribute almost nothing — regardless of their actual predictive power. StandardScaler (fit only on train) was applied to ensure every feature contributes proportionally to its own distribution, not its raw magnitude.

---

## Preprocessing Steps

- **Data quality check on suspicious zero values:**
  - `px_height = 0` (2 rows, 0.1%) → dropped (physically impossible, negligible size)
  - `sc_w = 0` (180 rows, 9%) → treated as a sentinel/missing value (screen width cannot be 0), replaced with NaN and filled using **class-wise median imputation** (same approach as the Pima Diabetes dataset in 02_logistic_regression)
- **Outlier check (IQR):** `fc` (18 outliers, values 17–19 MP) and `px_height` (2 outliers, values ~1949–1960) were inspected individually — both fall within realistic ranges for high-end phones and were **not capped** (same reasoning as the Bike Sharing dataset in 05_gradient_boosting: real variation, not data error)
- **Multicollinearity check:**
  - Correlation matrix revealed a strong `ram`–`price_range` correlation (0.92) — a feature-target relationship, not multicollinearity, and a strong early signal of `ram`'s predictive power
  - Moderate feature-feature correlations (`fc`-`pc`: 0.64, `sc_h`-`sc_w`: 0.58, `px_height`-`px_width`: 0.51, `four_g`-`three_g`: 0.58) — none severe enough to warrant dropping
  - VIF confirmed this: max VIF = 12.98 (`mobile_wt`), well below the extreme levels seen in other notebooks (e.g. 320+ in Bike Sharing, ~26,000 in Bank Marketing) — no features dropped
- **Train/test split:** 80/20, stratified (dataset was already perfectly balanced — 500/500/500/500 — but stratification was applied regardless as good practice)
- **Scaling:** StandardScaler, fit only on train, applied to both train and test
- **SMOTE:** **not applied** — target classes were already perfectly balanced (500 samples each), consistent with the project rule of only using SMOTE where genuinely needed rather than applying it by default

---

## Hyperparameter Tuning

GridSearchCV was run over `n_neighbors`, `weights`, and `metric`. An initial narrow grid selected `n_neighbors=100`, but extending the range revealed the CV score kept improving well beyond that point. A dedicated k-vs-CV-score sweep (k = 20 to 800) showed the curve peaking around **k=500** before declining — the classic bias-variance tradeoff curve for KNN (small k → high variance/noise sensitivity, very large k → high bias/underfitting).

**Final model chosen: `n_neighbors=300`, `weights='distance'`, `metric='manhattan'`**

k=300 was deliberately chosen over the CV-optimal k=500 because the marginal CV gain (+0.008) was negligible, and k=300 keeps the model closer to genuine "local neighborhood" behavior for a training set of only 1,598 rows.

**Why such a large k is still meaningful (not a red flag):** With `weights='distance'`, the vote weight of a neighbor decays as `1/distance`, so distant neighbors — even when technically included in the k-window — contribute almost nothing to the final vote. Effectively, the model still leans heavily on the nearest few points; the large k mainly adds stability against noise rather than diluting the decision with irrelevant neighbors. This large optimal k is itself a signal that the dataset has noisy/overlapping regions between adjacent price classes, requiring a wider neighborhood to make stable decisions.

**Note on train-set metrics:** With `weights='distance'`, every training point's nearest neighbor is itself (distance = 0), which forces training performance to a structural ~1.0 regardless of k. This is not genuine overfitting — the reliable overfit check here is the CV-vs-Test gap (CV: 0.7621 vs Test: 0.7735), which shows healthy generalization.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.7725 |
| Test F1 (macro) | 0.7735 |
| CV F1 (macro) | 0.7621 ± 0.0358 |
| ROC-AUC (macro, One-vs-Rest) | 0.9017 |
| Log Loss | 1.1164 (random baseline for 4 classes ≈ 1.386) |

**Per-class F1:** Class 0 = 0.90, Class 1 = 0.68, Class 2 = 0.67, Class 3 = 0.84

### Confusion Matrix Pattern

Errors are concentrated **exclusively between adjacent classes** (0↔1, 1↔2, 2↔3) — there is zero confusion between non-adjacent classes (e.g. 0↔3, 0↔2, 1↔3). This confirms that `price_range` behaves as an **ordinal** variable in the feature space: the model's mistakes are consistently "close" misclassifications rather than random ones. Classes 1 and 2 (the middle categories) show weaker performance because they can be confused from both sides, while classes 0 and 3 (the extremes) only have one adjacent class to be confused with.

### Feature Importance (Permutation Importance)

KNN has no native feature importance (no coefficients, no tree splits), so **Permutation Importance** was used instead — measuring the drop in F1 macro when each feature is randomly shuffled (averaged over 10 repeats for stability).

`ram` dominates overwhelmingly (importance = 0.493), roughly 11x higher than the second-ranked `battery_power` (0.045). This aligns directly with the 0.92 correlation between `ram` and `price_range` found earlier. Several features (`m_dep`, `sc_h`, `four_g`, `int_memory`) showed near-zero or slightly negative importance — indicating negligible practical contribution to the model's decisions, not that they actively harm it.

### ROC-AUC vs F1 Gap

ROC-AUC (0.90) is notably higher than F1 macro (0.77). This is expected: ROC-AUC measures the model's ability to *rank* the correct class above others (soft distinction), while F1 depends on the final hard decision (argmax). The gap indicates the model can usually distinguish classes reasonably well in probability space, even when its final hard classification is wrong at the boundary between adjacent classes.

### Log Loss

1.1164, compared to the random-guess baseline of ≈1.386 for 4 balanced classes — better than random but not sharply calibrated. This likely reflects that KNN's `predict_proba` output is a raw neighbor-vote ratio rather than a calibrated probability estimate, especially with a large k, producing "flatter" probability distributions in the overlapping middle classes — a similar calibration weakness to the one observed with AdaBoost in 08.

---

## When to Use KNN

- Small-to-medium datasets where the feature space has clear local structure
- Problems where decision boundaries are non-linear and not easily captured by a parametric model
- When interpretability is less important than local pattern-matching
- Suitable as a quick, non-parametric baseline before moving to more complex models

## Limitations

- **Computationally expensive at prediction time** — no real training phase, but every prediction requires distance calculations against the full training set, making it slow for large datasets
- **Curse of dimensionality** — as the number of features grows, distances between points become less meaningful, degrading performance
- **Sensitive to feature scale** — scaling is not optional, it is a prerequisite for meaningful distance calculations
- **No built-in feature importance** — requires post-hoc methods like Permutation Importance
- **Weak probability calibration** — `predict_proba` output is a neighbor-vote ratio, not a true probabilistic model, so Log Loss and probability-based metrics should be interpreted with caution
- **Sentinel-like zero values in `sc_w`** required class-wise median imputation, similar to the Pima Diabetes dataset — a reminder that "0" is not always a genuine zero in real-world tabular data

---

## Techniques Used

`StandardScaler`, `IQR-based outlier inspection`, `Correlation Matrix`, `VIF`, `Stratified Train/Test Split`, `GridSearchCV`, `Manhattan / Euclidean / Minkowski distance metrics`, `Uniform vs Distance-weighted voting`, `K-vs-CV-score sweep (bias-variance tradeoff curve)`, `Confusion Matrix`, `Multiclass ROC-AUC (One-vs-Rest, label_binarize aligned to model.classes_)`, `Log Loss`, `Permutation Importance`, `5-fold Cross-Validation`