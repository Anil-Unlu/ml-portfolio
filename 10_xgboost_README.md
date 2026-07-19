# 10 - XGBoost: Stellar Classification (SDSS17)

## Dataset

**Source:** SDSS17 Stellar Classification dataset (Sloan Digital Sky Survey, Data Release 17), sourced via a public GitHub mirror (`SivadithiyanOfcl/SDSS17`, originally from Kaggle/fedesoriano).

100,000 observations of space, each described by 17 feature columns plus a target class identifying the object as a **galaxy**, **star**, or **quasar (QSO)** based on spectral characteristics.

### Column descriptions

| Column | Description |
|---|---|
| `alpha` | Right Ascension angle (J2000 epoch) |
| `delta` | Declination angle (J2000 epoch) |
| `u`, `g`, `r`, `i`, `z` | Photometric magnitudes in the ultraviolet, green, red, near-infrared, and infrared filters |
| `run_ID` | Run number identifying the specific scan |
| `rerun_ID` | Rerun number specifying how the image was processed |
| `cam_col` | Camera column identifying the scanline within the run |
| `field_ID` | Field number identifying each field |
| `spec_obj_ID` | Unique identifier for optical spectroscopic objects |
| `redshift` | Redshift value based on the increase in wavelength |
| `plate` | Plate ID identifying each plate used in the survey |
| `MJD` | Modified Julian Date of the observation |
| `fiber_ID` | Fiber ID that directed light onto the focal plane |
| `class` | Target: `GALAXY`, `QSO`, or `STAR` |

## Preprocessing

- **Dropped pure identifier columns:** `rerun_ID` (zero variance, single unique value), `obj_ID` (catalog identifier, not a physical property, near-unique with some repeated observations), `spec_obj_ID` (fully unique per row — a row identifier with no generalization value). Keeping identifiers like these risks the model memorizing IDs instead of learning physical signal.
- **Sentinel value handling:** `u`, `g`, `z` contained a `-9999` placeholder (SDSS's code for a failed measurement). Affected exactly 1 row (all three columns simultaneously) out of 100,000 — dropped rather than imputed given the negligible impact.
- **Outlier check (IQR):** Photometric magnitude columns showed very low outlier rates (0.06%-0.32%) consistent with genuinely bright/faint objects rather than errors — no capping applied. `redshift` showed a much higher rate (8.99%), but grouping by class revealed this is a real physical signal, not noise: `STAR` redshift clusters near 0, `GALAXY` averages ~0.42, and `QSO` averages ~1.72 (quasars are the most distant, fastest-receding objects observed). No capping applied to `redshift`.
- **Multicollinearity (VIF):** Photometric bands (`u`,`g`,`r`,`i`,`z`) showed very high VIF (up to ~5000), and `plate`/`MJD` were highly correlated (0.97) — both structural relationships inherent to the data. Since XGBoost is tree-based (splits on individual features rather than fitting coefficients), high VIF does not bias the model, so no columns were dropped.
- **Target encoding:** `class` label-encoded to `GALAXY=0`, `QSO=1`, `STAR=2` (alphabetical order via `LabelEncoder`).
- **Train/test split:** 80/20, stratified on the target to preserve class proportions.
- **Class imbalance:** Moderate (`GALAXY` 59%, `STAR` 22%, `QSO` 19%). SMOTE applied on the training set only, after the split.
- **Scaling:** Not applied — XGBoost is tree-based and scale-invariant.

## Model: XGBoost

### Math & intuition

XGBoost (Extreme Gradient Boosting) builds an ensemble of decision trees sequentially, where each new tree is trained to correct the residual errors of the combined trees before it — the same core idea as the Gradient Boosting model in `05_gradient_boosting.ipynb`. What distinguishes XGBoost:

- It uses a **second-order Taylor approximation** of the loss function (both gradient and curvature/Hessian), giving more precise updates than standard gradient boosting.
- Built-in **L1/L2 regularization** on leaf weights, penalizing overly complex trees directly in the objective function.
- **Optimized tree-pruning** (depth-first growth with post-hoc pruning via a "gain threshold"), rather than greedy pre-pruning.
- Native handling of sparse data and missing values.

For this multiclass problem, XGBoost was configured with `objective='multi:softprob'`, which outputs a probability distribution over the 3 classes (via softmax) rather than a single hard label.

### When to use

Tabular data, especially with mixed feature scales, non-linear relationships, and no strict need for interpretability. Particularly strong on structured/tabular problems where feature interactions matter — one of the most common choices in competitive ML (Kaggle) for exactly this reason.

## Hyperparameters

**No tuning was performed.** The baseline model (default hyperparameters, only `objective`, `num_class`, `eval_metric`, and `random_state` set) already achieved very strong, well-generalizing performance (F1 macro 0.974, overfit gap 0.018). Given the marginal expected gain from tuning versus the added risk of overfitting and complexity, this was a deliberate decision — consistent with the project's philosophy elsewhere of not over-optimizing when a simpler model already generalizes well.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.98 |
| Test F1 macro | 0.9743 |
| Test ROC-AUC (macro, OvR) | 0.9956 |
| Per-class ROC-AUC | GALAXY 0.994, QSO 0.992, STAR 1.000 |
| Cohen's Kappa | 0.9603 |
| Log Loss | 0.0782 |
| Train F1 macro | 0.9919 |
| Overfit gap (Train − Test F1) | 0.0176 (healthy) |
| CV F1 macro (5-fold) | 0.9726 ± 0.0010 |

### Confusion matrix insights

- `STAR` is classified almost perfectly (4,305/4,319 correct), with essentially zero confusion against `QSO` — consistent with its near-zero redshift acting as a strong separator.
- The main source of error is `GALAXY` ↔ `QSO` confusion (187 + 200 misclassifications), which aligns with the overlapping tail of their redshift distributions — low-redshift quasars and high-redshift galaxies sit closer together in feature space.
- `GALAXY` ↔ `STAR` confusion is minimal (47 + 0), as expected given their very different redshift regimes.

### Feature importance

`redshift` alone accounts for ~83% of the model's feature importance, consistent with its strong class-separating power observed during EDA. The photometric bands (`g`, `u`, `z`, `i`, `r`) account for most of the remainder. All ID/metadata columns (`plate`, `MJD`, `fiber_ID`, `run_ID`, `cam_col`, `field_ID`) contributed negligibly (each under 1.1% importance) — this was specifically checked as a potential leakage risk (since these are observation-metadata rather than physical properties) and found to be a non-issue: the model learns from genuine physical/spectral signal, not from scan/plate identifiers.

## Cross-validation methodology note

SMOTE was applied inside an `imblearn` Pipeline together with the model, refit separately within each fold via `cross_val_score`. Since each fold's validation set remains untouched by SMOTE (only the training portion of each fold is resampled), the CV scores are measured on real, imbalanced data — directly comparable to the single-split test score, unlike the SVM/Naive Bayes/AdaBoost cases where CV was measured on SMOTE-balanced data. The near-identical CV mean (0.9726) and test F1 (0.9743) with a very small CV std (0.0010) confirms the model is stable and not overfit to a particular split.

## Limitations

- `redshift` dominating feature importance (~83%) means the model is heavily reliant on a single feature — while physically justified, this reduces robustness if `redshift` measurements were ever noisy or unavailable for new data.
- Threshold tuning is not applicable (multiclass problem, no single decision threshold).
- No hyperparameter tuning was performed; a tuned model might close the small remaining gap on the `GALAXY`↔`QSO` boundary cases, though the expected gain is likely marginal given the already-low overfit gap.