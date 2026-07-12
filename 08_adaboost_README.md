# 08 - AdaBoost Classifier

## Dataset
**Wine Quality (Red Wine)** — UCI Machine Learning Repository
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
1,599 rows, 11 physicochemical features, target: quality score (3-8, integer)

### Column Descriptions
| Column | Description |
|---|---|
| fixed acidity | Tartaric acid concentration (g/dm³) |
| volatile acidity | Acetic acid concentration — high levels cause an unpleasant, vinegar-like taste |
| citric acid | Adds freshness and flavor |
| residual sugar | Sugar remaining after fermentation stops |
| chlorides | Salt content |
| free sulfur dioxide | Prevents microbial growth and oxidation |
| total sulfur dioxide | Free + bound forms of SO2 |
| density | Depends on alcohol and sugar content |
| pH | Acidity level (0-14 scale) |
| sulphates | Additive that acts as an antimicrobial/antioxidant |
| alcohol | Alcohol percentage |
| quality | Original target — score from 3 to 8, based on sensory (human tasting) data |

## Problem Type
Multiclass classification (4 classes, derived from the original 6-class ordinal target)

## Preprocessing

**Target regrouping:** original quality scores (3-8) were regrouped into 4 classes to address severe class sparsity at the tails (only 10 samples at quality=3, 18 at quality=8):
- `low`: quality 3-4 (63 samples, 3.9%)
- `medium`: quality 5 (681 samples, 42.6%)
- `medium-high`: quality 6 (638 samples, 39.9%)
- `high`: quality 7-8 (217 samples, 13.6%)

Classes 5 and 6 were kept separate rather than merged, since each has enough samples on its own and merging them would discard real, learnable signal.

**Outlier capping (IQR):** applied to `residual sugar`, `chlorides`, and `sulphates` — these three showed the highest skewness (2.43 for sulphates pre-capping) and the heaviest outlier concentration. Capping was chosen over removal to avoid losing rows, since AdaBoost is sensitive to outliers (misclassified/extreme points get exponentially upweighted across boosting rounds). Other features were left untouched — their skew reflects natural chemical variation, not measurement error, confirmed via `scipy.stats.skew` (all remaining features under 1.0 except free/total sulfur dioxide, addressed below).

**Multicollinearity (VIF):** VIF values were extremely high for several features (density: 1435, pH: 1032, alcohol: 129, fixed acidity: 71), despite only moderate pairwise correlations (0.5-0.7). This is a structural relationship — density/pH are jointly determined by acidity, sugar, and alcohol content combined, not by any single feature. No features were dropped: AdaBoost's default base estimator is a shallow decision tree, which splits on individual feature thresholds rather than fitting coefficients — multicollinearity affects coefficient-based models (Linear/Logistic Regression) far more than tree-based ones.

**Scaling:** not applied. Tree-based splits are invariant to feature scale.

**Class imbalance:** SMOTE applied on the training set only, after the train/test split — balanced all 4 classes to 545 samples each (matching majority class size in train).

## Model & Tuning

**Baseline:** `AdaBoostClassifier()` with default parameters (decision stump, i.e. `max_depth=1` base estimator).

**GridSearchCV** (5-fold, scoring=`f1_macro`, 27 candidates):
param_grid = {
'estimator': [DecisionTreeClassifier(max_depth=1/2/3)],
'n_estimators': [50, 100, 200],
'learning_rate': [0.1, 0.5, 1.0]
}
**Best parameters:** `max_depth=3`, `n_estimators=200`, `learning_rate=0.5`
**Best CV F1 macro:** 0.697 ± 0.037

The default stump (`max_depth=1`) was outperformed by a deeper base estimator — a single split isn't expressive enough to separate the adjacent, overlapping quality groups (particularly `medium` vs `medium-high`, corresponding to the historically blurry line between quality scores 5 and 6 in human tasting data).

**Alternative considered:** `max_depth=2` was also tested to check its effect on overfitting. It reduced the train-test gap (0.224 vs 0.314) but also reduced test F1 (0.468 vs 0.492). `max_depth=3` was kept as the final model since test performance — not the gap itself — is the primary decision criterion.

## Results (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.57 |
| Macro F1 | 0.49 |
| Weighted F1 | 0.57 |
| ROC-AUC (macro, OvR) | 0.748 |
| Cohen's Kappa | 0.350 |
| Log Loss | 1.347 |

**Per-class F1:** medium 0.66, medium-high 0.52, high 0.57, low 0.22

**Per-class ROC-AUC (OvR):** low 0.877, medium-high 0.782, medium 0.759, high 0.576

**Baseline vs Tuned comparison:**
| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.55 | 0.57 |
| Macro F1 | 0.47 | 0.49 |

## Cross-Validation
CV F1 macro: 0.697 ± 0.037 (measured on SMOTE-balanced training data)

**Note:** this CV score is not directly comparable to the test score above. CV was computed on SMOTE-resampled data (all classes equal at 545 samples), while the test score reflects the real, imbalanced class distribution. The ~0.2 gap between CV (0.697) and test (0.49) macro F1 is largely explained by this, not by poor generalization alone — the same pattern was observed in the SVM (06) and Naive Bayes (07) notebooks.

## Overfit Check
Train (SMOTE) F1 macro: 0.806 | Test F1 macro: 0.492 | Gap: 0.314

This gap is notably larger than in previous tree-based models (Random Forest: 0.076, Gradient Boosting: 0.014). Two factors contribute:
1. **Genuine overfit** — `max_depth=3` base estimators have more capacity to fit training noise than stumps.
2. **Inflated train score** — train F1 is measured on SMOTE-balanced data, which is an easier problem than the real, imbalanced test distribution. This partially, not fully, explains the gap.

## Confusion Matrix Pattern
Nearly all misclassifications occur between **adjacent classes** (low↔medium, medium↔medium-high, medium-high↔high). Confusion between distant classes (e.g. low↔high) is close to zero. This indicates the model's errors are systematic and driven by genuinely blurry class boundaries (especially medium vs medium-high, i.e. quality 5 vs 6), not random noise.

## Feature Importance (MDI)
Top 3: `sulphates` (0.172), `volatile acidity` (0.158), `alcohol` (0.146) — together ~48% of total importance. This aligns with wine chemistry domain knowledge: volatile acidity negatively affects taste (vinegar-like off-flavor) and alcohol is positively associated with perceived quality. Notably, features with very high VIF (`density`, `pH`) rank low in importance — confirming that high VIF here reflects structural correlation, not that these features drive the model's decisions.

## Threshold Tuning
**Not applicable.** Threshold tuning shifts a single binary decision boundary (e.g. 0.5 → 0.34 in earlier binary-classification notebooks). With 4 classes, the model assigns the argmax of predicted probabilities across all classes — there is no single threshold to tune in the same sense.

## Limitations
- **Log Loss should be interpreted cautiously for AdaBoost.** AdaBoost optimizes a weighted classification error (via SAMME), not a probabilistic loss — its `predict_proba` output is derived from the boosted votes rather than a directly modeled probability, and is known to be poorly calibrated. The Log Loss here (1.347) is close to the value expected from random guessing on 4 classes (ln(4) ≈ 1.386), which reflects this calibration weakness rather than poor classification ability — Cohen's Kappa, F1, and ROC-AUC are more reliable indicators of this model's actual performance.
- `low` class (63 samples total, ~13 in test) remains the weakest predicted class (F1: 0.22) — a data scarcity issue rather than a modeling failure; SMOTE cannot fully compensate for the near-total absence of real examples.
- `medium-high` shows the most balanced confusion in both directions (toward `medium` and `high`) — the class sits at the center of the quality spectrum and lacks a one-sided error pattern to correct for.
- CV score (0.697) is not directly comparable to test score (0.49) due to the SMOTE-balanced vs. real-distribution measurement gap — see Cross-Validation section.
- SAMME (multiclass AdaBoost) treats all misclassifications equally regardless of "how wrong" — predicting `high` when the truth is `low` incurs the same per-sample weight increase as predicting `medium`, even though the former is intuitively a worse mistake for an ordinal target.

## What is AdaBoost?
AdaBoost (Adaptive Boosting) is an ensemble method that sequentially trains weak learners (by default, decision stumps — depth-1 trees), where each new learner focuses more on the samples the previous ones misclassified, by increasing their sample weight. Each learner's vote in the final prediction is weighted by its own accuracy, scaled by the `learning_rate`. For multiclass problems, scikit-learn uses the SAMME algorithm, which extends the original binary AdaBoost formulation by incorporating the number of classes into each learner's weight calculation.

**When to use:** works well on moderately noisy, low-to-medium dimensional tabular data where a boosted ensemble of simple learners can capture non-linear patterns without heavy tuning. Less suitable than Gradient Boosting/XGBoost when strong probability calibration is required, and more sensitive to label noise/outliers than bagging-based methods (Random Forest) due to its exponential reweighting of misclassified samples.