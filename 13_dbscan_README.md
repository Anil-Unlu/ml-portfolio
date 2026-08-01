# 13. DBSCAN — Customer Segmentation (Credit Card Behavioral Data)

## Overview
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) applied to unsupervised customer segmentation on credit card usage behavior. Unlike centroid-based methods (e.g. K-Means), DBSCAN groups points based on density connectivity and does not require the number of clusters to be specified in advance. It also explicitly identifies outliers as "noise" rather than forcing every point into a cluster.

## Dataset
**Credit Card Dataset for Clustering** (UCI / Kaggle) — 8,950 active credit card holders, 18 behavioral variables summarizing usage over 6 months.

| Column | Description |
|---|---|
| BALANCE | Remaining balance available for purchases |
| BALANCE_FREQUENCY | How frequently the balance is updated (0–1 score) |
| PURCHASES | Total purchase amount |
| ONEOFF_PURCHASES | Maximum one-off (non-installment) purchase amount |
| INSTALLMENTS_PURCHASES | Total installment purchase amount |
| CASH_ADVANCE | Cash advance amount given to the user |
| PURCHASES_FREQUENCY | How frequently purchases are made (0–1) |
| ONEOFF_PURCHASES_FREQUENCY | Frequency of one-off purchases (0–1) |
| PURCHASES_INSTALLMENTS_FREQUENCY | Frequency of installment purchases (0–1) |
| CASH_ADVANCE_FREQUENCY | Frequency of cash advance usage (0–1) |
| CASH_ADVANCE_TRX | Number of cash advance transactions |
| PURCHASES_TRX | Number of purchase transactions |
| CREDIT_LIMIT | Credit card limit |
| PAYMENTS | Total payments made |
| MINIMUM_PAYMENTS | Minimum payments made |
| PRC_FULL_PAYMENT | Percentage of months with full balance payment |
| TENURE | Number of months as a cardholder |

`CUST_ID` was dropped — pure identifier, no behavioral signal.

## Preprocessing

**Missing values:** `CREDIT_LIMIT` (1 row) and `MINIMUM_PAYMENTS` (313 rows, 3.5%) imputed with median.

**Skewness correction (Yeo-Johnson):** Most monetary/count columns were heavily right-skewed (skew up to ~15 for `MINIMUM_PAYMENTS`). Since DBSCAN is distance-based, extreme skew allows a few outliers to dominate Euclidean distance calculations. `PowerTransformer(method='yeo-johnson')` was applied — chosen over a fixed `log1p` transform because it learns a per-column optimal λ and natively handles zero values without needing a `+1` offset. Applied to 14 columns; skew improved to roughly ±1 range for all. `BALANCE_FREQUENCY` and `TENURE` retained residual skew due to a genuine ceiling effect (majority of customers at frequency=1.0 or tenure=12) rather than a transformable distribution shape — left as-is, noted as a structural property of the data.

**Multicollinearity (VIF):** Initial VIF calculation without an intercept term produced inflated values (a known statsmodels pitfall — `variance_inflation_factor` requires a constant column to avoid attributing shared mean variance across all features). After adding `add_constant()`, iterative elimination (threshold=10) dropped 3 columns: `CASH_ADVANCE_TRX` (VIF=41.9), `PURCHASES_TRX` (VIF=17.6), `ONEOFF_PURCHASES` (VIF=13.3) — each was redundant with its corresponding amount+frequency pair already present in the remaining features. Final feature set: 14 columns, all VIF < 10.

**Scaling:** StandardScaler applied to all 14 features. No train/test split was performed — this is an exploratory/unsupervised task (consistent with 12_pca), so `fit_transform` was applied on the full dataset.

**Dimensionality reduction (PCA):** Reduced from 14 to 5 components (cumulative explained variance = 80.97%), primarily to mitigate the curse of dimensionality — in high dimensions, Euclidean distances become less discriminative, which directly undermines DBSCAN's core mechanism (eps-based density thresholds lose meaning). PC1 (32.1%) and PC2 (23.8%) were also used for 2D visualization.

**Component interpretation:**
- **PC1** — positive loadings on `PURCHASES_FREQUENCY`, `PURCHASES`, `INSTALLMENTS_PURCHASES`; negative loadings on `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`. Represents a **purchase-oriented ↔ cash-advance-oriented** behavioral axis.
- **PC2** — positive loadings on `BALANCE`, `MINIMUM_PAYMENTS`, `PAYMENTS`, `CREDIT_LIMIT`. Represents overall **account size/volume**.

## Parameter Selection

**k-distance plot:** With `min_samples=10`, the sorted k-th nearest neighbor distance curve showed a gradual, non-sharp bend around eps≈0.8, consistent with the PC1–PC2 scatter plot showing one continuous density blob rather than well-separated regions — an early signal that this dataset lacks the naturally separated density structure DBSCAN performs best on.

**Grid search:** `eps` ∈ [0.3, 1.2] × `min_samples` ∈ [5, 10, 15, 20], evaluated jointly on Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, and noise percentage (all cluster-quality metrics computed on non-noise points only).

**Key finding — single-metric selection is misleading:** The highest Silhouette scores (up to 0.75) occurred at low eps values, but were degenerate: 66–87% of points were classified as noise, meaning only a handful of extremely tight micro-clusters remained. High Silhouette in this regime reflects near-trivial tight groupings, not meaningful segmentation. Cross-checking against noise% and Calinski-Harabasz was necessary to filter these out — a single metric in isolation would have selected an unusable result.

**Final parameters: eps=0.8, min_samples=20** — chosen for the best balance across all four indicators (moderate noise%, non-degenerate cluster count, relatively strong Calinski-Harabasz) rather than the single highest score on any one metric.

## Results

- **Clusters found:** 5 (+ noise)
- **Noise points:** 1,803 (20.15%)
- **Silhouette Score:** 0.2257
- **Calinski-Harabasz Index:** 1651.50
- **Davies-Bouldin Index:** 0.9859

**Cluster sizes:** Cluster 0 (n=4,009), Cluster 1 (n=2,226), Cluster 2 (n=792), Cluster 3 (n=89), Cluster 4 (n=31), Noise (n=1,803)

**Cluster profiles (behavioral interpretation):**
| Cluster | Profile |
|---|---|
| 0 | Purchase-driven, regular spenders — high `PURCHASES_FREQUENCY`, near-zero `CASH_ADVANCE` |
| 1 | Cash-advance-oriented users — low `PURCHASES`, high `CASH_ADVANCE` |
| 2 | High-volume "power users" — high on both purchases and cash advance, highest `BALANCE`/`CREDIT_LIMIT` |
| 3 | Small group, cash-advance-only, lower tenure — newer customers relying solely on cash advances |
| 4 | Small group, purchase-only, lower tenure, lowest balance — newer, low-volume shoppers |
| Noise | Mixed/borderline profiles not fitting any dominant pattern — transitional behavior between clusters |

## Interpretation: Silhouette is Weak, But Clusters Are Business-Meaningful

The final Silhouette Score (0.2257) falls in the "weak structure" range by conventional thresholds (Kaufman & Rousseeuw). However, visualizing the clusters on the PC1–PC2 plane shows a clean, interpretable separation that directly aligns with the PC1 axis (purchase-oriented vs. cash-advance-oriented) and PC2 axis (account volume) — Cluster 0 and Cluster 1 occupy opposite ends of PC1, Cluster 2 sits at high PC2, and noise points concentrate at the boundaries between groups.

This is an honest, expected outcome given the underlying data: credit card behavior is a **continuous spectrum**, not a set of naturally disjoint density regions. DBSCAN found a real and business-usable pattern, but the quantitative separation is inherently limited by the data's structure rather than by parameter choice. This limitation directly motivates the next notebook (HDBSCAN), which handles variable-density clusters more flexibly than DBSCAN's single global eps.

## Limitations

- No train/test split — this is an exploratory/unsupervised clustering task, not a predictive pipeline.
- SMOTE, threshold tuning not applicable (unsupervised, no target/class imbalance concept).
- Silhouette/Calinski-Harabasz/Davies-Bouldin are all sensitive to degenerate solutions (e.g. extreme noise%) — must be evaluated jointly, never in isolation.
- DBSCAN uses a single global `eps`, which assumes roughly uniform density across the dataset — the data in this notebook does not fully satisfy that assumption, contributing to the moderate Silhouette score.
- Cluster interpretation is based on Yeo-Johnson-transformed (not raw-scale) feature means; useful for relative comparison between clusters but not directly reportable as raw business figures without inverse-transforming.