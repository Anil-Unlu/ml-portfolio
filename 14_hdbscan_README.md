# 13 - HDBSCAN (Hierarchical Density-Based Spatial Clustering)

## Overview
Unsupervised clustering on the **Palmer Penguins** dataset (344 rows → 342 after dropping 2 fully-null rows) to explore whether HDBSCAN can recover natural species groupings from physical measurements alone, and to contrast its behavior against DBSCAN on data with varying cluster density.

## Dataset
Source: `seaborn.load_dataset('penguins')` (mirrors `mwaskom/seaborn-data` GitHub repo)

| Column | Description |
|---|---|
| species | Penguin species: Adelie, Chinstrap, Gentoo (target, excluded from clustering) |
| island | Island of observation: Torgersen, Biscoe, Dream (excluded from clustering) |
| bill_length_mm | Bill length in millimeters |
| bill_depth_mm | Bill depth in millimeters |
| flipper_length_mm | Flipper length in millimeters |
| body_mass_g | Body mass in grams |
| sex | Male/Female, 11 missing values (excluded from clustering) |

## Preprocessing
- Dropped 2 rows with fully-null numeric measurements (~0.6%), no imputation needed
- Outlier check (IQR) on all 4 numeric features → 0 outliers found (clean physical measurement data)
- StandardScaler applied (fit only on the full dataset — no train/test split, this is exploratory/unsupervised, same approach as the PCA notebook)

## Model
- `sklearn.cluster.HDBSCAN` (scikit-learn ≥1.3 built-in implementation)
- Final parameters: `min_cluster_size=10`, `min_samples=5`

### Why HDBSCAN over DBSCAN here
DBSCAN requires a single global `eps` (radius) that must fit the whole dataset. When clusters have different densities, one `eps` is rarely fair to all of them. HDBSCAN instead builds a hierarchy across many density thresholds and automatically selects the most **stable** clusters, so no `eps` needs to be chosen manually.

## Results

**Clusters found: 2** (not 3, despite 3 species in the data)
Cluster -1 (noise): 2 points
Cluster 0: 218 points → Adelie (151) + Chinstrap (67)
Cluster 1: 122 points → Gentoo (122, nearly pure)

| Metric | Score |
|---|---|
| Silhouette Score | 0.5341 |
| Calinski-Harabasz Score | 489.11 |
| Davies-Bouldin Score | 0.7076 |

(All metrics computed excluding noise points, as these scores are undefined for unassigned points.)

## Key Finding
HDBSCAN did not fail to find 3 species — it correctly identified that **Adelie and Chinstrap occupy the same density region** in the 4 measurement dimensions, while **Gentoo forms a clearly separate, denser region** . This was confirmed by cross-tabulating cluster labels against true species and by 2D PCA visualization (explained variance: PC1 68.8% + PC2 19.3% = 88.2%), where Adelie/Chinstrap visibly overlap in PCA space while Gentoo sits in a fully separate region.

This is a genuine illustration of HDBSCAN's varying-density strength: a single-`eps` DBSCAN would have struggled to simultaneously separate the tight Gentoo cluster and the more diffuse, overlapping Adelie/Chinstrap region without either merging everything or fragmenting into noise.

## Limitations
- Small dataset (342 rows) — results may vary considerably for extreme min_cluster_size values.
- No train/test split — this is an exploratory clustering task, not a predictive pipeline, so generalization in the supervised sense doesn't apply