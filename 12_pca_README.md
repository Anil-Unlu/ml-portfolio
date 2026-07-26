# 12 - PCA: Wine Dataset Dimensionality Reduction

## Dataset
Wine dataset (sklearn built-in), 178 rows, 13 chemical features, 3 wine classes (used only for visualization, not for fitting).

## Method
- StandardScaler (PCA requires scaling — variance-based method, unscaled features with larger ranges would dominate)
- No train/test split — used as an exploratory/visualization tool, not a predictive pipeline
- PCA fit on all 13 features first to inspect explained variance across components

## Explained Variance
| Components | Cumulative Variance |
|---|---|
| PC1 | 36.2% |
| PC1+PC2 | 55.4% |
| PC1-PC5 | 80.2% |

## 2D Visualization
Projected onto PC1+PC2 (55.4% variance retained). The 3 wine classes separate into clearly distinct clusters despite PCA never seeing the class labels during fitting — target used only for coloring points post-hoc.

## Component Interpretation (Loadings)
- **PC1** — dominated by flavanoids, total_phenols, od280/od315 → phenolic/antioxidant profile
- **PC2** — dominated by color_intensity, alcohol, proline → body/density profile

## Key Takeaway
PCA is unsupervised: fitting uses only X, never y. The clean class separation seen here means the directions of highest variance (found blindly) happen to align with the real chemical differences between wine classes — not guaranteed in general, but a good sign the features are informative.

## Limitations
- Only 55.4% of original variance retained in 2D — some information loss, PC5 needed for 80%
- PCA component interpretations should be considered approximate, as each component represents a weighted combination of multiple features rather than a directly interpretable single underlying factor.