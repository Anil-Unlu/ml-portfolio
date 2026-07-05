
Overview
This model predicts whether a bank client will subscribe to a term deposit based on demographic, financial, and campaign-related features, using a Support Vector Machine classifier.
Dataset
Bank Marketing (Term Deposit) — UCI Machine Learning Repository (bank-additional-full.csv, 41,188 rows originally, 40,787 after cleaning)
The data comes from a Portuguese bank's telemarketing campaign. The goal is to predict whether a client will say "yes" to a term deposit offer, so the bank can prioritize who to call.
Target: y — client subscribed to a term deposit (yes / no)
Column descriptions:
ColumnDescriptionageClient's agejobType of job (admin, blue-collar, technician, etc.)maritalMarital statuseducationEducation leveldefaultHas credit in default (simplified to no/unknown — "yes" was only 3 rows)housingHas housing loanloanHas personal loancontactContact communication type (cellular/telephone)monthLast contact monthday_of_weekLast contact day of weekcampaignNumber of contacts performed during this campaign for this clientpdaysDays since last contact from a previous campaign (-1 = never contacted, sentinel value fixed from original 999)previousNumber of contacts performed before this campaignpoutcomeOutcome of the previous marketing campaigneuribor3mEuribor 3-month rate (the only macroeconomic indicator kept after VIF analysis)was_previously_contactedBinary flag derived from pdays (1 = contacted before, 0 = never contacted)
Note: The original dataset also included duration (last call duration) and four additional macroeconomic indicators (emp.var.rate, cons.price.idx, cons.conf.idx, nr.employed). Both were removed — see Preprocessing.
How SVM Works
SVM finds the hyperplane that best separates two classes by maximizing the margin — the distance between the hyperplane and the closest points of each class (called support vectors). Only these support vectors determine the boundary; the rest of the data points don't affect it.
When classes aren't linearly separable, SVM uses the kernel trick to implicitly project data into a higher-dimensional space where a linear separation becomes possible — without ever computing that transformation explicitly. This project uses the RBF (Gaussian) kernel, which can produce flexible, non-linear, even disconnected decision regions.
Two hyperparameters control this behavior:

C: the trade-off between a wide margin (more tolerance for misclassified points) and a strict margin (fewer misclassifications, but risk of overfitting)
gamma: how far the influence of a single training point reaches (high gamma = tightly-fit, wiggly boundary; low gamma = smoother, more generalized boundary)

Because the decision boundary depends on distances between points, feature scaling is mandatory for SVM — unscaled features would distort the margin calculation.
When to Use SVM

Medium-sized datasets (SVM's training cost grows quadratically/cubically with sample size, making it impractical for very large datasets)
Problems where a clear margin of separation exists between classes
High-dimensional feature spaces (e.g., text classification)
Not ideal for: very large datasets, or datasets where classes overlap heavily in feature space

Preprocessing

Dropped rows with rare "unknown" categories in job and marital (low frequency, safe to drop)
Simplified default from 3 categories to 2 (yes had only 3 rows, merged into unknown)
Merged rare education category illiterate (18 rows) into basic.4y (closest hierarchical neighbor) to avoid unstable dummy variables
Dropped duration: this column causes data leakage — call duration is only known after the call ends, so it can't be used in a realistic pre-call prediction model
Fixed pdays sentinel value: the original 999 ("never contacted") was replaced with -1 instead of 0, after discovering a collision — 15 clients had a genuine pdays=0 (contacted same day), which would have been indistinguishable from the sentinel if 0 had been reused
Outlier capping (IQR) on age and campaign
Multicollinearity (VIF): the 5 macroeconomic indicators were severely collinear (VIF up to ~26,000). Kept only euribor3m (most interpretable, most correlated with the others), dropped the remaining four
One-hot encoding (10 categorical columns, drop_first=True to avoid the dummy variable trap)
Train/test split (80/20, stratified)
StandardScaler (fit on train only)
SMOTE (applied on training data only, after scaling)

Hyperparameter Tuning
Due to SVM's computational cost on ~58,000 rows (post-SMOTE), GridSearchCV was run on a 5,000-row subsample first, then the best parameters were validated on the full training set.

First attempt (C=10, gamma=0.1): CV ROC-AUC of 0.99 on the subsample, but this overfit badly — test performance dropped to Class 1 F1 = 0.27, worse than the untuned baseline. This is a case of the small subsample being "memorized" by an overly flexible boundary (high C + high gamma).
Second attempt (C=5, gamma='scale'): more conservative parameters, validated on the full test set with much better generalization.

Results (Test Set — Final Model: C=5, gamma='scale', kernel='rbf')
MetricBaseline (default)Tuned (final)Class 1 Precision0.410.45Class 1 Recall0.510.40Class 1 F10.460.42Accuracy0.860.88Test ROC-AUC—0.74
After threshold tuning (default 0.0 → -0.4552 on decision_function):

Precision: 0.378
Recall: 0.535
F1: 0.443 (improved from 0.42)

Cross-validation: CV ROC-AUC on a SMOTE-balanced subsample was 0.97 ± 0.02 — considerably higher than the real Test ROC-AUC of 0.74. This gap is expected and explained below.
Limitations

CV vs. Test ROC-AUC gap (0.97 vs 0.74): CV was computed on SMOTE-balanced data, where the two classes are equally represented, making separation easier. The real test set reflects the true imbalance (~11% positive class). This is a methodological artifact of measuring CV on resampled data, not a sign of instability — the real-world performance indicator is the test set result, not the CV score.
Modest overall performance: compared to tree-based ensemble models in this portfolio (e.g., Random Forest's 0.924 CV ROC-AUC on Telco Churn), SVM performed less well here. This dataset is high-dimensional (45 features, mostly one-hot encoded) and imbalanced — conditions where SVM's distance-based approach is less advantageous than ensemble methods.
Hyperparameter tuning on a subsample carries risk: as seen in the first tuning attempt, parameters that look optimal on a small subsample (via CV) can overfit badly when applied to the full dataset. Always validate on the full test set before trusting subsample-based tuning results.
Computational cost: SVM's training time scales poorly with sample size (quadratic/cubic), making full-dataset GridSearchCV impractical here. Tuning was done on a 5,000-row subsample as a practical trade-off.