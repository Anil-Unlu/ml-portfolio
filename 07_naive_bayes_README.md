# 07 - Naive Bayes: SMS Spam Classification

## Overview
This notebook applies **Multinomial Naive Bayes** to classify SMS messages as spam or ham (legitimate), using the UCI SMS Spam Collection dataset. Unlike the previous notebooks, this project uses **text data**, introducing NLP preprocessing (tokenization, stopword removal, lemmatization) and **TF-IDF vectorization** as new techniques in the portfolio.

## Dataset
**SMS Spam Collection** (UCI Machine Learning Repository)
- 5,572 SMS messages, labeled as `ham` (legitimate) or `spam`
- After removing 403 duplicate messages: 5,169 unique messages
- Class distribution: ~87.4% ham, ~12.6% spam (imbalanced)

### Columns
| Column | Description |
|---|---|
| `label` | Target variable: `ham` (legitimate message) or `spam` (unwanted/promotional message), encoded as 0/1 |
| `message` | Raw SMS text content |

## Math Intuition
Naive Bayes is based on **Bayes' Theorem**:

P(class | words) = P(words | class) × P(class) / P(words)

The "naive" part comes from assuming all words are **conditionally independent** given the class — i.e., the presence of one word doesn't affect the probability of another word appearing, given we already know the message is spam or ham. This assumption is technically false (words are correlated in real language), but it simplifies the computation dramatically and works remarkably well in practice for text classification.

For classification, the model picks the class that maximizes:
**Multinomial Naive Bayes** specifically models `P(word | class)` as a multinomial distribution — appropriate for frequency/count-based features (like TF-IDF weights), as opposed to `GaussianNB` (continuous numeric features assumed normally distributed) or `BernoulliNB` (binary presence/absence features).

## When to Use
- Text classification (spam detection, sentiment analysis, topic labeling)
- High-dimensional, sparse feature spaces (e.g., thousands of TF-IDF features)
- When training speed and simplicity matter — NB trains in a single pass over the data, no iterative optimization
- Works well even with relatively small datasets, since it estimates simple per-class word probabilities rather than complex decision boundaries
- Not ideal when features are strongly correlated (violates the independence assumption more severely) or when precise probability calibration matters (NB's predicted probabilities tend to be overconfident/poorly calibrated)

## Preprocessing Pipeline
1. **Duplicate removal**: 403 duplicate messages dropped (26.5% of duplicates were spam vs 12.6% baseline — spam messages are disproportionately templated/repeated). Prevents data leakage from identical messages appearing in both train and test.
2. **Train/test split** (80/20, stratified) — done *before* any text cleaning or vectorization to prevent leakage.
3. **Text cleaning**: lowercase, remove non-alphabetic characters, tokenize, remove stopwords, lemmatize (via NLTK).
4. **TF-IDF vectorization** (`max_features=3000`, `min_df=2`) — fit only on train, transformed on test. Resulted in a 2,789-word vocabulary.
5. **SMOTE** applied after vectorization, on train set only, to balance the minority (spam) class from 522 → 3,613 samples.

## Model & Results
- **Model**: `MultinomialNB` (default `alpha=1.0` smoothing)
- **Baseline (threshold=0.5)**: ROC-AUC 0.989, Spam F1 0.837, Spam Recall 0.94, Spam Precision 0.75
- **Threshold tuning** (best threshold ≈ 0.76, selected by maximizing F1 on precision-recall curve):
  - Spam Precision: 0.75 → **0.91**
  - Spam Recall: 0.94 → 0.92
  - Spam F1: 0.837 → **0.913**
  - False positives (legitimate messages misclassified as spam) reduced from 40 → 12 — prioritized here since misclassifying a real message as spam is typically more costly than missing a spam message.
- **Cross-validation** (5-fold, full pipeline with TF-IDF + SMOTE refit per fold, default threshold): CV ROC-AUC 0.984 ± 0.0035, CV F1 0.847 ± 0.0136 — very low variance across folds, indicating stable performance regardless of data split.
- **Overfit check**: Train F1 0.983 vs Test F1 0.913 (gap 0.071), Train AUC 0.999 vs Test AUC 0.989 (gap 0.010) — mild, acceptable gap. Naive Bayes' simplicity (few learned parameters relative to feature count) makes it inherently resistant to overfitting.

## Feature Importance (log-probability difference)
Since NB has no built-in feature_importances_, importance was derived from the difference in log P(word | class) between spam and ham:

**Top spam-indicative words**: `prize`, `claim`, `tone`, `www`, `uk`, `service`, `guaranteed`, `ppm`, `ringtone`, `awarded`, `urgent`, `nokia` — classic promotional/scam vocabulary (dataset reflects early-2000s ringtone/prize SMS scams).

**Top ham-indicative words**: `lol`, `home`, `later`, `sure`, `anything`, and Singlish-influenced terms (`lor`, `lt`, `gt`, `wat`, `da`) — reflecting the dataset's Singaporean origin.

This clear vocabulary separation explains the model's very high AUC.

## Limitations
- **Independence assumption violated**: words in real sentences are correlated (e.g., "free" and "prize" co-occur), but NB ignores this — works well here regardless, likely because spam/ham vocabularies are highly distinct.
- **Stopword removal can strip meaningful negations** (e.g., "no need to change" → "need change" after removing "no"), which can flip perceived sentiment in edge cases. Not fully explored/fixed in this notebook — kept as a known limitation of the bag-of-words approach.
- **NB probabilities are known to be poorly calibrated** (tend toward extreme values near 0 or 1); predicted probabilities were used for threshold tuning but shouldn't be interpreted as true confidence levels without calibration.
- **CV vs single-split threshold mismatch**: CV F1 (0.847) used default 0.5 threshold (not tuned per fold), while the single train/test split result (0.913 F1) used the tuned 0.76 threshold — not directly comparable, both reported for transparency.
- Dataset is relatively small (5,169 messages) and somewhat dated (references to ringtones, premium-rate numbers) — vocabulary patterns may not generalize to modern spam (e.g., phishing links, crypto scams).