# Customer Segmentation with K-Means Clustering

Unsupervised machine learning project applying K-Means clustering to segment mall customers based on their annual income and spending behavior.

## Dataset
Mall Customer Segmentation dataset (200 customers, 5 features: CustomerID, Gender, Age, Annual Income, Spending Score).

## Workflow
1. **EDA** – Distribution analysis, correlation heatmap, pairplot
2. **Preprocessing** – Feature scaling with StandardScaler
3. **Optimal K Selection** – Elbow Method + Silhouette Score analysis
4. **Model Training** – K-Means (k-means++ initialization)
5. **Evaluation** – Silhouette, Calinski-Harabasz, and Davies-Bouldin scores
6. **Cluster Profiling** – Business interpretation of each segment

## Results
Identified **5 distinct customer segments** with a Silhouette Score of 0.55:

| Segment | Profile |
|---|---|
| VIP Customers | High income, high spending |
| Potential Opportunity | High income, low spending |
| High-Spending Customers | Low income, high spending |
| Budget-Conscious | Low income, low spending |
| Standard Customers | Average income, average spending |

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Key Takeaway
Segmentation reveals actionable marketing strategies per customer group — e.g., loyalty programs for VIPs, targeted campaigns to convert the "Potential Opportunity" segment.