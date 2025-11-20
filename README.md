# Transaction Anomaly Detection Using Isolation Forest, LOF & SHAP Explainability

A complete end-to-end anomaly detection workflow for identifying suspicious financial transactions using:
Isolation Forest
Local Outlier Factor (LOF)
Ensemble model
SHAP explainability to understand why a transaction is anomalous

This project includes synthetic transaction generation, feature engineering, modeling, evaluation, explainability cards, and interactive visualizations using Plotly.

------------------------------------------------------------------------------------
Key Features

 Dual Detection Models: Isolation Forest + Local Outlier Factor with ensemble approach
 SHAP Explainability: Understand exactly why transactions are flagged
 Explainability Cards: Human-readable reports for each anomaly
 Interactive Dashboard: 6+ visualizations including ROC curves, confusion matrices, and timeline views
 Export Ready: CSV and JSON outputs for downstream systems
 Production-Ready: Handles 1,000-5,000+ transactions with real-world features
 
-------------------------------------------------------------------------------------
 How It Works
1. Data Generation

Generates 3,000 synthetic transactions with realistic patterns
5% built-in anomalies (150 transactions)
Features: amount, time, location, merchant category, transaction type, etc.

2. Feature Engineering

Creates derived features (log amounts, time-based flags, distance metrics)
One-hot encoding for categorical variables
StandardScaler normalization

3. Anomaly Detection

Isolation Forest: Identifies anomalies by isolating outliers
Local Outlier Factor: Detects local density deviations
Ensemble: Combines both models for higher precision

4. Explainability

SHAP Values: Calculate feature contributions for each prediction
Explainability Cards: Generate human-readable explanations

5. Visualization & Export

Interactive Plotly dashboards
Confusion matrices and ROC curves
Export to CSV and JSON

----------------------------------------------------------------------------------

Visualizations
The system generates 6 interactive visualizations:
1. Anomaly Score Distributions

Histogram comparison of normal vs anomalous transactions
Separate plots for Isolation Forest and LOF scores

2. Confusion Matrices

Performance comparison across all three models
Visual heatmaps with counts

3. Feature Importance (SHAP)

Top 15 most important features
Mean absolute SHAP values

4. Transaction Timeline

Scatter plot of all transactions over time
Anomalies highlighted as red stars
Log-scale y-axis for amount

5. SHAP Summary Plot

Detailed feature impact visualization
Shows positive/negative contributions

6. ROC Curves

Model comparison with AUC scores
Helps evaluate trade-offs
