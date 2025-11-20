print("\n🤖 Training anomaly detection models...")

# Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)
iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.score_samples(X_scaled)
df['iso_forest_anomaly'] = (iso_predictions == -1).astype(int)
df['iso_forest_score'] = -iso_scores  # Convert to anomaly score (higher = more anomalous)

# Local Outlier Factor
lof = LocalOutlierFactor(
    contamination=0.05,
    novelty=False,
    n_neighbors=20
)
lof_predictions = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_
df['lof_anomaly'] = (lof_predictions == -1).astype(int)
df['lof_score'] = lof_scores

# Ensemble: Both models agree
df['ensemble_anomaly'] = ((df['iso_forest_anomaly'] == 1) & (df['lof_anomaly'] == 1)).astype(int)
df['ensemble_score'] = (df['iso_forest_score'] + df['lof_score']) / 2

print("✅ Models trained successfully")