from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

def evaluate_model(y_true, y_pred, scores, model_name):
    """Evaluate anomaly detection model"""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, scores):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    return cm

# Evaluate models
cm_iso = evaluate_model(df['true_anomaly'], df['iso_forest_anomaly'],
                        df['iso_forest_score'], "Isolation Forest")
cm_lof = evaluate_model(df['true_anomaly'], df['lof_anomaly'],
                       df['lof_score'], "Local Outlier Factor")
cm_ensemble = evaluate_model(df['true_anomaly'], df['ensemble_anomaly'],
                             df['ensemble_score'], "Ensemble")