def generate_explainability_card(transaction_idx, df, df_encoded, shap_values, feature_names, top_n=5):
    """
    Generate a detailed explainability card for a specific transaction
    """
    row = df.iloc[transaction_idx]
    shap_row = shap_values[transaction_idx]

    # Get top contributing features
    feature_importance = list(zip(feature_names, shap_row))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_importance[:top_n]

    card = {
        'transaction_id': row['transaction_id'],
        'timestamp': row['timestamp'],
        'amount': row['amount'],
        'is_anomaly': row['ensemble_anomaly'],
        'anomaly_score': row['ensemble_score'],
        'iso_forest_score': row['iso_forest_score'],
        'lof_score': row['lof_score'],
        'top_suspicious_factors': []
    }

    # Generate human-readable explanations
    for feat, shap_val in top_features:
        explanation = {
            'feature': feat,
            'shap_value': shap_val,
            'contribution': 'Increases' if shap_val > 0 else 'Decreases',
        }

        # Add context-specific explanations
        if 'amount' in feat:
            explanation['reason'] = f"Transaction amount ${row['amount']:.2f} is {'unusually high' if row['amount'] > df['amount'].quantile(0.95) else 'unusual'}"
        elif 'hour' in feat:
            explanation['reason'] = f"Transaction at {int(row['hour'])}:00 is {'outside normal business hours' if row['is_late_night'] else 'at an unusual time'}"
        elif 'distance' in feat:
            explanation['reason'] = f"Transaction {row['distance_from_home']:.1f}km from home is {'very far' if row['distance_from_home'] > 100 else 'unusual'}"
        elif 'international' in feat:
            explanation['reason'] = "International transaction" if row['is_international'] else "Domestic transaction"
        elif 'days_since' in feat:
            explanation['reason'] = f"Only {row['days_since_last_transaction']:.2f} days since last transaction" if row['days_since_last_transaction'] < 0.1 else "Normal transaction frequency"
        else:
            explanation['reason'] = f"{feat} has unusual pattern"

        card['top_suspicious_factors'].append(explanation)

    return card

# Generate cards for top anomalies
top_anomalies_idx = df.nlargest(10, 'ensemble_score').index
explainability_cards = []


print("EXPLAINABILITY CARDS - TOP 10 SUSPICIOUS TRANSACTIONS")


for idx in top_anomalies_idx:
    card = generate_explainability_card(idx, df, df_encoded, shap_values, feature_names)
    explainability_cards.append(card)

    print(f"\n{'─'*80}")
    print(f"Transaction ID: {card['transaction_id']}")
    print(f"   Timestamp: {card['timestamp']}")
    print(f"   Amount: ${card['amount']:.2f}")
    print(f"   Anomaly Score: {card['anomaly_score']:.4f} (Isolation: {card['iso_forest_score']:.4f}, LOF: {card['lof_score']:.4f})")
    print(f"   Status: {'ANOMALOUS' if card['is_anomaly'] else 'Normal'}")
    print(f"\n   Top Suspicious Factors:")
    for i, factor in enumerate(card['top_suspicious_factors'], 1):
        print(f"   {i}. {factor['feature']}: {factor['reason']}")
        print(f"      → SHAP Impact: {factor['shap_value']:.4f} ({factor['contribution']} suspicion)")