def prepare_features(df):
    """Prepare features for anomaly detection"""
    feature_cols = [
        'amount_log', 'hour', 'day_of_week', 'distance_from_home',
        'is_international', 'days_since_last_transaction', 'is_weekend',
        'is_late_night', 'high_amount'
    ]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['merchant_category', 'transaction_type'], prefix=['merchant', 'type'])

    # Get all feature columns
    all_features = feature_cols + [col for col in df_encoded.columns if col.startswith(('merchant_', 'type_'))]

    X = df_encoded[all_features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, all_features, scaler, df_encoded

print("\n🔧 Preparing features...")
X_scaled, feature_names, scaler, df_encoded = prepare_features(df)
print(f"✅ Prepared {X_scaled.shape[1]} features")