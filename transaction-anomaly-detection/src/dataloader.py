def generate_transaction_data(n_samples=3000, anomaly_ratio=0.05):
    """
    Generate realistic synthetic transaction data with built-in anomalies
    """
    np.random.seed(42)
    random.seed(42)

    # Normal transactions
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomalies = n_samples - n_normal

    # Generate base features for normal transactions
    hour_probs = [0.03, 0.03, 0.08, 0.1, 0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.06, 0.03, 0.02, 0.01]
    normal_data = {
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(n_normal)],
        'amount': np.random.lognormal(4, 1.5, n_normal),  # Mean ~$100
        'hour': np.random.choice(range(8, 22), n_normal, p=hour_probs),  # Business hours
        'day_of_week': np.random.choice(range(7), n_normal),
        'merchant_category': np.random.choice(['Grocery', 'Restaurant', 'Gas', 'Retail', 'Entertainment'],
                                             n_normal, p=[0.25, 0.25, 0.15, 0.25, 0.1]),
        'transaction_type': np.random.choice(['Debit', 'Credit'], n_normal, p=[0.6, 0.4]),
        'distance_from_home': np.random.gamma(2, 5, n_normal),  # km
        'is_international': np.random.choice([0, 1], n_normal, p=[0.95, 0.05]),
        'days_since_last_transaction': np.random.exponential(1, n_normal),
    }

    # Generate anomalous transactions
    anomaly_data = {
        'transaction_id': [f'TXN{str(i+n_normal).zfill(6)}' for i in range(n_anomalies)],
        'amount': np.concatenate([
            np.random.lognormal(8, 1, n_anomalies//3),  # Very high amounts
            np.random.lognormal(1, 0.5, n_anomalies//3),  # Very low amounts
            np.random.lognormal(4, 2, n_anomalies - 2*(n_anomalies//3))
        ]),
        'hour': np.random.choice(range(24), n_anomalies),  # Any hour including late night
        'day_of_week': np.random.choice(range(7), n_anomalies),
        'merchant_category': np.random.choice(['Grocery', 'Restaurant', 'Gas', 'Retail', 'Entertainment', 'Unknown'],
                                             n_anomalies, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
        'transaction_type': np.random.choice(['Debit', 'Credit', 'Wire'], n_anomalies, p=[0.3, 0.3, 0.4]),
        'distance_from_home': np.concatenate([
            np.random.uniform(200, 2000, n_anomalies//2),  # Very far transactions
            np.random.gamma(2, 5, n_anomalies - n_anomalies//2)
        ]),
        'is_international': np.random.choice([0, 1], n_anomalies, p=[0.3, 0.7]),
        'days_since_last_transaction': np.concatenate([
            np.random.uniform(0, 0.01, n_anomalies//2),  # Rapid succession
            np.random.exponential(1, n_anomalies - n_anomalies//2)
        ]),
    }

    # Combine normal and anomalous data
    df_normal = pd.DataFrame(normal_data)
    df_anomaly = pd.DataFrame(anomaly_data)
    df = pd.concat([df_normal, df_anomaly], ignore_index=True)

    # Add derived features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_late_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    # Create timestamp
    start_date = datetime(2024, 1, 1)
    df['timestamp'] = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(len(df))]

    # True labels (for evaluation)
    df['true_anomaly'] = [0]*n_normal + [1]*n_anomalies

    return df

# Generate data
print("🔄 Generating synthetic transaction data...")
df = generate_transaction_data(n_samples=3000, anomaly_ratio=0.05)
print(f"✅ Generated {len(df)} transactions ({df['true_anomaly'].sum()} true anomalies)")
print("\n📊 Dataset Preview:")
print(df.head(10))