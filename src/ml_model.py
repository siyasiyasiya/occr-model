import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# --- Central Configuration ---
# All key parameters are at the top for easy changes.
INPUT_FILE = '../output/complete_features.csv'
OUTPUT_DIR = Path('../output/')
MANUAL_K = 8  # <-- SET YOUR DESIRED NUMBER OF CLUSTERS HERE (e.g., 4 or 5)
# ---

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Calculates SSE and Silhouette Scores for a range of k values.
    """
    sse = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print("   -> Calculating scores for k=2", end="")
    for k in k_range:
        print(f", {k}", end="", flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    print("... Done.")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return k_range, sse, silhouette_scores, optimal_k

def main():
    print("🎯 COMPREHENSIVE ON-CHAIN RISK MODELING PIPELINE")
    print("=" * 60)

    # --- 1. Load Data ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Input file not found at '{INPUT_FILE}'")
        print("   Please ensure the feature engineering script has run and the path is correct.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {df.shape[0]} accounts with {df.shape[1]} features")

    # --- 2. Feature Selection ---
    feature_cols = [
        'account_age_days', 'xrp_balance', 'balance_to_initial_ratio', 'total_portfolio_value', 'initial_balance',
        'total_transaction_count', 'transaction_success_rate', 'days_since_last_transaction', 'transaction_frequency_per_day',
        'recent_activity_ratio', 'failed_transaction_count', 'payment_transaction_ratio', 'recent_failure_count',
        'net_xrp_flow', 'counterparty_diversity_ratio', 'large_transaction_count', 'total_outgoing_xrp',
        'total_incoming_xrp', 'avg_outgoing_amount', 'total_asset_value', 'asset_weighted_avg_token_score',
        'high_risk_asset_ratio', 'verified_assets_ratio', 'total_assets_held', 'asset_avg_token_score',
        'portfolio_concentration_index', 'token_quality_diversification', 'xrp_portfolio_ratio', 'dormancy_score',
        'liquidity_risk_score', 'activity_consistency', 'operational_risk_score'
    ]
    available_features = [col for col in feature_cols if col in df.columns]
    print(f" Using {len(available_features)} curated features for modeling")
    X = df[available_features]

    # --- 3. Preprocessing ---
    print("\n Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # --- 4. Optimal Cluster Diagnostics ---
    print("\n Finding optimal clusters for diagnostics...")
    k_range, sse, silhouette_scores, auto_optimal_k = find_optimal_clusters(X_scaled)
    print(f"   -> Suggestion from Silhouette Score: k={auto_optimal_k} (score: {max(silhouette_scores):.3f})")
    
    # Plot diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(k_range, sse, marker='o', color='blue'); ax1.set_title('Elbow Method'); ax1.set_xlabel('Number of Clusters (k)'); ax1.set_ylabel('Sum of Squared Errors'); ax1.grid(True)
    ax2.plot(k_range, silhouette_scores, marker='s', color='red'); ax2.axvline(x=auto_optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Auto-Optimal k={auto_optimal_k}'); ax2.set_title('Silhouette Scores'); ax2.set_xlabel('Number of Clusters (k)'); ax2.set_ylabel('Silhouette Score'); ax2.grid(True); ax2.legend()
    plt.tight_layout()
    
    plots_dir = OUTPUT_DIR / "plots"; plots_dir.mkdir(exist_ok=True, parents=True)
    plot_path = plots_dir / "cluster_selection_diagnostics.png"
    plt.savefig(plot_path, dpi=300)
    # plt.show() # Commented out to prevent script pausing
    print(f"   -> Diagnostic plot saved to '{plot_path}'")
    
    # --- 5. Train Final K-Means Model ---
    final_k = MANUAL_K
    print(f"\n Training final K-Means model with interpretable k={final_k}...")
    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df['risk_cluster'] = kmeans_final.fit_predict(X_scaled)

    # --- 6. K-Means Cluster Analysis & Labeling ---
    print(" Analyzing K-Means cluster characteristics...")
    cluster_counts = df['risk_cluster'].value_counts().sort_index()

    df['calculated_risk_score'] = (
        df['dormancy_score'] * 0.25 + df['operational_risk_score'] * 0.25 +
        df['liquidity_risk_score'] * 0.20 + df['high_risk_asset_ratio'] * 0.15 +
        (1 - df['activity_consistency']) * 0.15
    ) * 100
    cluster_risk_scores = df.groupby('risk_cluster')['calculated_risk_score'].mean()
    
    sorted_clusters = cluster_risk_scores.sort_values().index
    risk_levels = [
        'Minimal Risk',
        'Very Low Risk', 
        'Low Risk',
        'Low-Medium Risk',
        'Medium Risk',
        'Medium-High Risk',
        'High Risk',
        'Very High Risk'
    ]
    cluster_risk_mapping = {cluster_id: risk_levels[i] for i, cluster_id in enumerate(sorted_clusters)}
    df['risk_label'] = df['risk_cluster'].map(cluster_risk_mapping)
    
    # --- 7. Train Complementary Isolation Forest Model ---
    print("\n Training complementary Isolation Forest model for direct risk scores...")
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(X_scaled)

    df['anomaly_score'] = iso_forest.score_samples(X_scaled) * -1
    df['risk_score_iso_forest'] = ((df['anomaly_score'] - df['anomaly_score'].min()) / (df['anomaly_score'].max() - df['anomaly_score'].min()) * 100)

    # --- 8. Final Results and Outputs ---
    print(f"\n FINAL RISK PROFILE SUMMARY (k={final_k}):")
    print("=" * 60)
    for cluster_id in sorted_clusters:
        count = cluster_counts[cluster_id]
        label = cluster_risk_mapping[cluster_id]
        score = cluster_risk_scores[cluster_id]
        print(f"Cluster {cluster_id}: {label:<15} | {count:>3} accounts | Avg. Risk Score: {score:>5.1f}")
    
    # Save Outputs
    results_dir = OUTPUT_DIR / "results"; results_dir.mkdir(exist_ok=True, parents=True)
    
    cluster_summary = pd.DataFrame({
        'cluster_id': cluster_risk_scores.index,
        'risk_score': cluster_risk_scores.values,
        'risk_label': [cluster_risk_mapping.get(cid) for cid in cluster_risk_scores.index],
        'account_count': cluster_counts.values
    }).sort_values('risk_score').reset_index(drop=True)

    cluster_summary.to_csv(results_dir / 'cluster_summary.csv', index=False)
    
    # Sort final output by the more granular Isolation Forest score
    df.sort_values('risk_score_iso_forest', ascending=False, inplace=True)
    final_csv_path = results_dir / 'risk_profiles.csv'
    df.to_csv(final_csv_path, index=False)
    
    print("\n ANALYSIS COMPLETE!")
    print(f"   -> Summary saved to '{results_dir / 'cluster_summary.csv'}'")
    print(f"   -> Full dataset with profiles & scores saved to '{final_csv_path}'")

if __name__ == "__main__":
    main()