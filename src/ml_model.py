import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib

# --- Central Configuration ---
INPUT_FILE = '../output/complete_features.csv'
OUTPUT_DIR = Path('../output/')
MANUAL_K = 5
# ---

def main():
    print("COMPREHENSIVE ON-CHAIN RISK MODELING PIPELINE (V7 - GMM)")
    print("=" * 60)

    # --- 1. Load Data ---
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at '{INPUT_FILE}'")
        return
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {df.shape[0]} accounts with {df.shape[1]} features")

    # --- 2. Principled Feature Selection ---
    print("\n Performing principled feature selection...")
    all_feature_cols = [
        'account_age_days', 'xrp_balance', 'balance_to_initial_ratio', 'total_portfolio_value', 'initial_balance',
        'total_transaction_count', 'transaction_success_rate', 'days_since_last_transaction', 'transaction_frequency_per_day',
        'recent_activity_ratio', 'failed_transaction_count', 'payment_transaction_ratio', 'recent_failure_count',
        'counterparty_diversity_ratio', 'avg_outgoing_amount', 'total_asset_value', 'asset_weighted_avg_token_score',
        'high_risk_asset_ratio', 'verified_assets_ratio', 'total_assets_held', 'asset_avg_token_score',
        'portfolio_concentration_index', 'token_quality_diversification', 'xrp_portfolio_ratio', 'dormancy_score',
        'liquidity_risk_score', 'activity_consistency', 'operational_risk_score'
    ]
    available_features = [col for col in all_feature_cols if col in df.columns]
    X_all = df[available_features]

    imputer_fs = SimpleImputer(strategy='median')
    X_imputed_fs = imputer_fs.fit_transform(X_all)
    scaler_fs = StandardScaler()
    X_scaled_fs = scaler_fs.fit_transform(X_imputed_fs)
    X_all_scaled = pd.DataFrame(X_scaled_fs, columns=X_all.columns)

    correlations = X_all_scaled.corrwith(X_all_scaled['total_transaction_count']).abs().sort_values(ascending=False)
    correlation_threshold = 0.3
    strong_features = correlations[correlations > correlation_threshold].index.tolist()
    key_heuristic_features = ['dormancy_score', 'operational_risk_score', 'liquidity_risk_score', 'high_risk_asset_ratio', 'activity_consistency']
    for feat in key_heuristic_features:
        if feat not in strong_features and feat in X_all.columns:
            strong_features.append(feat)

    feature_cols = list(set(strong_features))
    print(f"   -> Selected {len(feature_cols)} features with correlation strength > {correlation_threshold}")
    X = df[feature_cols]

    # --- 3. Preprocessing Final Feature Set ---
    print("\n Preprocessing final feature set...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # --- 4. Isolate Anomalous Users & Cluster with GMM ---
    print("\n Identifying anomalous accounts and clustering the core population...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df['is_outlier'] = iso_forest.fit_predict(X_scaled)
    outlier_count = (df['is_outlier'] == -1).sum()
    print(f"   -> Identified {outlier_count} anomalous accounts (hyper-power users).")
    
    df['risk_cluster'] = np.nan
    
    core_indices = df[df['is_outlier'] == 1].index
    X_scaled_core = X_scaled[core_indices]
    
    final_n_components = MANUAL_K
    # covariance_type='full' allows GMM to find flexible, elliptical clusters.
    gmm = GaussianMixture(n_components=final_n_components, covariance_type='full', random_state=42)
    
    print(f"\n Training GMM model with n_components={final_n_components} on {len(core_indices)} core accounts...")
    df.loc[core_indices, 'risk_cluster'] = gmm.fit_predict(X_scaled_core)
    
    # --- Get Cluster Confidence Score ---
    core_probabilities = gmm.predict_proba(X_scaled_core)
    df.loc[core_indices, 'cluster_confidence'] = core_probabilities.max(axis=1)
    df['cluster_confidence'].fillna(1.0, inplace=True) # Outliers have 100% confidence
    print("   -> Generated cluster confidence scores.")
    
    df.loc[df['is_outlier'] == -1, 'risk_cluster'] = -1
    print(f"   -> Clustered {len(core_indices)} core accounts into {final_n_components} groups.")

    # --- 5. Generate Definitive PCA Score for ALL Accounts ---
    print("\n Generating definitive continuous risk score using PCA...")
    # Use the full scaled data for PCA to get a score for every user
    pca_final = PCA(n_components=1)
    full_dataset_pca_transformed = pca_final.fit_transform(X_scaled)
    df['pca_score'] = full_dataset_pca_transformed
    
    # Create a temporary dataframe for the correlation check
    temp_df_for_corr = df[['pca_score']].copy()
    temp_df_for_corr['total_transaction_count'] = X_all['total_transaction_count']
    
    final_corr_check = temp_df_for_corr.corr().iloc[0, 1]
    if final_corr_check > 0:
        print("   -> Flipping PCA score to align with risk definition.")
        df['pca_score'] = -df['pca_score']
    
    pca_min = df['pca_score'].min()
    pca_max = df['pca_score'].max()
    df['risk_score_pca'] = 100 * (df['pca_score'] - pca_min) / (pca_max - pca_min)
    print("   -> Continuous PCA risk score (0-100) has been generated.")

    # --- 6. Final Labeling Based on Definitive PCA Score ---
    print("\n Applying final risk labels based on PCA score...")
    cluster_pca_scores = df.groupby('risk_cluster')['risk_score_pca'].mean()
    core_clusters_sorted = cluster_pca_scores[cluster_pca_scores.index != -1].sort_values().index
    
    risk_levels = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    cluster_risk_mapping = {'-1': 'Minimal Risk'}
    for i, cluster_id in enumerate(core_clusters_sorted):
        cluster_risk_mapping[cluster_id] = risk_levels[i]
            
    df['risk_label'] = df['risk_cluster'].map(cluster_risk_mapping)
    print("   -> Final risk labels assigned successfully.")

    # --- 7. Final Results and Outputs ---
    print(f"\n FINAL RISK PROFILE SUMMARY (n_components={final_n_components} core + 1 safe outlier group):")
    print("=" * 60)
    
    summary_data = df.groupby('risk_label').agg(
        account_count=('risk_label', 'count'),
        avg_pca_score=('risk_score_pca', 'mean')
    ).reset_index()
    
    label_order = ['Minimal Risk', 'Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    summary_data['risk_label'] = pd.Categorical(summary_data['risk_label'], categories=label_order, ordered=True)
    summary_data.sort_values('risk_label', inplace=True)

    for index, row in summary_data.iterrows():
        print(f"{row['risk_label']:<15} | {row['account_count']:>4} accounts | Avg. PCA Score: {row['avg_pca_score']:>5.1f}")
    
    # Save Outputs
    results_dir = OUTPUT_DIR / "results"; results_dir.mkdir(exist_ok=True, parents=True)
    df.sort_values('risk_score_pca', ascending=False, inplace=True)
    final_csv_path = results_dir / 'risk_profiles.csv'
    df.to_csv(final_csv_path, index=False)
    
    print("\n ANALYSIS COMPLETE!")
    print(f"   -> Full dataset with profiles & scores saved to '{final_csv_path}'")

    # --- 8. Save Model Artifacts for Deployment ---
    print("\n Saving model artifacts for future predictions...")
    
    artifacts_dir = OUTPUT_DIR / "artifacts"; artifacts_dir.mkdir(exist_ok=True, parents=True)

    # 1. The Scaler: To process new data with the same scaling as the training data.
    # 2. The Feature List: To ensure the order and number of features is identical.
    # 3. The Isolation Forest: To identify safe outliers.
    # 4. The GMM: To cluster the core accounts.
    # 5. The PCA model: To calculate the final continuous score.
    # 6. The Cluster-to-Label Mapping: To translate cluster IDs into human-readable labels.
    
    artifacts = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'isolation_forest': iso_forest,
        'gmm': gmm,
        'pca': pca_final,
        'cluster_risk_mapping': cluster_risk_mapping
    }
    
    for name, model in artifacts.items():
        joblib.dump(model, artifacts_dir / f'{name}.joblib')
        
    print(f"   -> All artifacts saved to '{artifacts_dir}'")
    

if __name__ == "__main__":
    main()