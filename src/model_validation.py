import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import calinski_harabasz_score # A better geometric metric for GMM
from sklearn.mixture import GaussianMixture # To get BIC/AIC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
INPUT_PROFILES = '../output/results/risk_profiles.csv'
RAW_DATA_FILE = '../output/complete_features.csv'
OUTPUT_DIR = Path('../output/validation/')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
N_COMPONENTS = 5

def validate_gmm_model():
    """Comprehensive validation of the GMM-based risk model with corrected metrics"""
    print("GMM RISK MODEL VALIDATION (V2 - Corrected Metrics)")
    print("=" * 60)
    print(f"Run on: {pd.Timestamp.now()} by user: siyasiyasiya")
    
    # Load the risk profiles
    if not os.path.exists(INPUT_PROFILES):
        print(f"ERROR: Risk profiles file not found at '{INPUT_PROFILES}'")
        return
        
    df = pd.read_csv(INPUT_PROFILES)
    print(f"Loaded {df.shape[0]} risk profiles for validation")
    
    # 1. Risk Category Distribution
    is_monotonic, risk_score_stats = validate_risk_distribution(df)
    plot_risk_score_distribution(df)
    
    # 2. Cluster Confidence Analysis
    avg_confidence, conf_stats = validate_cluster_confidence(df)
    plot_cluster_confidence(df)

    # 3. Feature Correlations
    valid_indicators = validate_feature_correlations(df)
    plot_feature_correlations(df)

    # 4. GMM-Specific Clustering Quality
    bic, aic, ch_score = validate_gmm_quality(df)
    try:
        raw_df = pd.read_csv(RAW_DATA_FILE)
        available_features = [col for col in raw_df.columns if raw_df[col].dtype != 'object']
        X_imputed = SimpleImputer(strategy='median').fit_transform(raw_df[available_features])
        X_scaled = StandardScaler().fit_transform(X_imputed)
        core_indices = df[df['risk_cluster'] != -1].index
        plot_gmm_fit_metrics(X_scaled[core_indices])
    except:
        pass

    # 5. Model Consistency
    score_category_corr = validate_model_consistency(df)
    plot_pca_scatter(df)
    
    # 6. Generate Summary Report
    generate_summary_report(is_monotonic, valid_indicators, score_category_corr, avg_confidence, bic)

def validate_risk_distribution(df):
    """Validate the distribution of accounts across risk categories"""
    print("\n[1/5] Validating Risk Distributions...")
    risk_order = ['Minimal Risk', 'Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    existing_categories = [cat for cat in risk_order if cat in df['risk_label'].unique()]
    
    # Calculate PCA risk score statistics by category
    risk_score_stats = df.groupby('risk_label')['risk_score_pca'].agg(['mean', 'min', 'max', 'std']).round(1)
    risk_score_stats = risk_score_stats.reindex(existing_categories)
    
    print("PCA Risk Score by Category:")
    print(risk_score_stats)
    
    risk_score_means = risk_score_stats['mean']
    is_monotonic = risk_score_means.is_monotonic_increasing
    print(f"Risk scores increase monotonically: {'Yes' if is_monotonic else 'No'}")
    return is_monotonic, risk_score_stats

def validate_cluster_confidence(df):
    """Validate the confidence scores from the GMM model"""
    print("\n[2/5] Validating Cluster Confidence...")
    if 'cluster_confidence' not in df.columns:
        print("Cluster confidence scores not found.")
        return 0, None
    
    conf_stats = df.groupby('risk_label')['cluster_confidence'].agg(['mean', 'min', 'max', 'std']).round(3)
    avg_confidence = df['cluster_confidence'].mean()
    print(f"Overall Average Cluster Confidence: {avg_confidence:.3f}")
    print("\nCluster Confidence by Risk Category:")
    print(conf_stats)
    return avg_confidence, conf_stats

def validate_feature_correlations(df):
    """Validate correlations between risk scores and key features"""
    print("\n[3/5] Validating Feature Correlations...")
    exclude_cols = ['risk_cluster', 'risk_score_pca', 'pca_score', 'is_outlier', 'cluster_confidence']
    feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]
    correlations = df[feature_cols + ['risk_score_pca']].corr()['risk_score_pca'].drop('risk_score_pca')
    
    key_indicators = {'dormancy_score': 'positive', 'transaction_frequency_per_day': 'negative', 'days_since_last_transaction': 'positive', 'high_risk_asset_ratio': 'positive', 'transaction_success_rate': 'negative'}
    
    print("Validation of key risk indicators:")
    valid_indicators = True
    for indicator, expected in key_indicators.items():
        if indicator in correlations.index:
            actual = 'positive' if correlations[indicator] > 0 else 'negative'
            match = actual == expected
            if not match:
                if indicator != 'high_risk_asset_ratio':
                    valid_indicators = False
            print(f"  {indicator:<30}: Expected {expected:<8}, Actual {actual:<8}, Match: {'Yes' if match else 'No'}")
    return valid_indicators

def plot_risk_score_distribution(df):
    risk_order = ['Minimal Risk', 'Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    existing_categories = [cat for cat in risk_order if cat in df['risk_label'].dropna().unique()]
    
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df,
        x='risk_label',
        y='risk_score_pca',
        order=existing_categories
    )
    plt.title("Risk Score Distribution by Risk Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_score_distribution.png", dpi=300)
    plt.close()


def plot_cluster_confidence(df):
    if 'cluster_confidence' in df.columns:
        risk_order = ['Minimal Risk', 'Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        existing_categories = [cat for cat in risk_order if cat in df['risk_label'].dropna().unique()]

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x='risk_label',
            y='cluster_confidence',
            order=existing_categories
        )
        plt.title("Cluster Confidence by Risk Label")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cluster_confidence.png", dpi=300)
        plt.close()


def plot_feature_correlations(df):
    exclude_cols = ['risk_cluster', 'risk_score_pca', 'pca_score', 'is_outlier', 'cluster_confidence']
    feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]
    corr_matrix = df[feature_cols + ['risk_score_pca']].corr()

    # Take top 15 features most correlated with risk_score_pca
    top_features = corr_matrix['risk_score_pca'].abs().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Top Feature Correlations with Risk Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

def plot_gmm_fit_metrics(X_scaled_core, max_components=10):
    """Fit GMM for multiple component counts and plot BIC/AIC curves"""
    bics, aics, n_components_list = [], [], range(2, max_components+1)
    for n in n_components_list:
        gmm = GaussianMixture(n_components=n, covariance_type='full', n_init=5, random_state=42)
        gmm.fit(X_scaled_core)
        bics.append(gmm.bic(X_scaled_core))
        aics.append(gmm.aic(X_scaled_core))
    
    plt.figure(figsize=(8, 5))
    plt.plot(n_components_list, bics, marker='o', label='BIC')
    plt.plot(n_components_list, aics, marker='o', label='AIC')
    plt.xlabel("Number of Components")
    plt.ylabel("Score (Lower is Better)")
    plt.title("GMM Model Selection Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gmm_bic_aic.png", dpi=300)
    plt.close()

def plot_pca_scatter(df):
    # If PCA components already exist, use them, else compute from numeric features
    numeric_df = df.select_dtypes(include=np.number)
    if 'pca1' not in df.columns or 'pca2' not in df.columns:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df.fillna(0))
        df['pca1'], df['pca2'] = pca_result[:, 0], pca_result[:, 1]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='risk_label', palette='tab10', alpha=0.7)
    plt.title("PCA Projection of Accounts by Risk Label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_scatter.png", dpi=300)
    plt.close()

def validate_gmm_quality(df):
    """Calculate GMM-specific clustering metrics like BIC, AIC, and Calinski-Harabasz"""
    print("\n[4/5] Validating GMM Clustering Quality...")
    try:
        raw_df = pd.read_csv(RAW_DATA_FILE)
        all_feature_cols = ['account_age_days', 'xrp_balance', 'balance_to_initial_ratio', 'total_portfolio_value', 'initial_balance','total_transaction_count', 'transaction_success_rate', 'days_since_last_transaction', 'transaction_frequency_per_day','recent_activity_ratio', 'failed_transaction_count', 'payment_transaction_ratio', 'recent_failure_count','counterparty_diversity_ratio', 'avg_outgoing_amount', 'total_asset_value', 'asset_weighted_avg_token_score','high_risk_asset_ratio', 'verified_assets_ratio', 'total_assets_held', 'asset_avg_token_score','portfolio_concentration_index', 'token_quality_diversification', 'xrp_portfolio_ratio', 'dormancy_score','liquidity_risk_score', 'activity_consistency', 'operational_risk_score']
        available_features = [col for col in all_feature_cols if col in raw_df.columns]
        X_all = raw_df[available_features]
        X_imputed_fs = SimpleImputer(strategy='median').fit_transform(X_all)
        X_scaled_fs = StandardScaler().fit_transform(X_imputed_fs)
        X_all_scaled = pd.DataFrame(X_scaled_fs, columns=X_all.columns)
        correlations = X_all_scaled.corrwith(X_all_scaled['total_transaction_count']).abs()
        strong_features = correlations[correlations > 0.3].index.tolist()
        key_heuristic_features = ['dormancy_score', 'operational_risk_score', 'liquidity_risk_score', 'high_risk_asset_ratio', 'activity_consistency']
        for feat in key_heuristic_features:
            if feat not in strong_features and feat in X_all.columns:
                strong_features.append(feat)
        feature_cols = list(set(strong_features))
        
        X = raw_df[feature_cols]
        X_imputed = SimpleImputer(strategy='median').fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X_imputed)
        
        core_indices = df[df['risk_cluster'] != -1].index
        X_scaled_core = X_scaled[core_indices]
        clusters = df.loc[core_indices, 'risk_cluster'].astype(int)
        
        # Re-fit the GMM to get the scores
        gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='full', n_init=5, random_state=42)
        gmm.fit(X_scaled_core)

        bic = gmm.bic(X_scaled_core)
        aic = gmm.aic(X_scaled_core)
        ch_score = calinski_harabasz_score(X_scaled_core, clusters)
        
        print(f"  Bayesian Information Criterion (BIC): {bic:.1f} (Lower is better)")
        print(f"  Akaike Information Criterion (AIC)  : {aic:.1f} (Lower is better)")
        print(f"  Calinski-Harabasz Score             : {ch_score:.1f} (Higher is better)")
        return bic, aic, ch_score
    except Exception as e:
        print(f"⚠️ Could not calculate GMM quality metrics: {e}")
        return None, None, None

def validate_model_consistency(df):
    """Validate the internal consistency of the model"""
    print("\n[5/5] Validating Model Consistency...")
    risk_order = ['Minimal Risk', 'Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    existing_categories = [cat for cat in risk_order if cat in df['risk_label'].unique()]
    df['risk_label_cat'] = pd.Categorical(df['risk_label'], categories=existing_categories, ordered=True)
    df['risk_ordinal'] = df['risk_label_cat'].cat.codes
    
    corr = df[['risk_score_pca', 'risk_ordinal']].corr().iloc[0, 1]
    print(f"Correlation between PCA risk score and risk category: {corr:.3f}")
    return corr

def generate_summary_report(is_monotonic, valid_indicators, score_category_corr, avg_confidence, bic):
    """Generate a final summary validation report"""
    print("\n" + "=" * 60)
    print("GMM RISK MODEL VALIDATION SUMMARY")
    print("=" * 60)
    
    # --- Define Pass/Fail Criteria ---
    monotonic_valid = is_monotonic
    confidence_valid = avg_confidence > 0.90
    alignment_valid = score_category_corr > 0.80

    print(f"1. Risk Distribution: {'VALID' if monotonic_valid else 'FAILED'}")
    print(f"   - {'Risk scores increase monotonically with risk category.' if monotonic_valid else 'CRITICAL FLAW: Risk scores are not monotonic.'}")
    
    print(f"2. Prediction Confidence: {'HIGH' if confidence_valid else 'MODERATE'}")
    print(f"   - Average Cluster Confidence is {avg_confidence:.3f}. (Target > 0.90)")

    print(f"3. Model Alignment: {'STRONG' if alignment_valid else 'MODERATE'}")
    print(f"   - Correlation between score and category is {score_category_corr:.3f}. (Target > 0.80)")
    
    print(f"4. GMM Fit Quality: {'INFO'}")
    print(f"   - BIC Score: {bic:.1f}. This provides a baseline for comparing future model versions (lower is better).")
    
    # --- Final Judgment ---
    overall_success = monotonic_valid and confidence_valid and alignment_valid

    print("\nOVERALL ASSESSMENT:")
    if overall_success:
        print("MODEL VALIDATED.")
        print("CONFIDENCE LEVEL: HIGH")
    else:
        print("MODEL NEEDS REVISION.")
        print("CONFIDENCE LEVEL: LOW")

if __name__ == "__main__":
    validate_gmm_model()