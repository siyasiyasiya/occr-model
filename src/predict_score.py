import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
import json
import time
from datetime import datetime

# --- Configuration ---
ARTIFACTS_DIR = Path('../artifacts/')
API_BASE_URL = "https://api.xrpscan.com/api/v1"
# Maintain a safe request rate (approx. 1 per second)
TARGET_TIME_PER_REQUEST = 1.0

# --- 1. Load All Trained Model Artifacts at Startup ---
print(" LTR (Load-Time-Resources): Loading model artifacts...")
try:
    scaler = joblib.load(ARTIFACTS_DIR / 'scaler.joblib')
    feature_cols = joblib.load(ARTIFACTS_DIR / 'feature_cols.joblib')
    iso_forest = joblib.load(ARTIFACTS_DIR / 'isolation_forest.joblib')
    gmm = joblib.load(ARTIFACTS_DIR / 'gmm.joblib')
    pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
    cluster_risk_mapping = joblib.load(ARTIFACTS_DIR / 'cluster_risk_mapping.joblib')
    pca_params = joblib.load(ARTIFACTS_DIR / 'pca_scaling_params.joblib')
    # In a production system, this would be loaded once.
    with open("../all_tokens.json", 'r') as f:
        tokens_data = json.load(f)
    token_df = pd.DataFrame(tokens_data)
    
    print("...artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load artifacts. Did you run the training script first? Details: {e}")
    exit()

# --- 2. Targeted On-Chain Data Collection ---
def _api_request(endpoint, params=None):
    """A helper function to safely make API requests."""
    start_time = time.time()
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        json_response = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"    - HTTP Error for endpoint {endpoint}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    - Request Error for endpoint {endpoint}: {e}")
        return None
    
    duration = time.time() - start_time
    wait_time = TARGET_TIME_PER_REQUEST - duration
    if wait_time > 0:
        time.sleep(wait_time)
        
    return json_response

def get_live_data_for_address(address: str):
    """
    Fetches only the necessary data from the API to generate the required features.
    """
    print(f"\n Collecting live on-chain data for {address}...")
    
    # 1. Account Info (Needed for age, balance)
    print("  - Fetching Account Info...")
    account_info = _api_request(f"/account/{address}")
    
    # 2. Transactions (Needed for activity, dormancy, etc.) - Fetch up to 15 pages
    print("  - Fetching Transaction History (this may take a moment)...")
    all_transactions = []
    marker = None
    for page in range(15): # Limit to 15 pages for reasonable performance
        params = {'limit': 25, 'marker': marker} if marker else {'limit': 25}
        data = _api_request(f"/account/{address}/transactions", params)
        if not data or 'transactions' not in data or not data['transactions']:
            break
        all_transactions.extend(data['transactions'])
        if 'marker' not in data:
            break
        marker = data['marker']
        
    # 3. Assets (Needed for portfolio features)
    print("  - Fetching Assets...")
    assets = _api_request(f"/account/{address}/assets")

    if not account_info:
        raise ValueError("Could not fetch essential account info. Cannot proceed.")

    return {
        "account_info": account_info,
        "transactions": all_transactions,
        "assets": assets if assets else [] # Ensure assets is a list
    }

# --- 3. Lean Feature Generation ---
def generate_features_from_live_data(live_data: dict, token_df: pd.DataFrame):
    """
    Takes live data and generates only the specific features needed by the model.
    This is a streamlined version of your feature engineering script.
    """
    print("\n Generating required features from live data...")
    account_info = live_data.get('account_info', {})
    transactions = live_data.get('transactions', [])
    assets = live_data.get('assets', [])
    
    features = {}
    current_time = pd.Timestamp.now(tz='UTC')

    # --- ACCOUNT FUNDAMENTALS ---
    inception_str = account_info.get('inception')
    if not inception_str:
        raise ValueError("Could not find 'inception' date for the account.")
        
    inception_time = pd.to_datetime(inception_str)
    
    if inception_time.tzinfo is None:
        inception_time = inception_time.tz_localize('UTC')
    else:
        inception_time = inception_time.tz_convert('UTC')

    features['account_age_days'] = (current_time - inception_time).days
    features['xrp_balance'] = float(account_info.get('xrpBalance', 0))
    features['initial_balance'] = float(account_info.get('initial_balance', 0))
    features['balance_to_initial_ratio'] = features['xrp_balance'] / max(features['initial_balance'], 1)

    # --- TRANSACTION PATTERNS ---
    total_txns = len(transactions)
    features['total_transaction_count'] = total_txns
    successful_txns = sum(1 for tx in transactions if tx.get('meta', {}).get('TransactionResult') == 'tesSUCCESS')
    features['transaction_success_rate'] = successful_txns / max(total_txns, 1)
    features['failed_transaction_count'] = total_txns - successful_txns
    tx_types = [tx.get('TransactionType', 'Unknown') for tx in transactions]
    features['payment_transaction_ratio'] = sum(1 for t in tx_types if t == 'Payment') / max(total_txns, 1)
    
    tx_dates = []
    for tx in transactions:
        date_str = tx.get('date')
        if date_str:
            try:
                tx_date = pd.to_datetime(date_str)
                if tx_date.tzinfo is None:
                    tx_date = tx_date.tz_localize('UTC')
                else:
                    tx_date = tx_date.tz_convert('UTC')
                tx_dates.append(tx_date)
            except (ValueError, TypeError):
                continue
    features['days_since_last_transaction'] = (current_time - max(tx_dates)).days if tx_dates else 999
    features['transaction_frequency_per_day'] = total_txns / max(features['account_age_days'], 1)
    
    recent_txns = [tx for tx in tx_dates if (current_time - tx).days <= 30]
    features['recent_activity_ratio'] = len(recent_txns) / max(total_txns, 1)
    features['recent_failure_count'] = features['failed_transaction_count'] # Simplified for live scoring
    
    # --- FINANCIAL FLOWS ---
    unique_counterparties = set(tx.get('Destination') for tx in transactions if 'Destination' in tx and tx['Destination'] != account_info['account'])
    features['counterparty_diversity_ratio'] = len(unique_counterparties) / max(total_txns, 1)
    
    outgoing_xrp = [float(tx['Amount'])/1e6 for tx in transactions if tx.get('Amount') and isinstance(tx['Amount'], str) and tx.get('Account') == account_info['account']]
    features['avg_outgoing_amount'] = np.mean(outgoing_xrp) if outgoing_xrp else 0
    
    # --- ASSET & PORTFOLIO FEATURES ---
    features['total_assets_held'] = len(assets)
    asset_values = [float(a.get('value', 0)) for a in assets]
    features['total_asset_value'] = sum(asset_values)
    features['total_portfolio_value'] = features['xrp_balance'] + features['total_asset_value']
    features['xrp_portfolio_ratio'] = features['xrp_balance'] / max(features['total_portfolio_value'], 1)

    asset_scores = []
    weighted_scores = []
    for asset in assets:
        token_row = token_df[(token_df['issuer'] == asset.get('counterparty')) & (token_df['currency'] == asset.get('currency'))]
        if not token_row.empty:
            score = float(token_row.iloc[0]['score'])
            asset_scores.append(score)
            weighted_scores.append(score * float(asset.get('value', 0)))

    features['asset_avg_token_score'] = np.mean(asset_scores) if asset_scores else 0
    features['asset_weighted_avg_token_score'] = sum(weighted_scores) / max(features['total_asset_value'], 1) if weighted_scores else 0
    
    high_risk_assets = sum(1 for score in asset_scores if score <= 0.003135)
    features['high_risk_asset_ratio'] = high_risk_assets / max(len(asset_scores), 1)
    
    features['verified_assets_ratio'] = sum(1 for a in assets if a.get('counterpartyName', {}).get('verified')) / max(len(assets), 1)
    
    if features['total_portfolio_value'] > 0:
        portfolio_shares = [features['xrp_balance'] / features['total_portfolio_value']] + [v / features['total_portfolio_value'] for v in asset_values]
        features['portfolio_concentration_index'] = sum(s**2 for s in portfolio_shares)
    else:
        features['portfolio_concentration_index'] = 1
        
    risk_categories_used = sum([high_risk_assets > 0, sum(1 for s in asset_scores if 0.003135 < s <= 0.231348) > 0, sum(1 for s in asset_scores if s > 0.231348) > 0])
    features['token_quality_diversification'] = risk_categories_used / 3.0
    
    # --- RISK INDICATORS ---
    features['dormancy_score'] = min(features['days_since_last_transaction'] / 90.0, 1.0)
    features['liquidity_risk_score'] = 1 / (1 + features['xrp_balance'])
    
    if len(tx_dates) > 1:
        intervals = np.diff(sorted(tx_dates)).astype('timedelta64[s]').astype(float) / (24 * 3600) # in days
        features['activity_consistency'] = 1 / (1 + np.std(intervals)) if len(intervals) > 0 else 0
    else:
        features['activity_consistency'] = 0
        
    features['operational_risk_score'] = 1 - features['transaction_success_rate']

    print("...features generated.")
    return pd.DataFrame([features])

# --- 4. The Prediction Pipeline ---
def predict_risk_for_address(address: str):
    """
    Orchestrates the entire prediction process for a single address.
    """
    try:
        # Step 1: Get live data
        live_data = get_live_data_for_address(address)
        
        # Step 2: Generate features from live data
        features_df = generate_features_from_live_data(live_data, token_df)
        
        # Step 3: Ensure column order and preprocess
        # Fill any missing columns in the new data with 0 (in case some features couldn't be calculated)
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[feature_cols] # Enforce same column order as training
        
        scaled_features = scaler.transform(features_df)
        
        # Step 4: Get cluster assignment
        is_outlier = iso_forest.predict(scaled_features)[0]
        cluster_id = -1 if is_outlier == -1 else gmm.predict(scaled_features)[0]
        
        # Step 5: Get final PCA score
        pca_score = pca.transform(scaled_features)[0, 0]
        
        # --- Align and scale the PCA score ---
        if pca_params['needs_flipping']:
            pca_score = -pca_score

        # Get the min and max from the loaded parameters
        pca_min_val = pca_params['min']
        pca_max_val = pca_params['max']

        # Scale the score to 0-100 using the exact same parameters as the training data
        if (pca_max_val - pca_min_val) != 0:
            final_score = 100 * (pca_score - pca_min_val) / (pca_max_val - pca_min_val)
        else:
            final_score = 50 # Assign neutral score if there was no variance in training
        final_score = np.clip(final_score, 0, 100) # Ensure score is between 0 and 100
        
        # Step 6: Map to final label and create result
        risk_label = cluster_risk_mapping.get(cluster_id, "Unknown")
        
        result = {
            'account_address': address,
            'risk_score': round(final_score, 2),
            'risk_label': risk_label,
            'assigned_cluster': int(cluster_id)
        }
        
        return result

    except Exception as e:
        print(f"❌ An error occurred during the prediction process: {e}")
        return {
            'account_address': address,
            'error': str(e)
        }

# --- 5. Make the Script Executable ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_score.py <ACCOUNT_ADDRESS>")
        sys.exit(1)
        
    input_address = sys.argv[1]
    
    final_result = predict_risk_for_address(input_address)
    
    print("\n--- On-Chain Credit Score Report ---")
    for key, value in final_result.items():
        print(f"  {key.replace('_', ' ').title():<20}: {value}")
    print("------------------------------------")