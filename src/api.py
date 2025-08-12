# # api.py

# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path
# import requests
# import json
# import time
# from datetime import datetime
# from fastapi import FastAPI, HTTPException

# # --- Configuration ---
# ARTIFACTS_DIR = Path('./artifacts/')
# API_BASE_URL = "https://api.xrpscan.com/api/v1"
# TARGET_TIME_PER_REQUEST = 1.0

# # --- 1. Load All Model Artifacts at Startup ---
# # This block runs only ONCE when the API server starts up.
# print(" LTR: Loading model artifacts...")
# try:
#     scaler = joblib.load(ARTIFACTS_DIR / 'scaler.joblib')
#     feature_cols = joblib.load(ARTIFACTS_DIR / 'feature_cols.joblib')
#     iso_forest = joblib.load(ARTIFACTS_DIR / 'isolation_forest.joblib')
#     gmm = joblib.load(ARTIFACTS_DIR / 'gmm.joblib')
#     pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
#     cluster_risk_mapping = joblib.load(ARTIFACTS_DIR / 'cluster_risk_mapping.joblib')
#     pca_params = joblib.load(ARTIFACTS_DIR / 'pca_scaling_params.joblib')
#     with open("./all_tokens.json", 'r') as f:
#         tokens_data = json.load(f)
#     token_df = pd.DataFrame(tokens_data)
#     print("...artifacts loaded successfully.")
# except FileNotFoundError as e:
#     print(f"❌ FATAL ERROR: Could not load artifacts. API cannot start. Details: {e}")
#     exit()

# # --- Initialize the FastAPI app ---
# app = FastAPI(
#     title="On-Chain Credit Score API",
#     description="An API to get a risk score for XRPL accounts.",
#     version="1.0.0"
# )

# # --- 2. Helper Functions ---

# def _api_request(endpoint, params=None):
#     start_time = time.time()
#     try:
#         response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"API request error: {e}")
#         return None
#     finally:
#         duration = time.time() - start_time
#         wait_time = TARGET_TIME_PER_REQUEST - duration
#         if wait_time > 0:
#             time.sleep(wait_time)

# def get_live_data_for_address(address: str):
#     print(f"\n Collecting live on-chain data for {address}...")
#     account_info = _api_request(f"/account/{address}")
#     if not account_info: raise ValueError("Could not fetch essential account info.")
#     all_transactions = []
#     marker = None
#     for _ in range(15):
#         params = {'limit': 25, 'marker': marker} if marker else {'limit': 25}
#         data = _api_request(f"/account/{address}/transactions", params)
#         if not data or not data.get('transactions'): break
#         all_transactions.extend(data['transactions'])
#         if 'marker' not in data: break
#         marker = data['marker']
#     assets = _api_request(f"/account/{address}/assets")
#     return {"account_info": account_info, "transactions": all_transactions, "assets": assets or []}

# def generate_features_from_live_data(live_data: dict, token_df: pd.DataFrame):
#     """
#     Takes live data and generates only the specific features needed by the model.
#     This is a streamlined version of your feature engineering script.
#     """
#     print("\n Generating required features from live data...")
#     account_info = live_data.get('account_info', {})
#     transactions = live_data.get('transactions', [])
#     assets = live_data.get('assets', [])
    
#     features = {}
#     current_time = pd.Timestamp.now(tz='UTC')

#     # --- ACCOUNT FUNDAMENTALS ---
#     inception_str = account_info.get('inception')
#     if not inception_str:
#         raise ValueError("Could not find 'inception' date for the account.")
        
#     inception_time = pd.to_datetime(inception_str)
    
#     if inception_time.tzinfo is None:
#         inception_time = inception_time.tz_localize('UTC')
#     else:
#         inception_time = inception_time.tz_convert('UTC')

#     features['account_age_days'] = (current_time - inception_time).days
#     features['xrp_balance'] = float(account_info.get('xrpBalance', 0))
#     features['initial_balance'] = float(account_info.get('initial_balance', 0))
#     features['balance_to_initial_ratio'] = features['xrp_balance'] / max(features['initial_balance'], 1)

#     # --- TRANSACTION PATTERNS ---
#     total_txns = len(transactions)
#     features['total_transaction_count'] = total_txns
#     successful_txns = sum(1 for tx in transactions if tx.get('meta', {}).get('TransactionResult') == 'tesSUCCESS')
#     features['transaction_success_rate'] = successful_txns / max(total_txns, 1)
#     features['failed_transaction_count'] = total_txns - successful_txns
#     tx_types = [tx.get('TransactionType', 'Unknown') for tx in transactions]
#     features['payment_transaction_ratio'] = sum(1 for t in tx_types if t == 'Payment') / max(total_txns, 1)
    
#     tx_dates = []
#     for tx in transactions:
#         date_str = tx.get('date')
#         if date_str:
#             try:
#                 tx_date = pd.to_datetime(date_str)
#                 if tx_date.tzinfo is None:
#                     tx_date = tx_date.tz_localize('UTC')
#                 else:
#                     tx_date = tx_date.tz_convert('UTC')
#                 tx_dates.append(tx_date)
#             except (ValueError, TypeError):
#                 continue
#     features['days_since_last_transaction'] = (current_time - max(tx_dates)).days if tx_dates else 999
#     features['transaction_frequency_per_day'] = total_txns / max(features['account_age_days'], 1)
    
#     recent_txns = [tx for tx in tx_dates if (current_time - tx).days <= 30]
#     features['recent_activity_ratio'] = len(recent_txns) / max(total_txns, 1)
#     features['recent_failure_count'] = features['failed_transaction_count'] # Simplified for live scoring
    
#     # --- FINANCIAL FLOWS ---
#     unique_counterparties = set(tx.get('Destination') for tx in transactions if 'Destination' in tx and tx['Destination'] != account_info['account'])
#     features['counterparty_diversity_ratio'] = len(unique_counterparties) / max(total_txns, 1)
    
#     outgoing_xrp = [float(tx['Amount'])/1e6 for tx in transactions if tx.get('Amount') and isinstance(tx['Amount'], str) and tx.get('Account') == account_info['account']]
#     features['avg_outgoing_amount'] = np.mean(outgoing_xrp) if outgoing_xrp else 0
    
#     # --- ASSET & PORTFOLIO FEATURES ---
#     features['total_assets_held'] = len(assets)
#     asset_values = [float(a.get('value', 0)) for a in assets]
#     features['total_asset_value'] = sum(asset_values)
#     features['total_portfolio_value'] = features['xrp_balance'] + features['total_asset_value']
#     features['xrp_portfolio_ratio'] = features['xrp_balance'] / max(features['total_portfolio_value'], 1)

#     asset_scores = []
#     weighted_scores = []
#     for asset in assets:
#         token_row = token_df[(token_df['issuer'] == asset.get('counterparty')) & (token_df['currency'] == asset.get('currency'))]
#         if not token_row.empty:
#             score = float(token_row.iloc[0]['score'])
#             asset_scores.append(score)
#             weighted_scores.append(score * float(asset.get('value', 0)))

#     features['asset_avg_token_score'] = np.mean(asset_scores) if asset_scores else 0
#     features['asset_weighted_avg_token_score'] = sum(weighted_scores) / max(features['total_asset_value'], 1) if weighted_scores else 0
    
#     high_risk_assets = sum(1 for score in asset_scores if score <= 0.003135)
#     features['high_risk_asset_ratio'] = high_risk_assets / max(len(asset_scores), 1)
    
#     features['verified_assets_ratio'] = sum(1 for a in assets if a.get('counterpartyName', {}).get('verified')) / max(len(assets), 1)
    
#     if features['total_portfolio_value'] > 0:
#         portfolio_shares = [features['xrp_balance'] / features['total_portfolio_value']] + [v / features['total_portfolio_value'] for v in asset_values]
#         features['portfolio_concentration_index'] = sum(s**2 for s in portfolio_shares)
#     else:
#         features['portfolio_concentration_index'] = 1
        
#     risk_categories_used = sum([high_risk_assets > 0, sum(1 for s in asset_scores if 0.003135 < s <= 0.231348) > 0, sum(1 for s in asset_scores if s > 0.231348) > 0])
#     features['token_quality_diversification'] = risk_categories_used / 3.0
    
#     # --- RISK INDICATORS ---
#     features['dormancy_score'] = min(features['days_since_last_transaction'] / 90.0, 1.0)
#     features['liquidity_risk_score'] = 1 / (1 + features['xrp_balance'])
    
#     if len(tx_dates) > 1:
#         intervals = np.diff(sorted(tx_dates)).astype('timedelta64[s]').astype(float) / (24 * 3600) # in days
#         features['activity_consistency'] = 1 / (1 + np.std(intervals)) if len(intervals) > 0 else 0
#     else:
#         features['activity_consistency'] = 0
        
#     features['operational_risk_score'] = 1 - features['transaction_success_rate']

#     print("...features generated.")
#     return pd.DataFrame([features])


# # --- 3. The API Endpoint ---
# @app.get("/score")
# def get_credit_score(address: str):
#     """
#     Takes an XRPL account address and returns its risk score and label.
#     This endpoint orchestrates the entire prediction process.
#     """
#     print(f"Received scoring request for address: {address}")
#     try:
#         live_data = get_live_data_for_address(address)
#         features_df = generate_features_from_live_data(live_data, token_df)
        
#         for col in feature_cols:
#             if col not in features_df.columns:
#                 features_df[col] = 0
#         features_df = features_df[feature_cols]
        
#         scaled_features = scaler.transform(features_df.values)
        
#         is_outlier = iso_forest.predict(scaled_features)[0]
#         cluster_id = -1 if is_outlier == -1 else gmm.predict(scaled_features)[0]
        
#         pca_score = pca.transform(scaled_features)[0, 0]
        
#         if pca_params['needs_flipping']:
#             pca_score = -pca_score
            
#         final_score = 100 * (pca_score - pca_params['min']) / (pca_params['max'] - pca_params['min'])
#         final_score = np.clip(final_score, 0, 100)
        
#         risk_label = cluster_risk_mapping.get(cluster_id, "Unknown")
        
#         # Instead of printing, we return a dictionary. FastAPI handles the JSON conversion.
#         return {
#             "address": address,
#             "risk_score": round(final_score, 2),
#             "risk_label": risk_label,
#             "model_version": "v7-gmm",
#             "timestamp": datetime.utcnow().isoformat()
#         }

#     except Exception as e:
#         # Return a proper HTTP error to the client.
#         print(f"❌ Error processing request for {address}: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred while scoring the address: {str(e)}")

# # --- 4. A Root Endpoint for Health Checks ---
# @app.get("/")
# def read_root():
#     """A simple endpoint to confirm the API is running."""
#     return {"status": "ok", "message": "On-Chain Credit Score API is running"}

# api.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
import asyncio
import time

# --- Import XRPL Libraries ---
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.models.requests import AccountInfo, AccountTx, AccountLines, LedgerEntry
from xrpl.models.response import Response
from xrpl.utils import datetime_to_ripple_time, ripple_time_to_datetime

# --- 1. Robust Path and Configuration Setup ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Public node URLs for direct connection
XRPL_NODE_URLS = {
    "mainnet": "https://s1.ripple.com:51234/",
    "testnet": "https://s.altnet.rippletest.net:51234/"
}

# --- 2. Load All Model Artifacts at Startup ---
print(" LTR: Loading model artifacts...")
try:
    scaler = joblib.load(ARTIFACTS_DIR / 'scaler.joblib')
    feature_cols = joblib.load(ARTIFACTS_DIR / 'feature_cols.joblib')
    iso_forest = joblib.load(ARTIFACTS_DIR / 'isolation_forest.joblib')
    gmm = joblib.load(ARTIFACTS_DIR / 'gmm.joblib')
    pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
    cluster_risk_mapping = joblib.load(ARTIFACTS_DIR / 'cluster_risk_mapping.joblib')
    pca_params = joblib.load(ARTIFACTS_DIR / 'pca_scaling_params.joblib')
    with open(PROJECT_ROOT / "all_tokens.json", 'r') as f:
        tokens_data = json.load(f)
    token_df = pd.DataFrame(tokens_data)
    print("...artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ FATAL ERROR: Could not load artifacts. API cannot start. Details: {e}")
    exit()

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="On-Chain Credit Score API",
    description="An API to get a risk score for XRPL accounts.",
    version="1.0.0"
)

# --- 3. Helper Functions ---
async def fetch_account_inception(client, address):
    """Fetches the account creation time by examining ledger entry"""
    try:
        ledger_request = LedgerEntry(
            ledger_index="validated",
            account_root=address
        )
        response = await client.request(ledger_request)
        if response.is_successful():
            # The 'index' field can be used to determine the first ledger this account appeared in
            # This is a simplification - a more accurate approach would be to use account_tx with
            # a binary search to find the first transaction for this account
            first_ledger = response.result.get("ledger_current_index", 0)
            # Estimate time based on ledger (rough approximation)
            ripple_epoch_time = int(time.time() - (first_ledger * 4))  # Approx 4 seconds per ledger
            return ripple_time_to_datetime(ripple_epoch_time).isoformat()
    except Exception as e:
        print(f"Error fetching account inception: {e}")
    
    # Fallback: if we can't determine inception, use the date of the oldest transaction
    return None

async def fetch_account_assets(client, address):
    """Fetches all assets (tokens) held by the account"""
    assets = []
    try:
        lines_request = AccountLines(account=address)
        lines_response = await client.request(lines_request)
        
        if lines_response.is_successful():
            for line in lines_response.result.get("lines", []):
                asset = {
                    "currency": line.get("currency"),
                    "counterparty": line.get("account"),
                    "value": line.get("balance"),
                    "counterpartyName": {"verified": False}  # Default to unverified
                }
                assets.append(asset)
    except Exception as e:
        print(f"Error fetching account assets: {e}")
    
    return assets

async def get_live_data_for_address(address: str, network: str):
    """
    Fetches live data directly from an XRPL node using xrpl-py.
    Formatted to match the structure expected by the feature generator.
    """
    node_url = XRPL_NODE_URLS.get(network)
    if not node_url:
        raise ValueError(f"Invalid network specified: {network}")
        
    print(f"\n Collecting live data for {address} from {network}...")
    client = AsyncJsonRpcClient(node_url)
    
    # Run requests in parallel for speed
    acc_info_request = AccountInfo(account=address, ledger_index="validated")
    acc_tx_request = AccountTx(account=address, limit=400)  # Fetch a large batch of recent transactions

    acc_info_response, acc_tx_response = await asyncio.gather(
        client.request(acc_info_request),
        client.request(acc_tx_request)
    )

    if not acc_info_response.is_successful():
        raise ValueError(f"Could not fetch account info for {address}. The account may not be activated on {network}.")
    
    account_data = acc_info_response.result["account_data"]
    
    # Fetch assets held by the account
    assets = await fetch_account_assets(client, address)
    
    # Reformat transaction data to match the structure your feature generator expects
    transactions_formatted = []
    tx_dates = []
    
    if acc_tx_response.is_successful():
        for tx_entry in acc_tx_response.result.get("transactions", []):
            tx = tx_entry.get("tx", {})
            # Add meta data which contains transaction result
            tx["meta"] = tx_entry.get("meta", {})
            
            if "date" in tx:
                # Convert Ripple epoch time to ISO format datetime
                tx_date = ripple_time_to_datetime(tx["date"])
                tx["date"] = tx_date.isoformat()
                tx_dates.append(tx_date)
                
            transactions_formatted.append(tx)
    
    # Get account inception date (or earliest transaction if not available)
    inception_date = await fetch_account_inception(client, address)
    if not inception_date and tx_dates:
        # Use earliest transaction as fallback
        inception_date = min(tx_dates).isoformat()
    
    # Calculate initial balance (approximation based on current reserves)
    reserve_base = 10  # XRP reserve base on XRPL (10 XRP for activation)
    
    # Reformat account info to match original structure
    account_info_formatted = {
        "account": account_data["Account"],
        "xrpBalance": str(float(account_data["Balance"]) / 1_000_000),  # Convert drops to XRP
        "inception": inception_date,
        "initial_balance": str(reserve_base)  # Use base reserve as approximation
    }

    return {
        "account_info": account_info_formatted,
        "transactions": transactions_formatted,
        "assets": assets
    }

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
    if inception_str:
        try:
            inception_time = pd.to_datetime(inception_str)
            if inception_time.tzinfo is None:
                inception_time = inception_time.tz_localize('UTC')
            else:
                inception_time = inception_time.tz_convert('UTC')
            features['account_age_days'] = (current_time - inception_time).days
        except (ValueError, TypeError):
            # Fallback if date parsing fails
            features['account_age_days'] = 1
    else:
        features['account_age_days'] = 1

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
    
    outgoing_xrp = []
    for tx in transactions:
        if tx.get('Account') == account_info['account'] and 'Amount' in tx:
            amount = tx['Amount']
            # Handle both string amounts (XRP) and object amounts (tokens)
            if isinstance(amount, str):
                outgoing_xrp.append(float(amount)/1e6)
            elif isinstance(amount, dict) and 'value' in amount:
                # This is a token amount, not XRP
                pass
    
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


# --- 4. The API Endpoint ---
@app.get("/score")
async def get_credit_score(
    address: str,
    network: str = Query("testnet", enum=["mainnet", "testnet"])
):
    """
    Takes an XRPL account address and returns its risk score and label.
    This endpoint orchestrates the entire prediction process.
    """
    print(f"Received scoring request for address: {address} on {network}")
    try:
        live_data = await get_live_data_for_address(address, network)
        features_df = generate_features_from_live_data(live_data, token_df)
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[feature_cols]
        
        scaled_features = scaler.transform(features_df.values)
        
        is_outlier = iso_forest.predict(scaled_features)[0]
        cluster_id = -1 if is_outlier == -1 else gmm.predict(scaled_features)[0]
        
        pca_score = pca.transform(scaled_features)[0, 0]
        
        if pca_params['needs_flipping']:
            pca_score = -pca_score
            
        final_score = 100 * (pca_score - pca_params['min']) / (pca_params['max'] - pca_params['min'])
        final_score = np.clip(final_score, 0, 100)
        
        risk_label = cluster_risk_mapping.get(cluster_id, "Unknown")
        
        return {
            "address": address,
            "network": network,
            "risk_score": round(final_score, 2),
            "risk_label": risk_label,
            "model_version": "v7-gmm",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"❌ Error processing request for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while scoring the address: {str(e)}")

# --- 5. A Root Endpoint for Health Checks ---
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "On-Chain Credit Score API is running"}