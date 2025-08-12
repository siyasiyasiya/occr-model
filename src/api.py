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
import time
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
import asyncio
import logging

# --- Import XRPL Libraries ---
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.models.requests import AccountInfo, AccountTx, AccountLines, LedgerCurrent
from xrpl.models.response import Response
from xrpl.utils import ripple_time_to_datetime

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-score-api")

# --- 1. Configuration ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_TIME_PER_REQUEST = 1.0  # Like the original API, throttle requests

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
    with open("all_tokens.json", 'r') as f:
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

async def _api_request(client, request_fn, *args, **kwargs):
    """
    Throttled API request function to mimic the original behavior
    """
    start_time = time.time()
    try:
        result = await request_fn(*args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"API request error: {e}")
        return None
    finally:
        # Throttle like the original implementation
        duration = time.time() - start_time
        wait_time = TARGET_TIME_PER_REQUEST - duration
        if wait_time > 0:
            await asyncio.sleep(wait_time)

async def get_account_first_transaction(client, address):
    """
    Determine when the account was created by finding the first transaction
    """
    try:
        # Get current ledger index
        ledger_request = LedgerCurrent()
        ledger_response = await client.request(ledger_request)
        if not ledger_response.is_successful():
            return None
            
        current_ledger = ledger_response.result.get("ledger_current_index", 0)
        
        # Start from earliest ledger and look for transactions
        first_tx = None
        marker = None
        
        # First, try with a big limit to get the oldest transactions
        tx_request = AccountTx(
            account=address, 
            ledger_index_min=1,  # Start from the beginning
            ledger_index_max=current_ledger,
            limit=200,  # Get a large batch to minimize requests
            forward=True  # Get oldest first
        )
        
        tx_response = await client.request(tx_request)
        if tx_response.is_successful() and tx_response.result.get("transactions"):
            transactions = tx_response.result["transactions"]
            # Get the oldest transaction
            if transactions:
                first_tx = transactions[0]
                
        if first_tx and "tx" in first_tx and "date" in first_tx["tx"]:
            tx_date = ripple_time_to_datetime(first_tx["tx"]["date"])
            return tx_date.isoformat()
            
        return None
    except Exception as e:
        logger.error(f"Error finding first transaction: {e}")
        return None

async def get_assets_with_verification(client, address):
    """
    Get account assets with verification status (estimated)
    """
    assets = []
    try:
        lines_request = AccountLines(account=address)
        lines_response = await _api_request(client, client.request, lines_request)
        
        if lines_response and lines_response.is_successful():
            for line in lines_response.result.get("lines", []):
                issuer = line.get("account")
                currency = line.get("currency")
                
                # Try to match with token database for verification status
                is_verified = False
                if token_df is not None:
                    token_match = token_df[(token_df['issuer'] == issuer) & 
                                           (token_df['currency'] == currency)]
                    if not token_match.empty:
                        # If we have a match in our token database, consider it verified
                        is_verified = True
                
                asset = {
                    "currency": currency,
                    "counterparty": issuer,
                    "value": line.get("balance"),
                    "counterpartyName": {
                        "verified": is_verified,
                        "name": "Unknown"  # We don't have this data
                    }
                }
                assets.append(asset)
                
            # Check for additional pages (marker)
            marker = lines_response.result.get("marker")
            while marker:
                lines_request = AccountLines(
                    account=address,
                    marker=marker
                )
                lines_response = await _api_request(client, client.request, lines_request)
                if not (lines_response and lines_response.is_successful()):
                    break
                    
                for line in lines_response.result.get("lines", []):
                    issuer = line.get("account")
                    currency = line.get("currency")
                    
                    # Check verification status
                    is_verified = False
                    if token_df is not None:
                        token_match = token_df[(token_df['issuer'] == issuer) & 
                                              (token_df['currency'] == currency)]
                        if not token_match.empty:
                            is_verified = True
                    
                    asset = {
                        "currency": currency,
                        "counterparty": issuer,
                        "value": line.get("balance"),
                        "counterpartyName": {
                            "verified": is_verified,
                            "name": "Unknown"
                        }
                    }
                    assets.append(asset)
                
                marker = lines_response.result.get("marker")
                
    except Exception as e:
        logger.error(f"Error fetching account assets: {e}")
    
    return assets

async def get_all_transactions_paginated(client, address):
    """
    Get all transactions using pagination, just like the original XRPScan API implementation
    """
    all_transactions = []
    marker = None
    
    # Similar to the original code, limit to 15 pagination calls
    for _ in range(15):
        try:
            if marker:
                tx_request = AccountTx(account=address, limit=25, marker=marker)
            else:
                tx_request = AccountTx(account=address, limit=25)
                
            tx_response = await _api_request(client, client.request, tx_request)
            
            if not (tx_response and tx_response.is_successful()):
                break
                
            txs = tx_response.result.get("transactions", [])
            if not txs:
                break
                
            # Format transactions to match XRPScan structure
            for tx_entry in txs:
                tx = tx_entry.get("tx", {})
                # Add meta data which contains transaction result
                tx["meta"] = tx_entry.get("meta", {})
                
                if "date" in tx:
                    # Convert Ripple epoch time to ISO format datetime
                    tx_date = ripple_time_to_datetime(tx["date"])
                    tx["date"] = tx_date.isoformat()
                    
                all_transactions.append(tx)
            
            # Check for more pages
            marker = tx_response.result.get("marker")
            if not marker:
                break
                
        except Exception as e:
            logger.error(f"Error in transaction pagination: {e}")
            break
            
    return all_transactions

async def get_account_initial_balance(client, address, first_tx_date=None):
    """
    Estimate initial balance - try to find earliest transactions that funded the account
    """
    try:
        # Default to 10 XRP (base reserve)
        initial_balance = 10.0
        
        # If we know the first transaction date, we can try to find funding transactions
        if first_tx_date:
            # This is a simplification - in reality this would require more complex analysis
            # of the earliest transactions
            pass
            
        return str(initial_balance)
    except Exception as e:
        logger.error(f"Error estimating initial balance: {e}")
        return "10"  # Default to base reserve

async def get_live_data_for_address(address: str, network: str):
    """
    Fetches live data directly from an XRPL node, formatted to match XRPScan API structure
    """
    node_url = XRPL_NODE_URLS.get(network)
    if not node_url:
        raise ValueError(f"Invalid network specified: {network}")
        
    logger.info(f"\n Collecting live data for {address} from {network}...")
    client = AsyncJsonRpcClient(node_url)
    
    # Get basic account info
    acc_info_request = AccountInfo(account=address, ledger_index="validated")
    acc_info_response = await _api_request(client, client.request, acc_info_request)

    if not (acc_info_response and acc_info_response.is_successful()):
        raise ValueError(f"Could not fetch account info for {address}. The account may not be activated on {network}.")
    
    account_data = acc_info_response.result["account_data"]
    
    # Get account inception (first transaction date)
    inception_date = await get_account_first_transaction(client, address)
    
    # Get all transactions with pagination (like original code)
    transactions = await get_all_transactions_paginated(client, address)
    
    # Estimate initial balance
    initial_balance = await get_account_initial_balance(client, address, inception_date)
    
    # Get assets with verification status
    assets = await get_assets_with_verification(client, address)
    
    # Format account info to match XRPScan API structure
    account_info_formatted = {
        "account": account_data["Account"],
        "xrpBalance": str(float(account_data["Balance"]) / 1_000_000),  # Convert drops to XRP
        "inception": inception_date,
        "initial_balance": initial_balance
    }

    # Log data structure for debugging
    logger.info(f"Account data retrieved for {address}: Balance={account_info_formatted['xrpBalance']} XRP, " +
               f"Transactions={len(transactions)}, Assets={len(assets)}")

    return {
        "account_info": account_info_formatted,
        "transactions": transactions,
        "assets": assets
    }

def generate_features_from_live_data(live_data: dict, token_df: pd.DataFrame):
    """
    Takes live data and generates only the specific features needed by the model.
    This is almost identical to the original function.
    """
    logger.info("\n Generating required features from live data...")
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
            logger.warning(f"Could not parse inception date: {inception_str}. Using default account age.")
            features['account_age_days'] = 30  # Default value
    else:
        logger.warning("No inception date found. Using default account age.")
        features['account_age_days'] = 30  # Default value

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
    
    if tx_dates:
        features['days_since_last_transaction'] = (current_time - max(tx_dates)).days
        # This also implies the account is at least as old as its first transaction
        if features['account_age_days'] is None: # This check may be redundant but is safe
            features['account_age_days'] = (current_time - min(tx_dates)).days
    else:
        # If there are no transactions, the last transaction was effectively at creation
        features['days_since_last_transaction'] = features.get('account_age_days', 999)
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

    # Log feature statistics for debugging
    logger.info("Feature generation complete. Key features:")
    logger.info(f"  Account age: {features['account_age_days']} days")
    logger.info(f"  XRP balance: {features['xrp_balance']} XRP")
    logger.info(f"  Transaction count: {features['total_transaction_count']}")
    logger.info(f"  Asset count: {features['total_assets_held']}")
    
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
    logger.info(f"Received scoring request for address: {address} on {network}")
    try:
        # Get live data with proper structure
        live_data = await get_live_data_for_address(address, network)
        
        # Generate features
        features_df = generate_features_from_live_data(live_data, token_df)
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[feature_cols]
        
        # Apply model pipeline
        scaled_features = scaler.transform(features_df.values)
        is_outlier = iso_forest.predict(scaled_features)[0]
        cluster_id = -1 if is_outlier == -1 else gmm.predict(scaled_features)[0]
        pca_score = pca.transform(scaled_features)[0, 0]
        
        if pca_params['needs_flipping']:
            pca_score = -pca_score
            
        final_score = 100 * (pca_score - pca_params['min']) / (pca_params['max'] - pca_params['min'])
        final_score = np.clip(final_score, 0, 100)
        
        risk_label = cluster_risk_mapping.get(cluster_id, "Unknown")
        
        # Log the final score
        logger.info(f"Credit score for {address}: {round(final_score, 2)} ({risk_label})")
        
        return {
            "address": address,
            "network": network,
            "risk_score": round(final_score, 2),
            "risk_label": risk_label,
            "model_version": "v7-gmm",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error processing request for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while scoring the address: {str(e)}")

# --- 5. A Root Endpoint for Health Checks ---
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "On-Chain Credit Score API is running"}