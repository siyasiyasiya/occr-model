import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

def extract_comprehensive_risk_features(wallet_data, token_df):
    """
    Extract all necessary features for wallet credit risk assessment
    """
    account_info = wallet_data.get('account_info', {})
    transactions = wallet_data.get('transactions', [])
    assets = wallet_data.get('assets', [])
    trustlines = wallet_data.get('trustlines', {}).get('lines', [])
    
    # Current time for calculations
    current_time = pd.Timestamp.now(tz='UTC')
    inception_str = account_info.get('inception')
    if inception_str:
        try:
            inception_time = pd.to_datetime(inception_str)
            if inception_time.tz is None:
                inception_time = inception_time.tz_localize('UTC')
            elif inception_time.tz != current_time.tz:
                inception_time = inception_time.tz_convert('UTC')
                
            account_age_days = (current_time - inception_time).days
        except Exception as e:
            raise ValueError(f"Invalid inception date format: {e}")
    else:
        raise ValueError("No inception date found - skipping file")
    
    features = {}
    
    # ACCOUNT FUNDAMENTALS
    features['account_age_days'] = account_age_days
    features['xrp_balance'] = float(account_info.get('xrpBalance', 0))
    features['initial_balance'] = account_info.get('initial_balance', 0)
    features['balance_to_initial_ratio'] = features['xrp_balance'] / max(features['initial_balance'], 1)
    
    # TRANSACTION PATTERNS
    total_txns = len(transactions)
    features['total_transaction_count'] = total_txns
    
    if total_txns > 0:
        # Success/failure analysis
        successful_txns = sum(1 for tx in transactions if tx.get('meta', {}).get('TransactionResult') == 'tesSUCCESS')
        features['transaction_success_rate'] = successful_txns / total_txns
        features['failed_transaction_count'] = total_txns - successful_txns
        
        # Transaction type diversity
        tx_types = [tx.get('TransactionType', 'Unknown') for tx in transactions]
        features['unique_transaction_types'] = len(set(tx_types))
        features['payment_transaction_ratio'] = sum(1 for t in tx_types if t == 'Payment') / total_txns
        features['trustset_transaction_ratio'] = sum(1 for t in tx_types if t == 'TrustSet') / total_txns
        
        # Timing analysis
        tx_dates = []
        for tx in transactions:
            date_str = tx.get('date')
            if date_str:
                try:
                    tx_dates.append(pd.to_datetime(date_str))
                except:
                    continue  # Skip invalid dates
        
        if tx_dates:
            features['days_since_last_transaction'] = (current_time - max(tx_dates)).days
            features['days_since_first_transaction'] = (current_time - min(tx_dates)).days
        else:
            features['days_since_last_transaction'] = 999
            features['days_since_first_transaction'] = 999
            
        features['transaction_frequency_per_day'] = total_txns / max(account_age_days, 1)
        
        
        # Recent activity (last 30 days)
        recent_txns = []
        for tx in transactions:
            date_str = tx.get('date')
            if date_str:
                try:
                    tx_date = pd.to_datetime(date_str)
                    if (current_time - tx_date).days <= 30:
                        recent_txns.append(tx)
                except:
                    continue

        features['recent_transaction_count'] = len(recent_txns)
        features['recent_activity_ratio'] = len(recent_txns) / total_txns
        features['is_recently_active'] = len(recent_txns) > 0
        
        # Recent failures
        recent_failures = sum(1 for tx in recent_txns if tx.get('meta', {}).get('TransactionResult') != 'tesSUCCESS')
        features['recent_failure_count'] = recent_failures
        features['has_recent_failures'] = recent_failures > 0
        
    else:
        # No transactions case
        features.update({
            'transaction_success_rate': 0, 'failed_transaction_count': 0,
            'unique_transaction_types': 0, 'payment_transaction_ratio': 0, 'trustset_transaction_ratio': 0,
            'days_since_last_transaction': 999, 'days_since_first_transaction': 999,
            'transaction_frequency_per_day': 0, 'recent_transaction_count': 0,
            'recent_activity_ratio': 0, 'is_recently_active': False,
            'recent_failure_count': 0, 'has_recent_failures': False
        })
    
    # FINANCIAL FLOWS
    outgoing_amounts = []
    incoming_amounts = []
    outgoing_xrp = []
    incoming_xrp = []
    unique_counterparties = set()
    large_transactions = []
    
    account_address = account_info.get('account', '')

    for tx in transactions:
        # Track counterparties
        tx_account = tx.get('Account')
        tx_destination = tx.get('Destination')
        
        if tx_account and tx_account != account_address:
            unique_counterparties.add(tx_account)
        if tx_destination and tx_destination != account_address:
            unique_counterparties.add(tx_destination)
        
        # Outgoing transactions
        if tx_account == account_address:
            amount = tx.get('Amount')
            if amount is not None:
                try:
                    if isinstance(amount, dict):
                        amount_value = float(amount.get('value', 0))
                        outgoing_amounts.append(amount_value)
                    else:
                        amount_xrp = float(amount) / 1000000  # Convert drops to XRP
                        outgoing_amounts.append(amount_xrp)
                        outgoing_xrp.append(amount_xrp)
                        
                        if amount_xrp > 10000:
                            large_transactions.append(amount_xrp)
                except (ValueError, TypeError):
                    continue  # Skip invalid amounts
        
        # Incoming transactions
        if tx_destination == account_address:
            meta = tx.get('meta', {})
            delivered = meta.get('delivered_amount')
            if delivered is not None:
                try:
                    if isinstance(delivered, dict):
                        amount_value = float(delivered.get('value', 0))
                        incoming_amounts.append(amount_value)
                    else:
                        amount_xrp = float(delivered) / 1000000
                        incoming_amounts.append(amount_xrp)
                        incoming_xrp.append(amount_xrp)
                except (ValueError, TypeError):
                    continue  # Skip invalid amounts
    
    # Financial flow features
    features['total_outgoing_value'] = sum(outgoing_amounts)
    features['total_incoming_value'] = sum(incoming_amounts)
    features['total_outgoing_xrp'] = sum(outgoing_xrp)
    features['total_incoming_xrp'] = sum(incoming_xrp)
    features['net_flow'] = sum(incoming_amounts) - sum(outgoing_amounts)
    features['net_xrp_flow'] = sum(incoming_xrp) - sum(outgoing_xrp)
    
    if outgoing_amounts:
        features['avg_outgoing_amount'] = np.mean(outgoing_amounts)
        features['max_outgoing_amount'] = max(outgoing_amounts)
        features['outgoing_amount_std'] = np.std(outgoing_amounts)
    else:
        features['avg_outgoing_amount'] = features['max_outgoing_amount'] = features['outgoing_amount_std'] = 0
    
    if incoming_amounts:
        features['avg_incoming_amount'] = np.mean(incoming_amounts)
        features['max_incoming_amount'] = max(incoming_amounts)
        features['incoming_amount_std'] = np.std(incoming_amounts)
    else:
        features['avg_incoming_amount'] = features['max_incoming_amount'] = features['incoming_amount_std'] = 0
    
    # Large transaction analysis
    features['large_transaction_count'] = len(large_transactions)
    features['largest_transaction_amount'] = max(large_transactions) if large_transactions else 0
    features['large_tx_to_balance_ratio'] = (max(large_transactions) / max(features['xrp_balance'], 1)) if large_transactions else 0
    
    # Counterparty diversity
    features['unique_counterparty_count'] = len(unique_counterparties)
    features['counterparty_diversity_ratio'] = len(unique_counterparties) / max(total_txns, 1)
    
    # ASSET HOLDINGS WITH TOKEN SCORE ANALYTICS
    features['total_assets_held'] = len(assets)
    
    if assets:
        asset_values = []
        for asset in assets:
            try:
                value = float(asset.get('value', 0))
                asset_values.append(value)
            except (ValueError, TypeError):
                asset_values.append(0)  # Use 0 for invalid values
        
        if asset_values and sum(asset_values) > 0:
            features['total_asset_value'] = sum(asset_values)
            features['max_single_asset_value'] = max(asset_values)
            features['asset_concentration_ratio'] = max(asset_values) / sum(asset_values)
            features['avg_asset_value'] = np.mean(asset_values)
        
            # TOKEN SCORE ANALYTICS FOR ASSETS
            asset_scores = []
            weighted_scores = []
            
            for i, asset in enumerate(assets):
                if i >= len(asset_values):
                    break

                currency = asset.get('currency', '')
                counterparty = asset.get('counterparty', '')

                if currency and counterparty:
                    # Look up token score
                    token_row = token_df[
                        (token_df['issuer'] == asset.get('counterparty')) &
                        (token_df['currency'] == asset.get('currency'))
                    ]
                    if not token_row.empty:
                        try:
                            token_score = float(token_row.iloc[0]['score'])
                            asset_value = asset_values[i]
                                
                            asset_scores.append(token_score)
                            weighted_scores.append(token_score * asset_value)
                        except (ValueError, TypeError, IndexError):
                            continue  # Skip invalid scores
            
            if asset_scores:
                features['asset_avg_token_score'] = np.mean(asset_scores)
                features['asset_min_token_score'] = min(asset_scores)
                features['asset_max_token_score'] = max(asset_scores)
                features['asset_token_score_std'] = np.std(asset_scores) if len(asset_scores) > 1 else 0
                features['asset_weighted_avg_token_score'] = sum(weighted_scores) / sum(asset_values) if sum(asset_values) > 0 else 0
                
                # Risk categorization based on token scores
                high_risk_assets = sum(1 for score in asset_scores if score <= 0.003135)  # Bottom 25% from your distribution
                medium_risk_assets = sum(1 for score in asset_scores if 0.003135 < score <= 0.231348)
                low_risk_assets = sum(1 for score in asset_scores if score > 0.231348)
                
                features['high_risk_asset_count'] = high_risk_assets
                features['medium_risk_asset_count'] = medium_risk_assets
                features['low_risk_asset_count'] = low_risk_assets
                features['high_risk_asset_ratio'] = high_risk_assets / len(asset_scores)
                
                # Value exposure to risky assets
                high_risk_value = 0
                scored_asset_indices = []
                j = 0
                for i, asset in enumerate(assets):
                    currency = asset.get('currency', '')
                    counterparty = asset.get('counterparty', '')
                    if currency and counterparty:
                        token_row = token_df[
                            (token_df['issuer'] == asset.get('counterparty')) &
                            (token_df['currency'] == asset.get('currency'))
                        ]
                        if not token_row.empty:
                            try:
                                token_score = float(token_row.iloc[0]['score'])
                                if token_score <= 0.003135 and i < len(asset_values):
                                    high_risk_value += asset_values[i]
                            except (ValueError, TypeError):
                                continue
                            
                features['high_risk_asset_value_exposure'] = high_risk_value
                features['high_risk_value_ratio'] = high_risk_value / sum(asset_values) if sum(asset_values) > 0 else 0
                
            else:
                # No token scores found
                features.update({
                    'asset_avg_token_score': 0, 'asset_min_token_score': 0, 'asset_max_token_score': 0,
                    'asset_token_score_std': 0, 'asset_weighted_avg_token_score': 0,
                    'high_risk_asset_count': 0, 'medium_risk_asset_count': 0, 'low_risk_asset_count': 0,
                    'high_risk_asset_ratio': 0, 'high_risk_asset_value_exposure': 0, 'high_risk_value_ratio': 0
                })
            
            # Verified vs unverified assets (original verification)
            verified_assets = []
            for asset in assets:
                counterparty_name = asset.get('counterpartyName', {})
                if isinstance(counterparty_name, dict) and counterparty_name.get('verified', False):
                    verified_assets.append(asset)
            
            features['verified_assets_count'] = len(verified_assets)
            features['verified_assets_ratio'] = len(verified_assets) / len(assets)
            
            verified_value = 0
            for asset in verified_assets:
                try:
                    verified_value += float(asset.get('value', 0))
                except (ValueError, TypeError):
                    continue
            
            features['verified_asset_value'] = verified_value
            features['verified_value_ratio'] = verified_value / sum(asset_values) if sum(asset_values) > 0 else 0

        else:
            # No valid asset values
            features.update({
                'total_asset_value': 0, 'max_single_asset_value': 0, 'asset_concentration_ratio': 0, 'avg_asset_value': 0,
                'asset_avg_token_score': 0, 'asset_min_token_score': 0, 'asset_max_token_score': 0,
                'asset_token_score_std': 0, 'asset_weighted_avg_token_score': 0,
                'high_risk_asset_count': 0, 'medium_risk_asset_count': 0, 'low_risk_asset_count': 0,
                'high_risk_asset_ratio': 0, 'high_risk_asset_value_exposure': 0, 'high_risk_value_ratio': 0,
                'verified_assets_count': 0, 'verified_assets_ratio': 0, 'verified_asset_value': 0, 'verified_value_ratio': 0
            })
    else:
        # No assets
        features.update({
            'total_asset_value': 0, 'max_single_asset_value': 0, 'asset_concentration_ratio': 0, 'avg_asset_value': 0,
            'asset_avg_token_score': 0, 'asset_min_token_score': 0, 'asset_max_token_score': 0,
            'asset_token_score_std': 0, 'asset_weighted_avg_token_score': 0,
            'high_risk_asset_count': 0, 'medium_risk_asset_count': 0, 'low_risk_asset_count': 0,
            'high_risk_asset_ratio': 0, 'high_risk_asset_value_exposure': 0, 'high_risk_value_ratio': 0,
            'verified_assets_count': 0, 'verified_assets_ratio': 0, 'verified_asset_value': 0, 'verified_value_ratio': 0
        })
    
    # TRUSTLINE ANALYSIS
    features['trustline_count'] = len(trustlines)
    
    if trustlines:
        all_balances = []
        active_balances = []
        limits = []
        
        for line in trustlines:
            try:
                balance = float(line.get('balance', 0))
                limit = float(line.get('limit', 0))
                all_balances.append(balance)
                limits.append(limit)
                if balance > 0:
                    active_balances.append(balance)
            except (ValueError, TypeError):
                all_balances.append(0)
                limits.append(0)
                continue
        
        features['total_trustline_balance'] = sum(all_balances)
        features['total_trustline_limit'] = sum(limits)
        features['trustline_utilization_ratio'] = sum(all_balances) / max(sum(limits), 1)
        features['active_trustlines_count'] = len(active_balances)
        
        if active_balances:
            features['avg_active_trustline_balance'] = np.mean(active_balances)
            features['max_trustline_balance'] = max(active_balances)
        else:
            features['avg_active_trustline_balance'] = 0
            features['max_trustline_balance'] = 0
        
        # No ripple settings (risk management indicator)
        no_ripple_count = sum(1 for line in trustlines if line.get('no_ripple', False))
        features['no_ripple_trustlines_count'] = no_ripple_count
        features['no_ripple_ratio'] = no_ripple_count / len(trustlines)
        
        # Quality settings
        quality_in_values = []
        quality_out_values = []
        for line in trustlines:
            try:
                quality_in_values.append(float(line.get('quality_in', 0)))
                quality_out_values.append(float(line.get('quality_out', 0)))
            except (ValueError, TypeError):
                quality_in_values.append(0)
                quality_out_values.append(0)
        
        features['avg_quality_in'] = np.mean(quality_in_values) if quality_in_values else 0
        features['avg_quality_out'] = np.mean(quality_out_values) if quality_out_values else 0
        
        # Trustline efficiency
        features['trustline_utilization_efficiency'] = features['active_trustlines_count'] / len(trustlines)
        features['unused_trustlines_count'] = len(trustlines) - features['active_trustlines_count']
        
    else:
        features.update({
            'total_trustline_balance': 0, 'total_trustline_limit': 0, 'trustline_utilization_ratio': 0,
            'active_trustlines_count': 0, 'avg_active_trustline_balance': 0, 'max_trustline_balance': 0,
            'no_ripple_trustlines_count': 0, 'no_ripple_ratio': 0, 'avg_quality_in': 0, 'avg_quality_out': 0,
            'trustline_utilization_efficiency': 0, 'unused_trustlines_count': 0
        })
    
    # SIMPLIFIED PORTFOLIO DIVERSIFICATION (ASSETS ONLY)
    # Calculate total portfolio value (XRP + assets only)
    total_portfolio_value = features['xrp_balance'] + features['total_asset_value']

    
    # Basic portfolio composition
    features['total_portfolio_value'] = total_portfolio_value
    features['xrp_portfolio_ratio'] = features['xrp_balance'] / max(total_portfolio_value, 1)
    features['asset_portfolio_ratio'] = features['total_asset_value'] / max(total_portfolio_value, 1)
    
    # Portfolio holding count (XRP + number of assets)
    total_holdings = 1 + features['total_assets_held']  # +1 for XRP
    features['portfolio_holding_count'] = total_holdings
    
    # Concentration risk (Herfindahl-Hirschman Index)
    portfolio_components = [features['xrp_balance']]
    if assets:
        for asset in assets:
            try:
                value = float(asset.get('value', 0))
                portfolio_components.append(value)
            except (ValueError, TypeError):
                portfolio_components.append(0)
    
    if total_portfolio_value > 0:
        portfolio_shares = [comp / total_portfolio_value for comp in portfolio_components]
        features['portfolio_concentration_index'] = sum(share ** 2 for share in portfolio_shares)  # Higher = more concentrated
    else:
        features['portfolio_concentration_index'] = 1  # Fully concentrated if no value
    
    # Token quality diversification (for assets only)
    if assets and features['asset_avg_token_score'] > 0:  # Only if we have scored assets
        asset_scores = []
        for asset in assets:
            currency = asset.get('currency', '')
            counterparty = asset.get('counterparty', '')
            if currency and counterparty:
                token_row = token_df[
                    (token_df['issuer'] == asset.get('counterparty')) &
                    (token_df['currency'] == asset.get('currency'))
                ]
                if not token_row.empty:
                    try:
                        score = float(token_row.iloc[0]['score'])
                        asset_scores.append(score)
                    except (ValueError, TypeError, IndexError):
                        continue
        
        if asset_scores:
            # Count assets in each risk category
            high_risk_count = sum(1 for score in asset_scores if score <= 0.003135)
            medium_risk_count = sum(1 for score in asset_scores if 0.003135 < score <= 0.231348)
            low_risk_count = sum(1 for score in asset_scores if score > 0.231348)
            
            # Simple diversification across risk categories
            risk_categories_used = sum([high_risk_count > 0, medium_risk_count > 0, low_risk_count > 0])
            features['token_quality_diversification'] = risk_categories_used / 3  # 0-1 scale
        else:
            features['token_quality_diversification'] = 0
    else:
        features['token_quality_diversification'] = 0
    
    # RISK INDICATORS
    # Dormancy risk
    features['is_dormant'] = features['recent_transaction_count'] == 0
    features['dormancy_score'] = min(features['days_since_last_transaction'] / 90, 1)  # 0-1 scale, 90+ days = 1
    
    # Liquidity risk
    features['liquidity_risk_score'] = 1 / (1 + features['xrp_balance'])  # Higher balance = lower risk
    
    # Activity consistency
    if total_txns > 1:
        tx_intervals = []
        tx_dates = []
        for tx in transactions:
            date_str = tx.get('date')
            if date_str:
                try:
                    tx_dates.append(pd.to_datetime(date_str))
                except:
                    continue
        
        if len(tx_dates) > 1:
            sorted_dates = sorted(tx_dates)
            for i in range(1, len(sorted_dates)):
                interval = (sorted_dates[i] - sorted_dates[i-1]).days
                tx_intervals.append(interval)
            
            if tx_intervals:
                features['avg_transaction_interval'] = np.mean(tx_intervals)
                features['transaction_interval_std'] = np.std(tx_intervals) if len(tx_intervals) > 1 else 0
                features['activity_consistency'] = 1 / (1 + features['transaction_interval_std'])
            else:
                features['avg_transaction_interval'] = features['transaction_interval_std'] = features['activity_consistency'] = 0
        else:
            features['avg_transaction_interval'] = features['transaction_interval_std'] = features['activity_consistency'] = 0
    else:
        features['avg_transaction_interval'] = features['transaction_interval_std'] = features['activity_consistency'] = 0
    
    # Operational risk
    features['operational_risk_score'] = 1 - features['transaction_success_rate']
    
    return features

def save_feature_modules(all_features, output_dir):
    """
    Save features in logical modules for better organization
    """
    features_df = pd.DataFrame(all_features)
    
    # Core identifiers (always needed)
    core_cols = ['account_id', 'file_name']
    
    # Feature modules
    feature_modules = {
        'account_fundamentals': [
            'account_age_days', 'xrp_balance', 'initial_balance', 'balance_to_initial_ratio'
        ],
        
        'transaction_patterns': [
            'total_transaction_count', 'transaction_success_rate', 'failed_transaction_count',
            'unique_transaction_types', 'payment_transaction_ratio', 'trustset_transaction_ratio',
            'days_since_last_transaction', 'days_since_first_transaction', 'transaction_frequency_per_day',
            'recent_transaction_count', 'recent_activity_ratio', 'is_recently_active', 
            'recent_failure_count', 'has_recent_failures'
        ],
        
        'financial_flows': [
            'total_outgoing_value', 'total_incoming_value', 'total_outgoing_xrp', 'total_incoming_xrp',
            'net_flow', 'net_xrp_flow', 'avg_outgoing_amount', 'max_outgoing_amount', 'outgoing_amount_std',
            'avg_incoming_amount', 'max_incoming_amount', 'incoming_amount_std', 'large_transaction_count', 
            'largest_transaction_amount', 'large_tx_to_balance_ratio', 'unique_counterparty_count', 'counterparty_diversity_ratio'
        ],
        
        'asset_analytics': [
            'total_assets_held', 'total_asset_value', 'max_single_asset_value', 'asset_concentration_ratio',
            'avg_asset_value', 'asset_avg_token_score', 'asset_min_token_score', 'asset_max_token_score',
            'asset_token_score_std', 'asset_weighted_avg_token_score', 'high_risk_asset_count',
            'medium_risk_asset_count', 'low_risk_asset_count', 'high_risk_asset_ratio',
            'high_risk_asset_value_exposure', 'high_risk_value_ratio', 'verified_assets_count',
            'verified_assets_ratio', 'verified_asset_value', 'verified_value_ratio'
        ],
        
        'portfolio_risk': [
            'total_portfolio_value', 'xrp_portfolio_ratio', 'asset_portfolio_ratio',
            'portfolio_holding_count', 'portfolio_concentration_index', 'token_quality_diversification'
        ],
        
        'trustline_analysis': [
            'trustline_count', 'total_trustline_balance', 'total_trustline_limit', 'trustline_utilization_ratio',
            'active_trustlines_count', 'avg_active_trustline_balance', 'max_trustline_balance',
            'no_ripple_trustlines_count', 'no_ripple_ratio', 'avg_quality_in', 'avg_quality_out',
            'trustline_utilization_efficiency', 'unused_trustlines_count'
        ],
        
        'risk_indicators': [
            'is_dormant', 'dormancy_score', 'liquidity_risk_score', 'avg_transaction_interval',
            'transaction_interval_std', 'activity_consistency', 'operational_risk_score'
        ]
    }
    
    # Save each module
    for module_name, feature_list in feature_modules.items():
        # Include core columns + module features
        module_cols = core_cols + [col for col in feature_list if col in features_df.columns]
        module_df = features_df[module_cols]
        
        # Save module
        module_file = output_dir / f"{module_name}.csv"
        module_df.to_csv(module_file, index=False)
        print(f"Saved {module_name}: {len(module_cols)-2} features")
    
    # Also save a lightweight summary file for quick analysis
    summary_cols = core_cols + [
        'xrp_balance', 'total_transaction_count', 'total_assets_held',
        'asset_avg_token_score', 'portfolio_concentration_index', 'operational_risk_score'
    ]

    summary_cols = [col for col in summary_cols if col in features_df.columns]  # Only include existing columns
    summary_df = features_df[summary_cols]
    summary_file = output_dir / "account_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved account_summary: {len(summary_cols)-2} key features")
    
    return feature_modules

def main():
    print("=" * 60)
    print("WALLET RISK FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Processing started at: 2025-08-11 17:24:38 UTC")
    print(f"User: siyasiyasiya")
    print("=" * 60)
    
    # ==================== LOAD TOKEN DATA ====================
    print("\n1. Loading token data...")
    
    # Load your 1000 tokens JSON file
    token_file_path = "../data/all_tokens.json"
    
    try:
        with open(token_file_path, 'r') as f:
            tokens_data = json.load(f)
        
        # Convert to DataFrame with just token and score
        token_records = []
        for token in tokens_data:
            issuer = token.get('issuer')
            currency = token.get('currency')

            token_records.append({
                'issuer': issuer,
                'currency': currency,
                'score': token['score']
            })
        
        token_df = pd.DataFrame(token_records)
        print(f"   ✅ Loaded {len(token_df)} tokens successfully")
        print(f"   📊 Token score range: {token_df['score'].min():.6f} to {token_df['score'].max():.6f}")
        
    except FileNotFoundError:
        print(f"   ❌ Error: Token file not found at {token_file_path}")
        print("   Please update the token_file_path variable with the correct path")
        return
    except Exception as e:
        print(f"   ❌ Error loading token data: {e}")
        return
    
    # ==================== PROCESS ACCOUNT FILES ====================
    print("\n2. Processing account files...")
    
    # Set up paths
    accounts_dir = Path("../data/accounts")
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Check if accounts directory exists
    if not accounts_dir.exists():
        print(f"   ❌ Error: Accounts directory not found at {accounts_dir}")
        print("   Please ensure the data/accounts directory exists with JSON files")
        return
    
    # Get all JSON files in the accounts directory
    json_files = list(accounts_dir.glob("*.json"))
    
    if not json_files:
        print(f"   ❌ No JSON files found in {accounts_dir}")
        return
    
    print(f"   📁 Found {len(json_files)} account files to process")
    
    # Process each account file
    all_features = []
    successful_extractions = 0
    failed_extractions = 0
    failed_files = []
    
    for i, json_file in enumerate(json_files, 1):
        try:
            # Load wallet data
            with open(json_file, 'r') as f:
                wallet_data = json.load(f)
            
            # Extract features
            features = extract_comprehensive_risk_features(wallet_data, token_df)
            
            # Add account identifier
            features['account_id'] = wallet_data['account_info']['account']
            features['file_name'] = json_file.name
            
            all_features.append(features)
            successful_extractions += 1
            
            # Progress update every 50 files or at the end
            if i % 20 == 0 or i == len(json_files):
                print(f"   🔄 Processed {i}/{len(json_files)} files... ({successful_extractions} successful)")
            
        except Exception as e:
            print(f"   ⚠️  Error processing {json_file.name}: {e}")
            failed_extractions += 1
            failed_files.append(json_file.name)
            continue
    
    print(f"\n   ✅ Processing complete!")
    print(f"   📈 Successfully processed: {successful_extractions} accounts")
    if failed_extractions > 0:
        print(f"   ❌ Failed to process: {failed_extractions} accounts")
        print(f"   📋 Failed files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    if not all_features:
        print("   ❌ No features extracted. Please check your data files.")
        return
    
    # ==================== SAVE RESULTS ====================
    print("\n3. Saving results...")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Reorder columns to put identifiers first
    id_columns = ['account_id', 'file_name']
    feature_columns = [col for col in features_df.columns if col not in id_columns]
    features_df = features_df[id_columns + feature_columns]
    
    # Save modular feature files
    print("   📁 Creating modular feature files...")
    feature_modules = save_feature_modules(all_features, output_dir)
    
    # Save complete dataset (for reference only)
    complete_file = output_dir / "complete_features.csv"
    features_df.to_csv(complete_file, index=False)
    print(f"   💾 Complete dataset saved to: {complete_file}")
    
    # Save summary statistics
    summary_file = output_dir / "feature_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("WALLET RISK FEATURE EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing Date: 2025-08-11 17:24:38 UTC\n")
        f.write(f"User: siyasiyasiya\n")
        f.write(f"Total accounts processed: {successful_extractions}\n")
        f.write(f"Failed extractions: {failed_extractions}\n")
        f.write(f"Total features extracted: {len(feature_columns)}\n\n")
        
        f.write("FEATURE MODULES CREATED:\n")
        f.write("-" * 25 + "\n")
        for module_name, feature_list in feature_modules.items():
            actual_features = [col for col in feature_list if col in features_df.columns]
            f.write(f"{module_name}: {len(actual_features)} features\n")
        
        f.write(f"\nFEATURE SUMMARY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
        f.write(numeric_features.describe().to_string())
        
        if failed_files:
            f.write(f"\n\nFAILED FILES:\n")
            f.write("-" * 15 + "\n")
            for failed_file in failed_files:
                f.write(f"- {failed_file}\n")
    
    print(f"   📊 Summary saved to: {summary_file}")
    
    # Display basic info
    print(f"\n4. Results Summary:")
    print(f"   📏 Dataset shape: {features_df.shape}")
    print(f"   🔢 Features per account: {len(feature_columns)}")
    print(f"   📂 Modular files created: {len(feature_modules)} modules")
    
    print(f"\n   📋 Sample of processed accounts:")
    sample_cols = ['account_id', 'xrp_balance', 'total_assets_held', 'total_transaction_count']
    available_cols = [col for col in sample_cols if col in features_df.columns]
    print(features_df[available_cols].head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎉 FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print("\nOutput files created:")
    print("📁 output/account_summary.csv - Quick analysis file")
    print("📁 output/account_fundamentals.csv - Basic account info")
    print("📁 output/transaction_patterns.csv - Transaction behavior")
    print("📁 output/financial_flows.csv - Money flow analysis")
    print("📁 output/asset_analytics.csv - Token holdings analysis")
    print("📁 output/portfolio_risk.csv - Portfolio metrics")
    print("📁 output/trustline_analysis.csv - Trustline data")
    print("📁 output/risk_indicators.csv - Risk scores")
    print("📁 output/complete_features.csv - All features combined")
    print("📁 output/feature_summary.txt - Processing summary")

if __name__ == "__main__":
    main()