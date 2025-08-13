# OCCR Model - On-Chain Credit Risk Engine

A comprehensive machine learning risk assessment system for XRPL (XRP Ledger) accounts that provides real-time credit scoring based on on-chain behavioral analysis and portfolio composition.

## 🔍 Overview

The OCCR (On-Chain Credit Risk) Model is a ML system that analyzes XRPL account behavior to generate comprehensive credit risk scores. The system processes account history, transaction patterns, asset holdings, and behavioral metrics to assess creditworthiness using sophisticated clustering and anomaly detection techniques.

### Key Features

- **Comprehensive Risk Assessment**: Analyzes 65+ behavioral and financial features across 7 risk dimensions
- **Advanced ML Pipeline**: Uses Isolation Forest for anomaly detection, GMM for behavioral clustering, and PCA for continuous scoring
- **Real-time XRPL Integration**: Direct connection to XRPL nodes and XRPScan API
- **Token Quality Analysis**: Incorporates asset verification and quality scoring from 1000+ token database
- **Production-ready API**: FastAPI endpoint with both XRPScan API and native XRPL node support
- **Model Validation Suite**: Comprehensive validation framework with statistical testing
- **Modular Architecture**: Clean separation between data collection, feature engineering, model training, and inference

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Feature Engineering](#feature-engineering)
- [Model Pipeline](#model-pipeline)
- [API Documentation](#api-documentation)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Model Validation](#model-validation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Engine │    │   ML Pipeline   │
│                 │────▶│                 │────▶│                 │
│ • XRPL Nodes    │    │ • 65+ Features  │    │ • Isolation     │
│ • XRPScan API   │    │ • 7 Dimensions  │    │   Forest        │
│ • Token DB      │    │ • Risk Metrics  │    │ • GMM (5 Clust) │
└─────────────────┘    │ • Portfolio     │    │ • PCA Scoring   │
                       │   Analytics     │    │ • Risk Labels   │
                       └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐              │
                       │   FastAPI       │◀─────────────┘
                       │   Endpoint      │
                       │ • /score        │
                       │ • Validation    │
                       └─────────────────┘
```

### Risk Assessment Framework

The system evaluates seven key risk dimensions:

1. **Account Fundamentals** (4 features) - Age, balance patterns, inception analysis
2. **Transaction Patterns** (14 features) - Volume, frequency, success rates, timing analysis  
3. **Financial Flows** (17 features) - Payment patterns, counterparty diversity, large transactions
4. **Asset Analytics** (20 features) - Token holdings, quality scoring, risk categorization
5. **Portfolio Risk** (6 features) - Diversification, concentration, composition analysis
6. **Trustline Analysis** (13 features) - Credit relationships, utilization patterns
7. **Risk Indicators** (7 features) - Dormancy, liquidity, operational risk, consistency

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
Docker (optional)
```

### Dependencies

```bash
# Core ML and Data Processing
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0

# API and Web Framework  
fastapi>=0.68.0
uvicorn>=0.15.0
requests>=2.26.0

# XRPL Integration
xrpl-py>=2.0.0

# Visualization (for validation)
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
pathlib
asyncio
logging
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/siyasiyasiya/occr-model.git
cd occr-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/accounts output artifacts
```

### Required Data Files

The system requires:
- `all_tokens.json` - Token quality database (1000+ tokens with scores)
- Pre-trained model artifacts in `artifacts/` directory (generated after training)

## ⚡ Quick Start

### 1. Data Collection

Collect XRPL account data for training:

```bash
# Create input file with account addresses
echo '["rAccount1...", "rAccount2..."]' > test_accounts.json

# Collect account data
python src/query_data.py
```

### 2. Feature Engineering

Extract comprehensive risk features:

```bash
# Generate feature dataset
python src/ml_features.py

# Output: Modular CSV files in output/ directory
```

### 3. Model Training

Train the ML risk model:

```bash
# Train GMM-based risk model
python src/ml_model.py

# Output: Model artifacts in artifacts/ directory
```

### 4. Model Validation

Validate model performance:

```bash
# Run comprehensive validation suite
python src/model_validation.py

# Output: Validation reports and plots
```

### 5. Start API Server

```bash
# Run the FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Server available at http://localhost:8000
```

### 6. Score an Account

```bash
# Direct scoring script
python src/predict_score.py rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH

# Or via API
curl "http://localhost:8000/score?address=rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH&network=mainnet"
```

## 📊 Feature Engineering

### Complete Feature Set (65 Features)

#### Account Fundamentals (4 features)
- `account_age_days`: Days since account inception
- `xrp_balance`: Current XRP balance  
- `initial_balance`: Estimated initial funding
- `balance_to_initial_ratio`: Balance growth ratio

#### Transaction Patterns (14 features)
- `total_transaction_count`: Total transaction volume
- `transaction_success_rate`: Success rate percentage
- `failed_transaction_count`: Number of failed transactions
- `unique_transaction_types`: Diversity of transaction types
- `payment_transaction_ratio`: Percentage of payment transactions
- `trustset_transaction_ratio`: Percentage of trustline transactions
- `days_since_last_transaction`: Days since last activity
- `days_since_first_transaction`: Account activity history length
- `transaction_frequency_per_day`: Daily transaction rate
- `recent_transaction_count`: 30-day transaction count
- `recent_activity_ratio`: Recent vs historical activity
- `recent_failure_count`: Recent failed transactions
- `is_recently_active`: Boolean recent activity flag
- `has_recent_failures`: Boolean recent failure flag

#### Financial Flows (17 features)
- `total_outgoing_value`: Total outbound transaction value
- `total_incoming_value`: Total inbound transaction value
- `total_outgoing_xrp`: XRP-specific outbound volume
- `total_incoming_xrp`: XRP-specific inbound volume
- `net_flow`: Net transaction flow
- `net_xrp_flow`: Net XRP flow
- `avg_outgoing_amount`: Average outbound transaction size
- `max_outgoing_amount`: Largest outbound transaction
- `outgoing_amount_std`: Outbound transaction variance
- `avg_incoming_amount`: Average inbound transaction size
- `max_incoming_amount`: Largest inbound transaction
- `incoming_amount_std`: Inbound transaction variance
- `large_transaction_count`: Transactions >10,000 XRP
- `largest_transaction_amount`: Maximum single transaction
- `large_tx_to_balance_ratio`: Large transaction relative to balance
- `unique_counterparty_count`: Number of unique trading partners
- `counterparty_diversity_ratio`: Counterparty diversity metric

#### Asset Analytics (20 features)
- `total_assets_held`: Number of different tokens
- `total_asset_value`: Combined token portfolio value
- `max_single_asset_value`: Largest single asset holding
- `asset_concentration_ratio`: Asset concentration measure
- `avg_asset_value`: Average asset holding size
- `asset_avg_token_score`: Average quality score of held tokens
- `asset_min_token_score`: Lowest quality token held
- `asset_max_token_score`: Highest quality token held
- `asset_token_score_std`: Token quality variance
- `asset_weighted_avg_token_score`: Value-weighted quality score
- `high_risk_asset_count`: Number of high-risk tokens (score ≤ 0.003135)
- `medium_risk_asset_count`: Number of medium-risk tokens
- `low_risk_asset_count`: Number of low-risk tokens
- `high_risk_asset_ratio`: Percentage of high-risk assets
- `high_risk_asset_value_exposure`: Value exposure to risky assets
- `high_risk_value_ratio`: Percentage value in risky assets
- `verified_assets_count`: Number of verified tokens
- `verified_assets_ratio`: Percentage of verified tokens
- `verified_asset_value`: Value in verified tokens
- `verified_value_ratio`: Percentage value in verified tokens

#### Portfolio Risk (6 features)
- `total_portfolio_value`: XRP + token combined value
- `xrp_portfolio_ratio`: XRP percentage of portfolio
- `asset_portfolio_ratio`: Token percentage of portfolio
- `portfolio_holding_count`: Total number of holdings (XRP + tokens)
- `portfolio_concentration_index`: Herfindahl-Hirschman concentration index
- `token_quality_diversification`: Risk category diversification (0-1 scale)

#### Trustline Analysis (13 features)
- `trustline_count`: Number of trustlines established
- `total_trustline_balance`: Combined trustline balances
- `total_trustline_limit`: Combined trustline limits
- `trustline_utilization_ratio`: Balance/limit utilization
- `active_trustlines_count`: Trustlines with positive balances
- `avg_active_trustline_balance`: Average active balance
- `max_trustline_balance`: Largest trustline balance
- `no_ripple_trustlines_count`: Risk management settings
- `no_ripple_ratio`: Percentage with no-ripple setting
- `avg_quality_in`: Average quality_in settings
- `avg_quality_out`: Average quality_out settings
- `trustline_utilization_efficiency`: Active/total trustline ratio
- `unused_trustlines_count`: Inactive trustlines

#### Risk Indicators (7 features)
- `is_dormant`: Boolean dormancy flag
- `dormancy_score`: Account inactivity score (0-1, 90+ days = 1)
- `liquidity_risk_score`: Liquidity risk based on XRP balance
- `avg_transaction_interval`: Average days between transactions
- `transaction_interval_std`: Transaction timing consistency
- `activity_consistency`: Regularity of transaction patterns
- `operational_risk_score`: Risk based on transaction failure rates

## 🤖 Model Pipeline

### Stage 1: Anomaly Detection (Isolation Forest)
- Identifies extreme outliers using Isolation Forest (2% contamination rate)
- Accounts flagged as anomalies are classified as "Minimal Risk" (hyper-active accounts)
- Separates core population from statistical outliers

### Stage 2: Behavioral Clustering (Gaussian Mixture Model)
- GMM with 5 components clusters normal accounts into distinct behavioral profiles
- Full covariance matrices capture complex feature relationships
- Cluster confidence scores indicate prediction certainty

### Stage 3: Continuous Risk Scoring (PCA)
- Principal Component Analysis projects all accounts onto single risk dimension  
- Scores normalized to 0-100 scale (higher = lower risk)
- Maintains continuous ranking within discrete risk categories

### Stage 4: Risk Label Assignment
Pre-defined cluster mappings based on PCA scores:
- **Minimal Risk**: Anomalous accounts (statistical outliers)
- **Very Low Risk**: Cluster with highest PCA scores
- **Low Risk**: Second-highest scoring cluster
- **Medium Risk**: Middle-scoring cluster  
- **High Risk**: Second-lowest scoring cluster
- **Very High Risk**: Lowest-scoring cluster

## 📡 API Documentation

### Endpoints

#### `GET /score`

Generates comprehensive credit risk score for an XRPL account.

**Parameters:**
- `address` (required): XRPL account address (e.g., "rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH")
- `network` (optional): "mainnet" or "testnet" (default: "testnet")

**Response:**
```json
{
  "address": "rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH",
  "network": "mainnet",
  "risk_score": 75.42,
  "risk_label": "Low Risk", 
  "model_version": "v7-gmm",
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

#### `GET /`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "On-Chain Credit Score API is running"
}
```

### Integration Examples

#### Python

```python
import requests

response = requests.get(
    "http://localhost:8000/score",
    params={
        "address": "rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH",
        "network": "mainnet"
    }
)

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_label']}")
```

#### cURL

```bash
curl -X GET "http://localhost:8000/score?address=rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH&network=mainnet"
```

#### JavaScript

```javascript
const response = await fetch(
  'http://localhost:8000/score?address=rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH&network=mainnet'
);
const result = await response.json();
console.log(`Risk Score: ${result.risk_score}`);
```

## 🔄 Data Collection

The `query_data.py` script collects comprehensive account data from XRPScan API:

### Features
- **Rate-limited requests**: 58 requests per minute to respect API limits
- **Comprehensive data**: Account info, transaction history (15 pages), assets, trustlines
- **Retry logic**: Automatic retry with backoff for rate limits
- **Batch processing**: Processes multiple accounts from JSON input file

### Usage

```bash
# Create input file with account addresses
echo '["rAccount1", "rAccount2", "rAccount3"]' > test_accounts.json

# Run data collection
python src/query_data.py

# Output: Individual JSON files in data/accounts/ directory
```

### Data Structure

Each account file contains:
```json
{
  "account_info": {...},      // Basic account information
  "transactions": [...],      // Transaction history (up to 375 transactions)
  "assets": [...],           // Token holdings with valuations  
  "trustlines": {...}        // Credit line relationships
}
```

## 🎯 Model Training

The `ml_model.py` script implements the complete training pipeline:

### Training Process

1. **Data Loading**: Loads complete feature dataset from `complete_features.csv`
2. **Feature Selection**: Selects 28 key features using correlation analysis + heuristic features
3. **Preprocessing**: Median imputation and standardization
4. **Anomaly Detection**: Isolation Forest identifies 2% outliers
5. **Core Clustering**: GMM with 5 components clusters normal accounts
6. **PCA Scoring**: Generates continuous risk scores for all accounts
7. **Risk Labeling**: Maps clusters to business-friendly risk categories
8. **Artifact Export**: Saves all models for production deployment

### Key Configuration

```python
MANUAL_K = 5                    # Number of GMM components
contamination = 0.02            # Isolation Forest outlier rate
correlation_threshold = 0.3     # Feature selection threshold
covariance_type = 'full'        # GMM covariance structure
```

### Output Artifacts

Training generates these artifacts in `artifacts/` directory:
- `scaler.joblib` - Feature standardization parameters
- `feature_cols.joblib` - Selected feature names and order
- `isolation_forest.joblib` - Trained anomaly detection model  
- `gmm.joblib` - Trained clustering model
- `pca.joblib` - Trained dimensionality reduction model
- `cluster_risk_mapping.joblib` - Cluster-to-label mappings
- `pca_scaling_params.joblib` - PCA score normalization parameters

## ✅ Model Validation

The `model_validation.py` script provides comprehensive model validation:

### Validation Framework

1. **Risk Distribution Validation**: Ensures risk scores increase monotonically across categories
2. **Cluster Confidence Analysis**: Validates GMM prediction confidence (target >90%)
3. **Feature Correlation Analysis**: Validates expected relationships between features and risk
4. **GMM Quality Metrics**: BIC, AIC, and Calinski-Harabasz scores
5. **Model Consistency**: Correlation between continuous scores and discrete categories

### Validation Outputs

- **Statistical Reports**: Text summary of validation results
- **Visualization Suite**: Risk distribution plots, confidence analysis, feature correlation heatmaps
- **Quality Metrics**: Clustering performance indicators
- **Pass/Fail Assessment**: Clear validation status for production readiness

### Running Validation

```bash
python src/model_validation.py

# Output: Validation reports and plots in output/validation/
```

## 🎯 Prediction

### Standalone Prediction Script

```bash
# Direct command-line prediction
python src/predict_score.py rN7n7otQDd6FczFgLdSqtcsAUxDkw6fzRH

# Output: Console report with risk score and label
```

### Prediction Pipeline

1. **Live Data Collection**: Fetches real-time account data from XRPScan API
2. **Feature Generation**: Computes all 28 model features
3. **Data Preprocessing**: Applies same scaling as training data
4. **Anomaly Detection**: Checks for outlier classification
5. **Cluster Assignment**: GMM prediction for normal accounts
6. **PCA Scoring**: Continuous risk score generation
7. **Risk Labeling**: Final business-friendly risk category

## 🚢 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t occr-model .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts -v $(pwd):/app/all_tokens.json occr-model
```
### Railway Deployment

This microservice is deployed on Railway at the following
```bash
https://occr-model-production.up.railway.app
```
### Production Configuration

#### Environment Variables
```bash
export XRPL_NETWORK=mainnet     # or testnet
export API_PORT=8000
export LOG_LEVEL=INFO
export ARTIFACTS_PATH=./artifacts
export TOKEN_DB_PATH=./all_tokens.json
```

#### Rate Limiting & Performance
- **API Request Rate**: 1 request per second to XRPL nodes
- **Transaction History**: Limited to 15 pages (375 transactions) for performance
- **Response Time**: <3 seconds average per account scoring
- **Concurrent Requests**: Handled via FastAPI async support

#### Monitoring & Maintenance
- **Health Checks**: Built-in endpoint for service monitoring
- **Error Handling**: Comprehensive exception handling with informative error messages  
- **Logging**: Structured logging for debugging and monitoring
- **Model Versioning**: Built-in version tracking in API responses

## 📁 Project Structure

```
occr-model/
├── src/
│   ├── api.py                 # FastAPI web service (production endpoint)
│   ├── query_data.py          # XRPL data collection from XRPScan API  
│   ├── ml_features.py         # Comprehensive feature engineering (65 features)
│   ├── ml_model.py            # GMM-based model training pipeline
│   ├── model_validation.py    # Statistical validation and testing suite
│   └── predict_score.py       # Standalone prediction script
├── data/
│   └── accounts/              # Individual account JSON files
├── output/
│   ├── complete_features.csv  # Full feature dataset
│   ├── results/
│   │   └── risk_profiles.csv  # Trained model results
│   └── validation/            # Validation reports and plots
├── artifacts/                 # Trained model artifacts (joblib files)
├── all_tokens.json           # Token quality database (1000+ tokens)
├── test_accounts.json        # Input file for data collection
└── requirements.txt          # Python dependencies
```

### Key Files

- **`api.py`**: Production FastAPI service with dual XRPL node/XRPScan API support
- **`ml_features.py`**: Comprehensive feature engineering with 65 features across 7 dimensions
- **`ml_model.py`**: Advanced ML pipeline with Isolation Forest + GMM + PCA
- **`model_validation.py`**: Statistical validation suite with visualization
- **`predict_score.py`**: Lightweight prediction script for single accounts
- **`query_data.py`**: Rate-limited data collection from XRPScan API

## 🔬 Model Performance

### Validation Metrics

Based on comprehensive validation suite:

- **Risk Monotonicity**: ✅ Risk scores increase monotonically across categories
- **Cluster Confidence**: 90%+ average confidence in cluster assignments
- **Feature Alignment**: Expected correlations validated for key risk indicators
- **Model Consistency**: >80% correlation between continuous scores and discrete categories
- **Clustering Quality**: Optimized BIC/AIC scores for 5-component GMM

### Business Metrics

- **Processing Speed**: <3 seconds per account (including API calls)
- **Feature Coverage**: 65 comprehensive features across all risk dimensions
- **Account Coverage**: Handles 99%+ of active XRPL accounts
- **Score Stability**: Consistent scoring across time periods
- **Risk Discrimination**: Clear separation between risk categories

### Technical Specifications

- **Model Type**: Ensemble (Isolation Forest + GMM + PCA)
- **Feature Count**: 65 features across 7 dimensions
- **Training Data**: Scalable to 1000+ accounts
- **Prediction Latency**: <3 seconds including API calls
- **Memory Usage**: <500MB for full model artifacts
- **API Throughput**: 20+ requests/minute (rate-limited by XRPL APIs)

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone https://github.com/siyasiyasiya/occr-model.git
cd occr-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run validation suite
python src/model_validation.py

# Test API endpoint
uvicorn src.api:app --reload
```

### Development Workflow

1. **Data Collection**: Update `test_accounts.json` and run `query_data.py`
2. **Feature Engineering**: Modify `ml_features.py` for new features
3. **Model Training**: Update `ml_model.py` and retrain with `python src/ml_model.py`
4. **Validation**: Run `model_validation.py` to verify model performance
5. **API Testing**: Test endpoint with `predict_score.py` or direct API calls

### Code Quality Standards

- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Robust exception handling with informative messages
- **Type Hints**: Python type hints for better code clarity
- **Performance**: Optimized for production deployment
- **Modularity**: Clean separation between data, features, model, and API

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- **XRPL Documentation**: https://xrpl.org/docs.html
- **XRPScan API**: https://xrpscan.com/api
- **XRPL-Py Library**: https://github.com/XRPLF/xrpl-py  
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Scikit-learn**: https://scikit-learn.org/

*A comprehensive on-chain credit risk assessment system combining advanced machine learning with deep blockchain analytics.*
