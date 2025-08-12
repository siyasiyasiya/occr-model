# OCCR Model

OCCR Model is a Python-based framework for extracting, analyzing, and scoring risk and behavior features from blockchain wallet data, with a focus on the XRP Ledger ecosystem. It is well-suited for account profiling, risk analysis, and wallet clustering.

## Table of Contents

- Project Overview
- Features
- Repository Structure
- Installation
- Usage
- Model Outputs
- API
- Requirements
- Contributing

---

## Project Overview

The project is built to:
- Collect and process account data from the XRP Ledger.
- Engineer comprehensive risk and behavioral features per wallet.
- Train unsupervised models (Isolation Forest, GMM, PCA) for risk scoring and clustering.
- Provide a REST API for live wallet score predictions.

---

## Features

- Data Collection: Automated scripts pull account, transaction, asset, and trustline data.
- Feature Engineering: Modular extraction of account fundamentals, transaction patterns, financial flows, asset analytics, portfolio risks, trustline analysis, and risk indicators.
- Unsupervised Modeling: Uses Isolation Forest for outlier detection, Gaussian Mixture Model for clustering, and PCA for continuous scoring.
- Modular Outputs: Saves results in logically grouped CSVs for analysis.
- Live Prediction API: FastAPI-based endpoint for live wallet scoring.
- Robust Error Handling: Logs failed data extractions and processing errors.

---

## Repository Structure

```
occr-model/
│
├── src/
│   ├── api.py          # REST API for live predictions
│   ├── ml_features.py  # Feature extraction and engineering
│   ├── ml_model.py     # Model training and artifact generation
│   ├── predict_score.py# Live scoring logic
│   ├── query_data.py   # Data collection from the XRPSCAN API
│
├── artifacts/          # Saved model artifacts (.joblib files)
├── output/             # Output CSVs and summary files
├── all_tokens.json     # Token metadata for feature engineering
├── requirements.txt    # Python dependencies
├── .gitignore          # Standard gitignore
```

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/siyasiyasiya/occr-model.git
    cd occr-model
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Data Collection
Run data collection to generate wallet JSON files:
```bash
python src/query_data.py
```
This will fetch account info, transactions, assets, and trustlines for specified accounts.

### 2. Feature Extraction
Extract features from account files:
```bash
python src/ml_features.py
```
Generates modular feature CSV files in `output/`.

### 3. Model Training
Train models and export artifacts:
```bash
python src/ml_model.py
```
Saves all relevant model files to `artifacts/`.

### 4. Live Score Prediction
Run the API server for live scoring:
```bash
uvicorn src.api:app --reload
```
Query scores via:
```
GET /score?address=<wallet_address>
```

### 5. Predict Score from Script
Use `predict_score.py` to fetch live data and predict scores for an address:
```bash
python src/predict_score.py --address <wallet_address>
```

---

## Model Outputs

After running feature extraction and modeling, the following files are created in `output/`:

- `account_summary.csv`: Quick analysis summary
- `account_fundamentals.csv`: Basic account info
- `transaction_patterns.csv`: Transaction behavior
- `financial_flows.csv`: Money flow analysis
- `asset_analytics.csv`: Token holdings analysis
- `portfolio_risk.csv`: Portfolio metrics
- `trustline_analysis.csv`: Trustline data
- `risk_indicators.csv`: Risk scores
- `complete_features.csv`: All features combined
- `feature_summary.txt`: Processing summary

Saved model artifacts in `artifacts/` include:
- scaler, feature_cols, isolation_forest, gmm, pca, cluster_risk_mapping, pca_scaling_params

---

## API

The FastAPI server exposes endpoints for:
- `/score?address=<wallet_address>`: Returns the risk score and cluster for the specified wallet.
- `/`: Health check endpoint.

---

## Requirements

- Python 3.7+
- See `requirements.txt` for all Python dependencies.
- Internet access for live data/API calls.

---

## Contributing

Open issues or pull requests to report bugs, propose features, or discuss improvements.

---

## License

Please contact the repository owner for license details.

---

## Contact

Owner: [siyasiyasiya](https://github.com/siyasiyasiya)

---