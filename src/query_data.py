import requests
import json
import os
import time

# --- Configuration ---
BASE_URL = "https://api.xrpscan.com/api/v1"
INPUT_FILE = "test_accounts.json"  # Input file is now a JSON list of accounts
ACCOUNTS_DIR = "data/accounts"

# --- Create directories if they don't exist ---
os.makedirs(ACCOUNTS_DIR, exist_ok=True)
TARGET_TIME_PER_REQUEST = 60.0 / 58.0 

# --- Set to keep track of tokens we've already fetched ---
processed_tokens = set()

def api_request(endpoint, params=None):
    """
    Makes a request to the API and then waits just long enough
    to maintain a steady, safe request rate.
    """
    start_time = time.time()
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        json_response = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"    - HTTP Error for endpoint {endpoint}: {e}")
        if e.response.status_code == 429:
            print("    - Hit rate limit. Waiting for 15 seconds before retrying...")
            time.sleep(15) # Longer penalty wait
            return api_request(endpoint, params)
    except requests.exceptions.RequestException as e:
        print(f"    - Request Error for endpoint {endpoint}: {e}")
        return None

    # Calculate how long the request took and how long we need to wait
    duration = time.time() - start_time
    wait_time = TARGET_TIME_PER_REQUEST - duration
    
    if wait_time > 0:
        time.sleep(wait_time)
        
    return json_response

def get_full_transaction_history(account):
    """Fetches transaction history with pagination."""
    all_transactions = []
    marker = None
    page = 1

    while page <= 15:
        print(f"    - Fetching transactions page {page}...")
        params = {'limit': 25, 'marker': marker} if marker else {'limit': 25}
        data = api_request(f"/account/{account}/transactions", params)
        if not data or 'transactions' not in data or not data['transactions']:
            break
        all_transactions.extend(data['transactions'])
        if 'marker' not in data:
            break
        marker = data['marker']
        page += 1
    return all_transactions

def process_account(account):
    """Fetches all data for a single account and saves it to a file."""
    account_file = os.path.join(ACCOUNTS_DIR, f"{account}.json")
    if os.path.exists(account_file):
        print(f"Skipping {account} - data already exists.")
        return

    print(f"Processing account: {account}")

    # 1. Get Account Info
    print("  - Fetching Account Info...")
    account_info = api_request(f"/account/{account}")

    # 2. Get Full Transaction History
    transactions = get_full_transaction_history(account)

    # 3. Get Assets (Balances)
    print("  - Fetching Assets...")
    assets = api_request(f"/account/{account}/assets")
    time.sleep(1.5)

    # 4. Get Trustlines
    print("  - Fetching Trustlines...")
    trustlines = api_request(f"/account/{account}/trustlines2")

    # Combine all data into a single dictionary
    final_account_data = {
        "account_info": account_info,
        "transactions": transactions,
        "assets": assets,
        "trustlines": trustlines
    }

    # Save the combined data to the account's JSON file
    with open(account_file, 'w') as f:
        json.dump(final_account_data, f, indent=4)
    print(f"  - Successfully saved data for {account}")

if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r') as f:
            # Load the list of accounts from the JSON file
            accounts = json.load(f)
        
        print(f"Found {len(accounts)} accounts to process from '{INPUT_FILE}'.")
        
        for i, account_address in enumerate(accounts):
            print(f"\n--- Starting account {i+1}/{len(accounts)} ---")
            process_account(account_address)

        print("\n\nData collection complete!")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Please create it and add the account addresses.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_FILE}'. Please ensure it is a valid JSON array of strings.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")