import json
import random

input_file = 'trustlines.json'  

sample_size = 500  

with open(input_file, 'r') as f:
    data = json.load(f)

all_accounts = [line['account'] for line in data['lines']]

if len(all_accounts) < sample_size:
    print(f"Error: The file contains only {len(all_accounts)} accounts, which is less than the requested sample size of {sample_size}.")
else:
    selected_accounts = random.sample(all_accounts, sample_size)
    
    output_file = 'test_accounts.json'
    with open(output_file, 'w') as f:
        json.dump(selected_accounts, f, indent=4)
        
    print(f"Successfully selected {sample_size} unique accounts and saved them to {output_file}")