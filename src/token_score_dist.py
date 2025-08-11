import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('../data/all_tokens.json')

# Basic statistics
print(df['score'].describe())

# Plot histogram
plt.hist(df['score'], bins=50)
plt.xlabel('Token Score')
plt.ylabel('Count')
plt.title('Distribution of Token Scores')
plt.show()