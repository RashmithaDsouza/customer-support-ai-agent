import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Print columns
print("Dataset columns:")
print(df.columns.tolist())

# 3. Print shape
print("\nDataset shape:")
print(df.shape)

# 4. Print first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 5. Print top 50 author_id counts
print("\nTop 50 author_id counts:")
print(df["author_id"].value_counts().head(50))

# 6. Filter for AppleSupport
apple = df[df["author_id"] == "AppleSupport"]

# 7. Print AppleSupport row count
print("\nAppleSupport row count:")
print(len(apple))

# 8. Print AppleSupport inbound distribution
print("\nAppleSupport inbound distribution:")
print(apple["inbound"].value_counts())

# 9. Print first 10 AppleSupport replies
print("\nFirst 10 AppleSupport replies:")
apple_replies = apple[apple["inbound"] == False]
print(apple_replies[["tweet_id", "text", "in_response_to_tweet_id"]].head(10))

# 10. Print first 10 customer inbound messages
print("\nFirst 10 customer inbound messages:")
inbound_msgs = df[df["inbound"] == True]
print(inbound_msgs[["tweet_id", "author_id", "text", "response_tweet_id"]].head(10))