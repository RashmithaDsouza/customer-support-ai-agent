import pandas as pd
import re
import html
from pathlib import Path

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Normalize excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    raw_data_path = Path("data/raw/twcs.csv")
    output_path = Path("data/processed/apple_support_pairs.csv")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    
    print("Filtering AppleSupport rows...")
    apple_replies = df[df["author_id"] == "AppleSupport"].copy()
    total_apple_replies = len(apple_replies)
    
    print("Matching replies to customer tweets...")
    # Filter customer inbound messages
    customer_messages = df[df["inbound"] == True].copy()
    
    # Merge to find parent tweet that is a customer message
    merged = pd.merge(
        apple_replies,
        customer_messages,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_reply', '_customer')
    )
    
    # Calculate unmatched replies based on unique matched replies
    unique_matched_replies = merged['tweet_id_reply'].nunique()
    unmatched_replies = total_apple_replies - unique_matched_replies
    
    # Create required customer-reply pairs format
    pairs_df = pd.DataFrame({
        'customer_tweet_id': merged['tweet_id_customer'],
        'customer_author_id': merged['author_id_customer'],
        'customer_text': merged['text_customer'],
        'reply_tweet_id': merged['tweet_id_reply'],
        'reply_text': merged['text_reply'],
        'created_at': merged['created_at_reply']
    })
    
    print("Cleaning text...")
    pairs_df['customer_text'] = pairs_df['customer_text'].apply(clean_text)
    pairs_df['reply_text'] = pairs_df['reply_text'].apply(clean_text)
    
    print("Removing rows with empty text...")
    pairs_df = pairs_df[(pairs_df['customer_text'] != "") & (pairs_df['reply_text'] != "")]
    matched_pairs = len(pairs_df)
    
    print(f"Saving results to {output_path}...")
    pairs_df.to_csv(output_path, index=False)
    
    # Print requested stats
    print("\n--- Summary ---")
    print(f"Total AppleSupport replies: {total_apple_replies}")
    print(f"Matched customer-reply pairs: {matched_pairs}")
    print(f"Unmatched replies: {unmatched_replies}")
    print(f"Output file path: {output_path.absolute()}")

    # Subsampling Step
    sample_output_path = Path("data/processed/apple_support_sample.csv")
    print("\nSampling 8,000 pairs...")
    sampled_df = pairs_df.sample(n=8000, random_state=42)
    
    print(f"Saving sampled dataset to {sample_output_path}...")
    sampled_df.to_csv(sample_output_path, index=False)
    
    print("\n--- Subsampling Summary ---")
    print(f"Original pair count: {matched_pairs}")
    print(f"Sampled pair count: {len(sampled_df)}")
    print(f"Random seed: 42")
    print(f"Sampled output path: {sample_output_path.absolute()}")

if __name__ == "__main__":
    main()
