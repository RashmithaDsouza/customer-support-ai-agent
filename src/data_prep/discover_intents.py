import sys
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import warnings

# Suppress sklearn warnings about memory leak on Windows with KMeans
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure stdout uses UTF-8 to avoid UnicodeEncodeError when redirecting output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    input_path = Path("data/processed/apple_support_sample.csv")
    output_path = Path("data/processed/intent_clusters.csv")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Sampling 200 customer messages...")
    # Randomly sample 200 messages
    sample_df = df.sample(n=200, random_state=42).copy()
    
    texts = sample_df['customer_text'].tolist()
    
    print("Generating sentence embeddings with all-MiniLM-L6-v2... (this may take a moment)")
    # Use a lightweight model suitable for local CPU execution
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Clustering embeddings using KMeans (8 clusters)...")
    # Cluster into 8 groups
    kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(embeddings)
    
    sample_df['cluster'] = clusters
    
    print("\n--- Cluster Samples ---")
    for cluster_id in range(8):
        print(f"\nCluster {cluster_id}:")
        cluster_texts = sample_df[sample_df['cluster'] == cluster_id]['customer_text'].head(10)
        for i, text in enumerate(cluster_texts, 1):
            print(f"  {i}. {text}")
            
    print(f"\nSaving clustered results to {output_path}...")
    
    # Prepare output dataframe with specified columns
    output_df = pd.DataFrame({
        'sample_index': sample_df.index,
        'cluster': sample_df['cluster'],
        'customer_text': sample_df['customer_text']
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"Done! Saved {len(output_df)} rows to {output_path.absolute()}")

if __name__ == "__main__":
    main()
