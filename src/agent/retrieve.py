import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global model cache for performance in the same run
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL

def retrieve_top_k(customer_message, k=3):
    """
    Retrieves the top k historical resolution pairs from the 8000-pair sample.
    Caches the embeddings to disk to ensure pipeline runs extremely fast.
    """
    data_path = Path("data/processed/apple_support_sample.csv")
    embeddings_path = Path("data/processed/apple_support_sample_embeddings.npy")
    
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")
        
    df = pd.read_csv(data_path)
    
    model = get_model()
    
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        print("Building retrieval index for the first time... (this will be cached)")
        texts = df['customer_text'].tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        np.save(embeddings_path, embeddings)
        
    query_embedding = model.encode([customer_message])
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "customer_text": row["customer_text"],
            "reply_text": row["reply_text"],
            "similarity": float(similarities[idx])
        })
        
    return results
