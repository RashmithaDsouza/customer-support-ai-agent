from sklearn.metrics import cohen_kappa_score
import pandas as pd

def calculate_agreement(human_scores, llm_scores):
    """
    Calculates weighted Cohen's kappa for ordinal 1-5 scores.
    Expects lists or arrays of numerical scores.
    """
    # Filter out missing/invalid scores and cast to int to prevent ValueError in cohen_kappa_score
    valid_pairs = []
    for h, l in zip(human_scores, llm_scores):
        if pd.notnull(h) and pd.notnull(l):
            valid_pairs.append((int(round(float(h))), int(round(float(l)))))
            
    if not valid_pairs:
        return {"kappa": 0.0, "interpretation": "No valid data.", "n_samples": 0}
        
    h_clean, l_clean = zip(*valid_pairs)
    kappa = cohen_kappa_score(h_clean, l_clean, weights="quadratic")
    
    if kappa < 0: interp = "Poor agreement"
    elif kappa < 0.20: interp = "Slight agreement"
    elif kappa < 0.40: interp = "Fair agreement"
    elif kappa < 0.60: interp = "Moderate agreement"
    elif kappa < 0.80: interp = "Substantial agreement"
    else: interp = "Almost perfect agreement"
    
    return {
        "kappa": float(kappa),
        "interpretation": interp,
        "n_samples": len(h_clean)
    }
