from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

def calculate_intent_metrics(y_true, y_pred, labels):
    """
    Calculates accuracy, macro-F1, per-intent metrics, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_intent = {}
    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_intent": per_intent,
        "confusion_matrix": cm_df
    }
