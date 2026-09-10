from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

def calculate_escalation_metrics(y_true, y_pred):
    """
    Calculates precision, recall, F1, and confusion matrix for escalation logic.
    We treat 'escalate' as the positive class, and 'auto' as the negative class.
    """
    labels = ["auto", "escalate"]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    return {
        "metrics": per_class,
        "confusion_matrix": cm_df
    }
