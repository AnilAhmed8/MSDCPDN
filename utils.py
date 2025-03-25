from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(y_true, y_pred):
    """
    Computes evaluation metrics.
    Returns: confusion matrix, precision, recall, F1, accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return cm, precision, recall, f1, accuracy

def print_metrics(cm, precision, recall, f1, accuracy):
    """
    Prints evaluation metrics.
    """
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
