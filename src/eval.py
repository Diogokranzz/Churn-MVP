from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def evaluate(y_true, y_pred_probs, threshold=0.5):
    auc = roc_auc_score(y_true, y_pred_probs)
    preds = (y_pred_probs >= threshold).astype(int)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}
