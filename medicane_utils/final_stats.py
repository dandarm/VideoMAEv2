from typing import Optional, Union  
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss
)

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,   # 🠈 qui
    pos_label: Union[int, str] = 1,         # 🠈 e qui
    show_report: bool = True
) -> pd.Series:
    """
    Restituisce un pandas Series con le metriche principali per la classificazione binaria.
    Se passi anche y_score, calcola ROC-AUC, PR-AUC, log-loss e Brier score.
    """
    # Controlli veloci
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_score is not None:
        y_score = np.asarray(y_score)

    # Metriche base
    metrics = {
        "accuracy"              : accuracy_score(y_true, y_pred),
        "balanced_accuracy"     : balanced_accuracy_score(y_true, y_pred),
        "precision"             : precision_score(y_true, y_pred, pos_label=pos_label),
        "recall"                : recall_score(y_true, y_pred, pos_label=pos_label),
        "f1"                    : f1_score(y_true, y_pred, pos_label=pos_label),
        "matthews_corrcoef"     : matthews_corrcoef(y_true, y_pred),
        "cohen_kappa"           : cohen_kappa_score(y_true, y_pred),
    }

    # Metriche che richiedono punteggi/ probabilità
    if y_score is not None:
        metrics.update({
            "roc_auc"           : roc_auc_score(y_true, y_score),
            "pr_auc"            : average_precision_score(y_true, y_score),
            "log_loss"          : log_loss(y_true, y_score, labels=[0,1]),
            "brier_score_loss"  : brier_score_loss(y_true, y_score),
        })

    # Confusion matrix “flattened”
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, pos_label]).ravel()
    metrics.update({
        "true_negatives" : tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives" : tp,
    })

    if show_report:
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_true, y_pred, labels=[0, pos_label]))
        print(f"True Positive: {tp}")
        print(f"True Negative: {tn}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, labels=[0, pos_label], target_names=["neg", "pos"]))

    return pd.Series(metrics, name="metrics")

# ESEMPIO D’USO -----------------------------------------------------------
# y_true  = np.random.randint(0, 2, size=100)
# y_pred  = np.random.randint(0, 2, size=100)
# y_score = np.random.rand(100)
# results = evaluate_binary_classifier(y_true, y_pred, y_score)
# print("\n=== Metriche ===")
# print(results)
