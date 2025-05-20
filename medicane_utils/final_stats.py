from typing import Optional, Union  
import numpy as np
import pandas as pd
from typing import List, Union
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
import re
import matplotlib.pyplot as plt
from typing import Sequence

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





# ------------------------------------------------------------
# 1. funzione d'appoggio: estrae hh_mm in minuti dall'inizio del giorno
# ------------------------------------------------------------
def label_to_minutes(label: str) -> Optional[int]:   # 👈 cambio qui
    """
    Estrae 'hh_mm' dal nome label e lo converte in minuti dopo mezzanotte.
    Esempio: 'label_03_20' → 200  (3*60 + 20).
    Ritorna None se il pattern non viene trovato.
    """
    m = re.search(r'(\d{2})_(\d{2})$', label)
    if m:
        h, mnt = map(int, m.groups())
        return h * 60 + mnt

    return 0



d = {"tn":"True Negatives", "fp": "False Positives", "fn": "False Negatives", "tp":"True Positives"}
# ------------------------------------------------------------
# 2. plotting function
# ------------------------------------------------------------
def plot_metrics_over_time(
    conf_df: pd.DataFrame,
    metrics: Sequence[str] = ("tn", "fp", "fn", "tp"),
    sum_labels: List = [],
    xlabel: str = "Time shift (hh:mm)",
    ylabel: str = "Count",
    title: str = "Confusion-matrix counts over time",
) -> None:
    """
    Plotta i valori di `metrics` (colonne di conf_df) sull'asse x ordinato per ora.
    conf_df deve avere l'indice = label e colonne con le metriche.
    """
    # aggiunge colonna temporale e ordina
    times = conf_df.index.to_series().apply(label_to_minutes)
    plot_df = conf_df.copy()
    plot_df["time_min"] = times
    plot_df = plot_df.dropna(subset=["time_min"]).sort_values("time_min")

    # prepara x (etichette hh:mm) e y
    x_ticks = [
        f"{int(t // 60):02d}:{int(t % 60):02d}" for t in plot_df["time_min"]
    ]
    x = plot_df["time_min"].values  

    # un'unica figura con N linee (niente subplots)
    plt.figure(figsize=(10, 5))
    for m in metrics:
        if m in plot_df.columns:
            plt.plot(x, plot_df[m].values, marker="o", label=d[m])

    plt.plot(x, sum_labels, label="Total positives", marker='.')

    plt.xticks(x, x_ticks, rotation=75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# USO
# ------------------------------------------------------------
# conf_df = confusion_counts_per_label(df, pred_col="y_pred", label_cols=label_cols)
# plot_metrics_over_time(conf_df, metrics=("tp", "fp", "fn", "tn"))
