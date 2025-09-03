import os
from PIL import Image
from typing import List, Union, Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
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
    log_loss,
)

import torch
import torchvision.transforms as transforms
#import torchvision.transforms.functional as F
import torch.nn.functional as F



from utils import NativeScalerWithGradNormCount as NativeScaler

from dataset import build_pretraining_dataset
from torch.utils.data import DataLoader
from utils import get_model
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato



# Funzione per caricare le immagini
def load_images(image_folder, transform, device):
    images = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        images.append(image)
    return torch.cat(images)


def load_frame_sequence(frame_dir, transform, num_frames=16, device="cuda"):
    """
    Carica una sequenza di frame da una directory e li preprocessa.
    Args:
        frame_dir (str): Percorso della directory contenente i frame.
        transform (callable): Trasformazione da applicare ai frame.
        num_frames (int): Numero di frame da includere nella sequenza.
        device (str): Dispositivo su cui caricare i tensori.
    Returns:
        Tensor: Tensore della sequenza con dimensioni (1, num_frames, C, H, W).
    """
    frame_paths = sorted([os.path.join(frame_dir, fname) for fname in os.listdir(frame_dir)])

    # Assicurati di avere abbastanza frame
    if len(frame_paths) < num_frames:
        raise ValueError(f"Non ci sono abbastanza frame in {frame_dir}. Trovati: {len(frame_paths)}")

    # Seleziona i primi `num_frames`
    selected_frames = frame_paths[:num_frames]

    # Carica e trasforma i frame
    frames = []
    for frame_path in selected_frames:
        img = Image.open(frame_path).convert("RGB")
        frames.append(transform(img).unsqueeze(0))  # Aggiungi dimensione batch
    frames = torch.cat(frames, dim=0)  # Combina i frame lungo il batch

    # Aggiungi dimensione per batch e sposta sul dispositivo
    return frames.unsqueeze(0).to(device)  # (1, num_frames, C, H, W)



# Funzione per calcolare l'errore di ricostruzione (MSE e PSNR)
def reconstruction_metrics(original, reconstructed):
    mse = F.mse_loss(reconstructed.logit(), original).item()
    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
    return mse, psnr

# Configurazione dei parametri
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#pretrained_model_path = './pytorch_model.bin'  # Modello preaddestrato
#specialized_model_path = './output_old2/checkpoint-149.pth'  # Modello addestrato
#image_folder = './sequenced_imgs/freq-1.6_part3'  # Cartella con le immagini di test




#region Funzioni per eseguire l'inferenza

#####  USARE   validation_one_epoch_collect from engine_or_finetuning
# val_stats, all_paths, all_preds, all_labels = validation_one_epoch_collect(val_m.data_loader, pretrained_model, device)
#########################################
# def predict_label(model, videos):    
#     #with torch.no_grad():
#     with torch.cuda.amp.autocast():
#         logits = model(videos)  # (B, nb_classes)
#         predicted_classes = torch.argmax(logits, dim=1)  # intero con l'indice di classe
    
#     return predicted_classes

# def get_path_pred_label(model, data_loader):
#     all_paths = []
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for videos, labels, folder_path in data_loader:
#             videos = videos.to(device, non_blocking=True)
#             predicted_classes = predict_label(model, videos) # shape (batch, num_class)
#             labels = labels.detach().cpu().numpy()
#             pred_classes = predicted_classes.detach().cpu().numpy()
            
#             all_labels.extend(labels)
#             all_preds.extend(pred_classes)
#             all_paths.extend(folder_path)

#     return all_paths, all_preds, all_labels

def create_df_predictions(all_paths, all_preds, all_labels):
    video_folder_name = pd.Series(all_paths).str.split('/').str.get(-1)
    predictions_series = pd.Series(all_preds)
    labels_series = pd.Series(all_labels)
    res_df = pd.concat([video_folder_name, predictions_series, labels_series], axis=1)
    res_df.columns = ['path', 'predictions', 'labels']

    return res_df


# se non mi servono le predizioni uso questo:
def get_only_labels(data_loader):    
    all_labels = []
    all_paths = []
    for videos, labels, folder_path in data_loader:
        labels = labels.detach().cpu().numpy()    
        all_labels.extend(labels)
        all_paths.extend(folder_path)
    
    all_preds = [0] * len(all_labels)
    return all_paths, all_preds, all_labels



def video_pred_2_img_pred(df_video_w_predictions):
    # Espando dataframe con le associazioni (path X offsets) -> predictions
    records = []
    for _, row in df_video_w_predictions.iterrows():
        for orig_path in row['orig_paths']:
            records.append({
                'path': orig_path,
                'predictions': row['predictions'],
                'tmp_label': row['labels'],
                'tile_offset_x': row['tile_offset_x'],
                'tile_offset_y': row['tile_offset_y']
            })

    # Li trasformiamo in un nuovo DataFrame
    df_mapping = pd.DataFrame(records)

    #df_mapping[['path', 'tile_offset_x', 'tile_offset_y']].duplicated().sum()
    # se ci sono path duplicati in combinazione con gli offsets
    return df_mapping

#endregion



# region Funzioni per calcolare le metriche di classificazione

def confusion_counts_per_label(
    df: pd.DataFrame,
    pred_col: str,
    label_cols: List[str],
    pos_label: Union[int, str] = 1
) -> pd.DataFrame:
    """
    Per ogni colonna in `label_cols` calcola la confusion-matrix contro `pred_col`
    (supposta già binaria) e restituisce un DataFrame indicizzato per label
    con le quattro celle tn, fp, fn, tp.

    Parameters
    ----------
    df : pd.DataFrame
        Il dataframe che contiene sia le predizioni che le etichette.
    pred_col : str
        Nome della colonna con le predizioni 0/1.
    label_cols : list[str]
        Elenco delle colonne-etichetta da confrontare.
    pos_label : int | str, default=1
        Valore considerato “positivo” (serve solo per stabilire l’ordine
        [0, pos_label] nella confusion matrix).

    Returns
    -------
    pd.DataFrame
        Indice = nome label; colonne = tn, fp, fn, tp.
    """
    records = []
    for lab in label_cols:
        tn, fp, fn, tp = confusion_matrix(
            df[lab].values,
            df[pred_col].values,
            labels=[0, pos_label]
        ).ravel()
        records.append(
            {"label": lab, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        )

    return pd.DataFrame(records).set_index("label")


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    pos_label: Union[int, str] = 1,
    show_report: bool = False,
) -> pd.Series:
    """Calcola un insieme esteso di metriche per la classificazione binaria:

    - Balanced Accuracy
    - Recall = POD
    - False Alarm Ratio  FAR
    - CSI
    - HSS

    Se `y_score` è fornito vengono calcolate anche le metriche che richiedono
    probabilità o punteggi di confidenza (ROC-AUC, PR-AUC, log-loss e Brier score).
    Restituisce un ``pandas.Series`` contenente tutte le metriche.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_score is not None:
        y_score = np.asarray(y_score)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        #"precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        #"f1": f1_score(y_true, y_pred, pos_label=pos_label),
        #"matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        #"cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_score is not None:
        metrics.update({
            "roc_auc": roc_auc_score(y_true, y_score),
            "pr_auc": average_precision_score(y_true, y_score),
            "log_loss": log_loss(y_true, y_score, labels=[0, 1]),
            "brier_score_loss": brier_score_loss(y_true, y_score),
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, pos_label]).ravel()
    #fpr = fp / (fp + tn) if (fp + tn) else 0
    #fnr = fn / (fn + tp) if (fn + tp) else 0
    far = fp / (fp + tp) if (fp + tp) else 0
    #pod = tp / (tp + fn) if (tp + fn) else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    hss_den = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fp * fn) / hss_den) if hss_den else 0

    metrics.update({
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        #"false_positive_rate": fpr,
        #"false_negative_rate": fnr,
        "far": far,
        #"pod": pod,
        "csi": csi,
        "hss": hss,
    })

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred, labels=[0, pos_label]))
    if show_report:        
        print(f"True Positive: {tp}")
        print(f"True Negative: {tn}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, labels=[0, pos_label], target_names=["neg", "pos"]))

    return pd.Series(metrics, name="metrics")


def plot_confusion_and_results(
    results: Union[pd.Series, Mapping[str, float]],
    labels: Sequence[str] = ("Negative", "Positive"),
    title: str = "Confusion matrix and metrics",
    savepath: Optional[str] = None,
    cmap: str = "Blues",
    normalize: bool = True,
) -> None:
    """Plotta una matrice di confusione con annotazioni e, a lato, un riquadro
    con le metriche principali contenute in ``results``.

    Parametri
    ---------
    results: pandas.Series o dict
        Output di ``evaluate_binary_classifier`` che include almeno
        ``true_negatives``, ``false_positives``, ``false_negatives``, ``true_positives``
        e le metriche scalari (es. ``accuracy``, ``balanced_accuracy``, ``recall``, ...).
    labels: sequence of str
        Nomi mostrati sugli assi [true, pred] in ordine (neg, pos).
    title: str
        Titolo della figura.
    savepath: str or None
        Se fornito, salva la figura in questo percorso (dpi=300, tight layout).
    cmap: str
        Colormap per la heatmap.
    normalize: bool
        Se True usa la matrice normalizzata per il colore; in ogni caso annota
        sia il conteggio che la percentuale.
    """

    # Supporta sia Series che dict
    tn = int(results.get("true_negatives", results.get("tn", 0)))
    fp = int(results.get("false_positives", results.get("fp", 0)))
    fn = int(results.get("false_negatives", results.get("fn", 0)))
    tp = int(results.get("true_positives", results.get("tp", 0)))

    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    # Normalizzazione per riga (per true class)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(counts, row_sums, where=row_sums != 0)
        cm_norm[np.isnan(cm_norm)] = 0.0

    # Setup figura con due pannelli
    fig, (ax_cm, ax_txt) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    data_for_color = cm_norm if normalize else counts
    im = ax_cm.imshow(data_for_color, interpolation="nearest", cmap=cmap,
                      vmin=0.0, vmax=(1.0 if normalize else data_for_color.max() or 1.0))

    # Assi e ticks
    ax_cm.set_title(title)
    ax_cm.set_ylabel("True label")
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_xticks([0, 1], labels=labels)
    ax_cm.set_yticks([0, 1], labels=labels)

    # Annotazioni: conteggio + percentuale per cella
    for i in range(2):
        for j in range(2):
            count = int(counts[i, j])
            pct = cm_norm[i, j]
            text = f"{count}\n({pct:,.1%})"
            ax_cm.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    # Barra colore
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    #cbar.ax.set_ylabel("Row-normalized" if normalize else "Count", rotation=90, va="center")

    # Pannello destro: lista metrica -> valore
    ax_txt.axis("off")

    # Ordine preferito di visualizzazione
    preferred_keys = [
        #"accuracy",
        "balanced_accuracy",
        "recall",
        "far",
        "csi",
        "hss",
        "roc_auc",
        "pr_auc",
        "brier_score_loss",
        "log_loss",
    ]

    # Raccoglie metri che esistono in results
    lines = []
    for k in preferred_keys:
        if k in results:
            v = results[k]
            # Formattazione: percentuali per metriche in [0,1], 3 decimali altrimenti
            if k in { "balanced_accuracy", "recall", "far", "csi", "hss", "roc_auc", "pr_auc"}:   # "accuracy",
                fmt = f"{v:.3f}"
            else:
                fmt = f"{v:.3f}"
            nice = k.replace("_", " ").title()
            lines.append((nice, fmt))

    # Aggiunge i conteggi confusionali alla fine
    #lines.append((" ", " "))
    #lines.append(("True Negatives", f"{tn:d}"))
    #lines.append(("False Positives", f"{fp:d}"))
    #lines.append(("False Negatives", f"{fn:d}"))
    #lines.append(("True Positives", f"{tp:d}"))

    # Disegno testo in due colonne allineate
    x_key, x_val = 0.05, 0.62
    y = 0.95
    dy = 0.085
    for (name, val) in lines:
        ax_txt.text(x_key, y, name, transform=ax_txt.transAxes, ha="left", va="top", fontsize=11)
        ax_txt.text(x_val, y, val, transform=ax_txt.transAxes, ha="right", va="top", fontsize=11)
        y -= dy

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_binary_classifier_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    pos_label: Union[int, str] = 1,
    show_report: bool = False,
) -> dict:
    """Calcola metriche di classificazione binaria sfruttando operazioni GPU.
    ---------------------------------------------------------
    - Balanced Accuracy
    - Recall = POD
    - False Alarm Ratio  FAR
    ---------------------------------------------------------
    - CSI  non viene calcolata durante il training per ottimizzare la velocità
    - HSS  non viene calcolata durante il training per ottimizzare la velocità
    """

    y_true = y_true.to(dtype=torch.int64)
    y_pred = y_pred.to(dtype=torch.int64)

    tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    tn = ((y_true != pos_label) & (y_pred != pos_label)).sum()
    fp = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    fn = ((y_true == pos_label) & (y_pred != pos_label)).sum()

    tp_f = tp.float()
    tn_f = tn.float()
    fp_f = fp.float()
    fn_f = fn.float()
    total = tp_f + tn_f + fp_f + fn_f

    accuracy = (tp_f + tn_f) / total if total > 0 else torch.tensor(0.0, device=y_true.device)
    recall = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else torch.tensor(0.0, device=y_true.device)
    specificity = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #precision = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0, device=y_true.device)
    balanced_accuracy = (recall + specificity) / 2

    #mcc_den = torch.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    #mcc = (tp_f * tn_f - fp_f * fn_f) / mcc_den if mcc_den > 0 else torch.tensor(0.0, device=y_true.device)
    #po = (tp_f + tn_f) / total if total > 0 else torch.tensor(0.0, device=y_true.device)
    #pe = ((tp_f + fp_f) * (tp_f + fn_f) + (fn_f + tn_f) * (fp_f + tn_f)) / (total * total) if total > 0 else torch.tensor(0.0, device=y_true.device)
    #cohen_kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else torch.tensor(0.0, device=y_true.device)

    #fpr = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #fnr = fn_f / (fn_f + tp_f) if (fn_f + tp_f) > 0 else torch.tensor(0.0, device=y_true.device)
    far = fp_f / (fp_f + tp_f) if (fp_f + tp_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #pod = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #csi = tp_f / (tp_f + fp_f + fn_f) if (tp_f + fp_f + fn_f) > 0 else torch.tensor(0.0, device=y_true.device)
    #hss_den = (tp_f + fn_f) * (fn_f + tn_f) + (tp_f + fp_f) * (fp_f + tn_f)
    #hss = 2 * (tp_f * tn_f - fp_f * fn_f) / hss_den if hss_den > 0 else torch.tensor(0.0, device=y_true.device)

    if show_report:
        conf = torch.stack([torch.stack([tn, fp]), torch.stack([fn, tp])])
        print(conf.cpu())

    return {
        "accuracy": accuracy.item(),
        "balanced_accuracy": balanced_accuracy.item(),
        #"precision": precision.item(),
        "recall": recall.item(),
        #"f1": f1.item(),
        #"matthews_corrcoef": mcc.item(),
        #"cohen_kappa": cohen_kappa.item(),
        #"true_negatives": tn.item(),
        #"false_positives": fp.item(),
        #"false_negatives": fn.item(),
        #"true_positives": tp.item(),
        #"false_positive_rate": fpr.item(),
        #"false_negative_rate": fnr.item(),
        "far": far.item(),
        #"pod": pod.item(),
        #"csi": csi.item(),
        #"hss": hss.item(),
    }


def label_to_minutes(label: str) -> Optional[int]:
    """Estrae 'hh_mm' dal nome *label* e lo converte in minuti dopo mezzanotte.

    Esempio: ``label_03_20`` → ``200`` (3*60 + 20). Ritorna ``None`` se il
    pattern non viene trovato.
    """
    m = re.search(r"(\d{2})_(\d{2})$", label)
    if m:
        h, mnt = map(int, m.groups())
        return h * 60 + mnt
    return 0


d = {"tn": "True Negatives", "fp": "False Positives", "fn": "False Negatives", "tp": "True Positives"}


def plot_metrics_over_time(
    conf_df: pd.DataFrame,
    metrics: Sequence[str] = ("tn", "fp", "fn", "tp"),
    sum_labels: List = [],
    xlabel: str = "Time shift (hh:mm)",
    ylabel: str = "Count",
    title: str = "Confusion-matrix counts over time",
) -> None:
    """Plotta i valori di *metrics* (colonne di ``conf_df``) sull'asse x ordinato per ora."""

    times = conf_df.index.to_series().apply(label_to_minutes)
    plot_df = conf_df.copy()
    plot_df["time_min"] = times
    plot_df = plot_df.dropna(subset=["time_min"]).sort_values("time_min")

    x_ticks = [f"{int(t // 60):02d}:{int(t % 60):02d}" for t in plot_df["time_min"]]
    x = plot_df["time_min"].values

    plt.figure(figsize=(10, 5))
    for m in metrics:
        if m in plot_df.columns:
            plt.plot(x, plot_df[m].values, marker="o", label=d[m])

    plt.plot(x, sum_labels, label="Total positives", marker=".")

    plt.xticks(x, x_ticks, rotation=75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()




def calc_metrics_unsupervised():
    args = prepare_args()
    device = torch.device(args.device)

    # Carica i modelli
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    # cambio il path dentro args
    #args.init_ckpt = specialized_model_path
    #specialized_model = get_model(args)

    patch_size = pretrained_model.encoder.patch_embed.patch_size
    data_loader_test = None # get_dataloader(args, patch_size)

    # Ricostruzione e calcolo delle metriche
    with torch.no_grad():
        for batch in data_loader_test:
            images, bool_masked_pos, decode_masked_pos = batch

            # Sposta i dati sul dispositivo
            images = images.to(args.device)
            bool_masked_pos = bool_masked_pos.to(args.device, non_blocking=True).flatten(1).to(torch.bool)
            decode_masked_pos = decode_masked_pos.to(args.device, non_blocking=True).flatten(1).to(torch.bool)

            # Passa i dati al modello
            # Ricostruzioni dei modelli
            pretrained_reconstructed = pretrained_model(images, bool_masked_pos, decode_masked_pos)
            #specialized_reconstructed = specialized_model(images, bool_masked_pos, decode_masked_pos)

            #print("Shape dell'output:", outputs.shape)
            # Calcola le metriche
            #pretrained_mse, pretrained_psnr = reconstruction_metrics(images, pretrained_reconstructed)
            #specialized_mse, specialized_psnr = reconstruction_metrics(images, specialized_reconstructed)

    # Stampa i risultati
    print('=== Risultati delle Metriche di Ricostruzione ===')
    #print(f'Modello Preaddestrato - MSE: {pretrained_mse:.4f}, PSNR: {pretrained_psnr:.2f} dB')
    #print(f'Modello Specializzato - MSE: {specialized_mse:.4f}, PSNR: {specialized_psnr:.2f} dB')

    # Conclusione
    #if specialized_mse < pretrained_mse:
    #    print('Il modello specializzato ha una migliore ricostruzione!')
    #else:
    #    print('Il modello preaddestrato ha una migliore ricostruzione.')

if __name__ == "__main__":
    calc_metrics_unsupervised()#image_folder)
