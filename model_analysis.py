import os
from glob import glob
from PIL import Image
from typing import List, Union, Optional, Sequence, Mapping, Tuple
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

from sklearn.decomposition import PCA

from utils import NativeScalerWithGradNormCount as NativeScaler

from dataset import build_pretraining_dataset
from torch.utils.data import DataLoader
from utils import get_model
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from dataset.data_manager import make_validation_data_builder_from_manos_tracks, make_validation_data_builder_from_entire_year
from dataset.build_dataset import calc_avg_cld_idx



# region  Funzione per caricare le immagini
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

# endregion


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


# region raccogliere le logits

def load_logits_dir(save_dir='output/val_logits', prefix='val'):
    """Carica e concatena i file .npz prodotti dalla validazione.
    Ordine di preferenza:
      1) singolo file globale `{prefix}_all_merged.npz` (se esiste)
      2) file merged per-rank `{prefix}_rank*_merged.npz`
      3) shard per-batch `{prefix}_rank*_*part*.npz`

    Ritorna: logits (N,C), labels (N,), preds (N,), paths (N,)
    """
    # 1) unico file globale
    global_merged = os.path.join(save_dir, f"{prefix}_merged.npz")
    if os.path.exists(global_merged):
        files = [global_merged]
    else:
        # 2) merged per-rank
        merged = sorted(glob(os.path.join(save_dir, f"{prefix}_rank*_merged.npz")))
        # 3) altrimenti shard per-batch
        files = merged if len(merged) > 0 else sorted(glob(os.path.join(save_dir, f"{prefix}_rank*_*part*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f'No npz files found in {save_dir}')
    logits_list, labels_list, preds_list, paths_list = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            logits_list.append(d['logits'])
            labels_list.append(d['labels'])
            preds_list.append(d['preds'])
            paths_list.append(d['paths'])
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    paths = np.concatenate(paths_list, axis=0)
    return logits, labels, preds, paths


def merge_all_rank_merged(save_dir='output/val_logits', prefix='val', output_filename=None,
                          delete_inputs=True, value_key: str = 'logits'):
    """Unisce tutti i file merged per-rank in un unico `{prefix}_all_merged.npz`.

    - Cerca: `{prefix}_rank*_merged.npz`
    - Salva: `{output_filename}` se fornito, altrimenti `{prefix}_all_merged.npz`
    - Se `delete_inputs=True`, cancella i singoli merged per-rank dopo il salvataggio.
    - `value_key` indica la chiave principale da concatenare (es. 'logits' o 'embeddings').

    Ritorna: percorso del file globale salvato.
    """
    files = sorted(glob(os.path.join(save_dir, f"{prefix}_rank*_merged.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No per-rank merged files found in {save_dir}")

    values_list, labels_list, preds_list, paths_list = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            values_list.append(d[value_key])
            labels_list.append(d['labels'])
            preds_list.append(d['preds'])
            paths_list.append(d['paths'])

    merged_values = np.concatenate(values_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    paths = np.concatenate(paths_list, axis=0)

    out = output_filename or os.path.join(save_dir, f"{prefix}_all_merged.npz")
    np.savez_compressed(out,
                        **{value_key: merged_values},
                        labels=labels,
                        preds=preds,
                        paths=paths)

    if delete_inputs:
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove {f}: {e}")
    return out


def plot_embeddings_umap(npz_path: str,
                         color_by: str = 'labels',
                         output_path: str = 'embeddings_umap.png',
                         random_state: int = 42,
                         n_neighbors: int = 15,
                         min_dist: float = 0.1,
                         show: bool = False):
    """Carica un file npz e produce una proiezione UMAP 2D delle embeddings.

    Parametri
    ---------
    npz_path: file generato da collect_embeddings con chiave `embeddings`.
    color_by: 'labels', 'preds' oppure 'none'.
    output_path: file immagine di output (PNG).
    random_state, n_neighbors, min_dist: iperparametri UMAP.
    show: se True richiama plt.show() oltre a salvare.
    """
    import numpy as np

    try:
        import umap
    except ImportError as e:
        raise ImportError("Richiede il pacchetto 'umap-learn'.") from e

    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data:
        raise KeyError(f"Il file {npz_path} non contiene la chiave 'embeddings'.")

    embeddings = data['embeddings']
    labels = data['labels'] if 'labels' in data else None
    preds = data['preds'] if 'preds' in data else None

    if embeddings.ndim != 2:
        raise ValueError(f"Attesa matrice 2D, trovata shape {embeddings.shape}")

    n_samples, emb_dim = embeddings.shape

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean'
    )
    projection = reducer.fit_transform(embeddings)

    if color_by == 'labels' and labels is not None:
        colors = labels
        color_label = 'Label'
    elif color_by == 'preds' and preds is not None:
        colors = preds
        color_label = 'Prediction'
    else:
        colors = None
        color_label = None

    plt.figure(figsize=(8, 6))
    scatter_kwargs = dict(s=8, alpha=0.7, cmap='Spectral')
    if colors is None:
        plt.scatter(projection[:, 0], projection[:, 1], color='tab:blue', **scatter_kwargs)
    else:
        plt.scatter(projection[:, 0], projection[:, 1], c=colors, **scatter_kwargs)
        cbar = plt.colorbar()
        cbar.set_label(color_label)

    title = f"UMAP embeddings (N={n_samples}, dim={emb_dim})"
    plt.title(title)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    print(title)
    print(f"Salvato plot in: {output_path}")


def cleanup_npz_shards(save_dir='output/val_logits', prefix='val', only_if_merged=True, dry_run=False):
    """Cancella gli shard per-batch `{prefix}_rank*_*part*.npz`.

    - Se `only_if_merged=True`, procede solo per i rank che hanno il corrispondente
      `{prefix}_rank{R}_merged.npz` presente (più sicuro).
    - `dry_run=True` stampa i file che verrebbero rimossi, senza cancellarli.

    Ritorna: lista dei file (effettivamente) rimossi.
    """
    import re
    part_files = sorted(glob(os.path.join(save_dir, f"{prefix}_rank*_*part*.npz")))
    if len(part_files) == 0:
        return []

    removed = []
    if only_if_merged:
        merged_available = set()
        for f in glob(os.path.join(save_dir, f"{prefix}_rank*_merged.npz")):
            m = re.search(rf"{re.escape(prefix)}_rank(\d+)_merged\\.npz$", os.path.basename(f))
            if m:
                merged_available.add(m.group(1))

        for f in part_files:
            m = re.search(rf"{re.escape(prefix)}_rank(\d+)_part\d+\\.npz$", os.path.basename(f))
            if not m:
                continue
            rk = m.group(1)
            if rk in merged_available:
                if dry_run:
                    print(f"Would remove {f}")
                else:
                    try:
                        os.remove(f)
                        removed.append(f)
                    except Exception as e:
                        print(f"Warning: could not remove {f}: {e}")
    else:
        for f in part_files:
            if dry_run:
                print(f"Would remove {f}")
            else:
                try:
                    os.remove(f)
                    removed.append(f)
                except Exception as e:
                    print(f"Warning: could not remove {f}: {e}")

    return removed


def build_video_level_df(
    manos_csv: str = 'medicane_data_input/medicanes_new_windows.csv',
    input_dir: str = '../fromgcloud',
    output_dir: str = '../airmassRGB/supervised/',
    logits_dir: str = 'output/val_logits',
    prefix: str = 'val',
    add_logit_scores: bool = True,
    compute_avg_cloud_if_missing: bool = True,
    entire_year: int=None,
) -> pd.DataFrame:
    """Crea un DataFrame a livello di video unendo:
    - logits/predizioni/label dalla validazione
    - info da df_video (neighboring, offsets, tempi)
    - avg_cloud_idx per video (calcolato se mancante)

    Parametri
    ---------
    manos_csv: CSV delle finestre di validazione (Manos) per costruire df_video
    input_dir: cartella con le immagini di partenza
    output_dir: cartella dei video (cartelle con img_00001.png..)
    logits_dir: cartella dove sono gli npz di validazione
    prefix: prefisso file npz (es. 'val')
    add_logit_scores: se True aggiunge colonne logit_0/1, margin, prob_pos
    compute_avg_cloud_if_missing: calcola avg_cloud_idx se non presente in df_video

    Ritorna
    -------
    DataFrame con una riga per video e colonne:
      path, predictions, labels, [logit_0, logit_1, margin, prob_pos],
      label (da df_video), neighboring, avg_cloud_idx, tile_offset_x, tile_offset_y,
      start_time, end_time
    """
    # 1) Carica logits/preds/labels/paths
    logits, labels, preds, paths = load_logits_dir(save_dir=logits_dir, prefix=prefix)
    df_predictions = create_df_predictions(paths, preds, labels)  # path, predictions, labels

    # 2) Punteggi derivati dai logits (opzionale)
    df_scores = None
    if add_logit_scores and logits is not None:
        logits = np.asarray(logits)
        if logits.ndim == 2 and logits.shape[1] >= 2:
            # softmax numericamente stabile via shift
            z = logits - logits.max(axis=1, keepdims=True)
            expz = np.exp(z)
            prob_pos = expz[:, 1] / expz.sum(axis=1)
            df_scores = pd.DataFrame({
                'path': pd.Series(paths).astype(str).str.split('/').str.get(-1),
                'logit_0': logits[:, 0],
                'logit_1': logits[:, 1],
                'margin': logits[:, 1] - logits[:, 0],
                'prob_pos': prob_pos,
            })

    # 3) Costruisce builder di validation (df_video contiene 'neighboring')
    if entire_year is not None:
        val_b = make_validation_data_builder_from_entire_year(entire_year, input_dir, output_dir)
    else:
        val_b = make_validation_data_builder_from_manos_tracks(manos_csv, input_dir, output_dir)
    df_video = val_b.df_video.copy()

    # 4) Assicura avg_cloud_idx per video
    if compute_avg_cloud_if_missing and 'avg_cloud_idx' not in df_video.columns:
        base = output_dir if output_dir.endswith('/') else output_dir + '/'
        df_video['avg_cloud_idx'] = (base + df_video['path']).apply(calc_avg_cld_idx)

    # 5) Merge finale livello video
    cols_keep = [
        'path', 'label', 'neighboring', 'avg_cloud_idx',
        'tile_offset_x', 'tile_offset_y', 'start_time', 'end_time'
    ]
    base_df = df_predictions
    if df_scores is not None:
        base_df = base_df.merge(df_scores, on='path', how='left')

    df_merged = base_df.merge(df_video[cols_keep], on='path', how='left')
    return df_merged


_HAS_SK = True # usa la PCA per ridurre se ci sono pi ù di due dimensioni
#_HAS_SK = False

def plot_logits_by_label(logits, labels, class_names=None, max_points=20000, alpha=0.5):
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    C = logits.shape[1]
    # Subsample per scatter denso
    if logits.shape[0] > max_points:
        rng = np.random.RandomState(0)
        idx = rng.choice(logits.shape[0], size=max_points, replace=False)
        X = logits[idx]
        y = labels[idx]
    else:
        X = logits
        y = labels
    uniq = np.unique(y)
    cmap = plt.cm.get_cmap('tab10', max(10, len(uniq)))
    if C == 2:
        # Scatter 2D dei due logit (z0, z1) per label
        z0, z1 = X[:, 0], X[:, 1]
        plt.figure(figsize=(6, 6))
        for i, lab in enumerate(uniq):
            m = (y == lab)
            lbl = class_names[lab] if class_names is not None and lab < len(class_names) else f'class {lab}'
            plt.scatter(z0[m], z1[m], s=10, alpha=alpha, label=lbl, color=cmap(i))
        # retta decisionale z1 = z0
        lim_min = float(np.min(np.concatenate([z0, z1]))) - 1
        lim_max = float(np.max(np.concatenate([z0, z1]))) + 1
        lims = [lim_min, lim_max]
        plt.plot(lims, lims, 'k--', linewidth=1, alpha=0.6)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel('logit[0]')
        plt.ylabel('logit[1]')
        plt.title('Logits binari per label (z0 vs z1)')
        plt.legend()
        plt.tight_layout()
    else:
        # PCA a 2D per scatter colorato per label
        if _HAS_SK:
            Z = PCA(n_components=2).fit_transform(X)
            xlabel, ylabel = 'PC1', 'PC2'
        else:
            Z = X[:, :2] if X.shape[1] >= 2 else np.pad(X, ((0,0),(0, 2-X.shape[1])), mode='constant')
            xlabel, ylabel = 'logit[0]', 'logit[1]'
        plt.figure(figsize=(6, 6))
        for i, lab in enumerate(uniq):
            m = (y == lab)
            lbl = class_names[lab] if class_names is not None and lab < len(class_names) else f'class {lab}'
            plt.scatter(Z[m, 0], Z[m, 1], s=10, alpha=alpha, label=lbl, color=cmap(i))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Logits (ridotti) per label')
        plt.legend()
        plt.tight_layout()


def plot_logit_margin_hist(logits, labels, class_names=None, bins=50, density=False, alpha=0.5):
    """Istogramma del margine logit per classificazione binaria.

    Mostra la distribuzione di d = z1 - z0 separata per label.
    - logits: array (N, 2)
    - labels: array (N,)
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("plot_logit_margin_hist richiede logits di forma (N, 2)")

    d = logits[:, 1] - logits[:, 0]
    uniq = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10', max(10, len(uniq)))

    plt.figure(figsize=(8, 4))
    for i, lab in enumerate(uniq):
        m = (labels == lab)
        lbl = class_names[lab] if class_names is not None and lab < len(class_names) else f'class {lab}'
        plt.hist(d[m], bins=bins, alpha=alpha, density=density, label=lbl, color=cmap(i))
    # set x axis log
    plt.yscale('log')
    plt.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('logit[1] - logit[0]')
    plt.ylabel('density' if density else 'count')
    plt.title('Logit distribution for each class')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('logits_by_label.png')



# endregion

# region Plot: margin istogramma con info df (neighboring, avg_cloud_idx)

def plot_logit_margin_hist_with_df(
    df: pd.DataFrame,
    bins: int = 50,
    density: bool = False,
    alpha: float = 0.9,
    cmap_name: str = 'Blues_r',
    label_col: str = 'labels',
    margin_col: str = 'margin',
    neighbor_col: str = 'neighboring',
    cloud_col: str = 'avg_cloud_idx',
    save_prefix: Optional[str] = None,
):
    """Crea due istogrammi del margine `z1 - z0`, uno per ciascuna classe
    (label 0 e label 1), arricchiti con:
    - colore delle barre proporzionale all'`avg_cloud_idx` medio nel bin (cmap blu→bianco, 0→blu, 1→bianco)
    - overlay arancione per i soli esempi con `neighboring == True`.

    Parametri
    ---------
    df : pd.DataFrame
        DataFrame prodotto da `build_video_level_df` con colonne richieste.
    bins : int
        Numero di bin dell'istogramma.
    density : bool
        Se True normalizza le altezze (come in plt.hist(density=True)).
    alpha : float
        Opacità delle barre principali (colorate per cloud_idx).
    cmap_name : str
        Nome della colormap matplotlib (default 'Blues_r' per 0→blu, 1→bianco).
    label_col, margin_col, neighbor_col, cloud_col : str
        Nomi delle colonne nel DataFrame.
    save_prefix : Optional[str]
        Se fornito, salva i plot in file PNG con questo prefisso per label.
    """
    required = {label_col, margin_col, neighbor_col, cloud_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"plot_logit_margin_hist_with_df: colonne mancanti nel DataFrame: {missing}")

    # Filtra righe valide (margin e label non null)
    dfx = df.dropna(subset=[label_col, margin_col, cloud_col])
    labels = dfx[label_col].astype(int).values
    margins = dfx[margin_col].values
    clouds = dfx[cloud_col].clip(lower=0.0, upper=1.0).values
    neighb = dfx[neighbor_col].astype(bool).values

    # Usa stessi bin per entrambe le classi per confronto corretto
    data_min, data_max = np.nanmin(margins), np.nanmax(margins)
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
        # fallback
        data_min, data_max = -1.0, 1.0
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    # Colormap 0→blu, max→bianco (usa max reale, non 1 fisso)
    cmap = plt.cm.get_cmap(cmap_name)
    global_max_cloud = float(np.nanmax(dfx[cloud_col].values)) if cloud_col in dfx.columns else 1.0
    if not np.isfinite(global_max_cloud) or global_max_cloud <= 0:
        global_max_cloud = 1.0
    norm = plt.Normalize(vmin=0.0, vmax=global_max_cloud)

    for lab in [0, 1]:
        m = (labels == lab)
        if not np.any(m):
            continue
        d_lab = margins[m]
        c_lab = clouds[m]
        n_lab = neighb[m]

        # Conteggio principale per tutti i sample (colorato per avg_cloud_idx medio nel bin)
        counts_all, _ = np.histogram(d_lab, bins=bin_edges)
        # Media cloud per bin (usiamo somma pesata / conteggio)
        sum_cloud = np.zeros_like(counts_all, dtype=float)
        # Assegna ciascun valore al bin e accumula
        bin_idx = np.clip(np.digitize(d_lab, bin_edges) - 1, 0, len(counts_all) - 1)
        for ci, bi in zip(c_lab, bin_idx):
            sum_cloud[bi] += float(ci)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_cloud = np.where(counts_all > 0, sum_cloud / counts_all, np.nan)

        # Eventuale normalizzazione densities
        heights_all = counts_all.astype(float)
        if density:
            total = heights_all.sum()
            if total > 0:
                heights_all = heights_all / total

        # Larghezze e posizioni barre
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1]

        # Colore per bin in base alla media cloud
        bar_colors = [cmap(norm(mc)) if np.isfinite(mc) else (0.9, 0.9, 0.9, 1.0) for mc in mean_cloud]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        # Barre base colorate per cloud
        ax.bar(centers, heights_all, width=widths, align='edge', color=bar_colors, alpha=alpha, edgecolor='none', label='Tutti (colorati per avg_cloud_idx)')

        # Overlay per neighboring=True in arancione
        if np.any(n_lab):
            counts_nei, _ = np.histogram(d_lab[n_lab], bins=bin_edges)
            heights_nei = counts_nei.astype(float)
            if density:
                total = counts_all.sum()
                if total > 0:
                    heights_nei = heights_nei / total
            # Disegna come barre più strette e trasparenti
            ax.bar(centers + 0.1 * widths, heights_nei, width=0.8 * widths, align='edge', color='orange', alpha=0.6, edgecolor='none', label=f'neighboring=True (N={counts_nei.sum()})')

        # Stile asse y coerente con plot originale
        ax.set_yscale('log')
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('logit[1] - logit[0]')
        ax.set_ylabel('density' if density else 'count')
        ax.set_title(f'Distribuzione margine logit — label {lab}')

        # Annotazione conteggi a sinistra/destra di 0 (totale su questo label)
        n_left = int(np.sum(d_lab < 0))
        n_right = int(np.sum(d_lab >= 0))
        txt = f"N<0={n_left}\nN>=0={n_right}"
        ax.text(0.99, 0.98, txt, transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        # Colorbar per avg_cloud_idx
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(f'avg_cloud_idx (0=blu, max≈{global_max_cloud:.2f}=bianco)')

        ax.legend()
        fig.tight_layout()

        if save_prefix is not None:
            out = f"{save_prefix}_label{lab}.png"
            try:
                fig.savefig(out, dpi=150)
            except Exception:
                pass

# endregion

# region Nuovi plot: scatter cloud_idx e istogrammi per neighboring

def scatter_margin_vs_cloud(
    df: pd.DataFrame,
    sample: Optional[int] = 50000,
    alpha: float = 0.3,
    s: float = 8.0,
    label_col: str = 'labels',
    margin_col: str = 'margin',
    cloud_col: str = 'avg_cloud_idx',
    neighbor_col: str = 'neighboring',
    save_path: Optional[str] = None,
):
    """Scatter plot: x = margin (z1 - z0), y = avg_cloud_idx, colore per classe (0/1).

    - Usa un eventuale sottocampionamento per performance (parametro `sample`).
    - Limita l'asse Y al massimo reale del dataset (non 1 fisso).
    """
    required = {label_col, margin_col, cloud_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"scatter_margin_vs_cloud: colonne mancanti: {missing}")

    # neighbor_col è opzionale: se manca, non verrà usato
    subset_cols = [label_col, margin_col, cloud_col] + ([neighbor_col] if neighbor_col in df.columns else [])
    dfx = df.dropna(subset=subset_cols).copy()
    dfx[label_col] = dfx[label_col].astype(int)
    has_neighbor = neighbor_col in dfx.columns
    if has_neighbor:
        dfx[neighbor_col] = dfx[neighbor_col].astype(bool)

    if sample is not None and len(dfx) > sample:
        rng = np.random.RandomState(0)
        dfx = dfx.iloc[rng.choice(len(dfx), size=sample, replace=False)]

    x = dfx[margin_col].values
    y = dfx[cloud_col].values
    y_max = float(np.nanmax(y)) if len(y) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    uniq = sorted(dfx[label_col].unique())
    # Colori convenzionali per classi: 0 blu, 1 arancione
    base_cmap = plt.cm.get_cmap('tab10', 10)
    class_colors = {0: base_cmap(0), 1: base_cmap(1)}

    fig, ax = plt.subplots(figsize=(8, 5))
    handles = []
    labels = []
    for lab in uniq:
        m = (dfx[label_col].values == lab)
        color = class_colors.get(int(lab), base_cmap(int(lab)))
        if has_neighbor:
            m_true = m & (dfx[neighbor_col].values == True)
            m_false = m & (dfx[neighbor_col].values == False)
            # Non-neighboring: punti standard
            sc1 = ax.scatter(x[m_false], y[m_false], s=s, alpha=alpha, color=color)
            # Neighboring=True: triangolino rosso con riempimento del colore della classe 0
            sc2 = ax.scatter(
                x[m_true], y[m_true],
                marker='^',
                s=s*1.6,
                alpha=min(1.0, alpha+0.1),
                facecolors=class_colors.get(0, base_cmap(0)),
                edgecolors='red',
                linewidths=0.8,
            )
            if np.any(m):
                handles.append(sc1)
                labels.append(f'class {lab}')
        else:
            sc = ax.scatter(x[m], y[m], s=s, alpha=alpha, color=color, label=f'class {lab}')
            handles.append(sc)
            labels.append(f'class {lab}')

    ax.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('logit[1] - logit[0] (margin)')
    ax.set_ylabel('avg_cloud_idx')
    ax.set_ylim(0, y_max * 1.02)
    ax.set_title('Margin vs Cloud Index (colori per classe)')
    # Legenda: classi + indicatore neighboring (se presente)
    if has_neighbor:
        from matplotlib.lines import Line2D
        neigh_handle = Line2D(
            [0], [0], marker='^', linestyle='None',
            markerfacecolor=class_colors.get(0, base_cmap(0)),
            markeredgecolor='red', markeredgewidth=1.0,
            label='neighboring=True'
        )
        handles = handles + [neigh_handle]
        labels = labels + ['neighboring=True']
        ax.legend(handles, labels, loc='best')
    else:
        ax.legend(loc='best')
    fig.tight_layout()

    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=150)
        except Exception:
            pass


def plot_margin_hist_by_neighboring(
    df: pd.DataFrame,
    bins: int = 50,
    density: bool = False,
    alpha: float = 0.5,
    label_col: str = 'labels',
    margin_col: str = 'margin',
    neighbor_col: str = 'neighboring',
    save_prefix: Optional[str] = None,
):
    """Crea due istogrammi del margine, uno per neighboring=True e uno per neighboring=False,
    sovrapponendo le due classi con colori distinti (0/1).
    """
    required = {label_col, margin_col, neighbor_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"plot_margin_hist_by_neighboring: colonne mancanti: {missing}")

    dfx = df.dropna(subset=[label_col, margin_col, neighbor_col]).copy()
    dfx[label_col] = dfx[label_col].astype(int)
    dfx[neighbor_col] = dfx[neighbor_col].astype(bool)

    margins = dfx[margin_col].values
    data_min, data_max = np.nanmin(margins), np.nanmax(margins)
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
        data_min, data_max = -1.0, 1.0
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    base_cmap = plt.cm.get_cmap('tab10', 10)
    class_colors = {0: base_cmap(0), 1: base_cmap(1)}

    for neigh_val in [True, False]:
        sub = dfx[dfx[neighbor_col] == neigh_val]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for lab in [0, 1]:
            s = sub[sub[label_col] == lab]
            if s.empty:
                continue
            vals = s[margin_col].values
            ax.hist(vals, bins=bin_edges, density=density, alpha=alpha, color=class_colors[lab], label=f'class {lab}')

        ax.set_yscale('log')
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('logit[1] - logit[0]')
        ax.set_ylabel('density' if density else 'count')
        ax.set_title(f'Logit distribution — neighboring={neigh_val}')
        # Annotazioni: N<0 in alto a sinistra, N>=0 a destra (totale o per-classe)
        n_left = int(np.sum(sub[margin_col].values < 0))
        ax.text(0.01, 0.98, f"N<0 = {n_left}", transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        if neigh_val is False:
            n_right_cls0 = int(np.sum((sub[label_col].values == 0) & (sub[margin_col].values >= 0)))
            n_right_cls1 = int(np.sum((sub[label_col].values == 1) & (sub[margin_col].values >= 0)))
            ax.text(0.99, 0.98, f"N>=0 class 0 = {n_right_cls0}", transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=class_colors[0],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            ax.text(0.99, 0.90, f"N>=0 class 1 = {n_right_cls1}", transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=class_colors[1],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            n_right = int(np.sum(sub[margin_col].values >= 0))
            ax.text(0.99, 0.98, f"N>=0 = {n_right}", transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        ax.legend(loc='center')
        fig.tight_layout()

        if save_prefix is not None:
                out = f"{save_prefix}_neighboring_{str(neigh_val).lower()}.png"
                try:
                    fig.savefig(out, dpi=150)
                except Exception:
                    pass

# endregion

# region Plot: avg_cloud_idx istogramma separato per neighboring, con filtro margin>0

def plot_cloud_hist_by_neighboring(
    df: pd.DataFrame,
    bins: int = 50,
    density: bool = False,
    alpha: float = 0.5,
    label_col: str = 'labels',
    margin_col: str = 'margin',
    neighbor_col: str = 'neighboring',
    cloud_col: str = 'avg_cloud_idx',
    save_prefix: Optional[str] = None,
    margin_threshold: float = 0.0,
    clip_cloud_to: Optional[Tuple[float, float]] = (0.0, 1.0),
):
    """Crea due istogrammi di ``avg_cloud_idx`` (asse x), uno per ``neighboring=True`` e
    uno per ``neighboring=False``, sovrapponendo le due classi (0/1) con colori distinti.

    I dati inclusi rispettano il filtro ``margin > margin_threshold`` (default 0.0).

    Parametri
    ---------
    df : pd.DataFrame
        DataFrame con colonne: labels, margin, neighboring, avg_cloud_idx.
    bins : int
        Numero di bin dell'istogramma.
    density : bool
        Se True normalizza le altezze.
    alpha : float
        Opacità delle barre.
    label_col, margin_col, neighbor_col, cloud_col : str
        Nomi colonne nel DataFrame.
    save_prefix : Optional[str]
        Se valorizzato salva i plot PNG con questo prefisso.
    margin_threshold : float
        Filtro su margin; vengono considerati i soli sample con ``margin > margin_threshold``.
    clip_cloud_to : Optional[Tuple[float, float]]
        Se non None, i valori di ``avg_cloud_idx`` sono troncati nell'intervallo dato (default (0, 1)).
    """

    required = {label_col, margin_col, neighbor_col, cloud_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"plot_cloud_hist_by_neighboring: colonne mancanti: {missing}")

    dfx = df.dropna(subset=[label_col, margin_col, neighbor_col, cloud_col]).copy()
    dfx[label_col] = dfx[label_col].astype(int)
    dfx[neighbor_col] = dfx[neighbor_col].astype(bool)

    # Filtro: margin > threshold
    dfx = dfx[dfx[margin_col] > margin_threshold]
    if dfx.empty:
        raise ValueError("plot_cloud_hist_by_neighboring: nessun dato dopo il filtro margin > threshold")

    # Range bin su avg_cloud_idx
    clouds = dfx[cloud_col].values.astype(float)
    if clip_cloud_to is not None:
        lo, hi = clip_cloud_to
        clouds = np.clip(clouds, lo, hi)
    cmin, cmax = float(np.nanmin(clouds)), float(np.nanmax(clouds))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0
    bin_edges = np.linspace(cmin, cmax, bins + 1)

    base_cmap = plt.cm.get_cmap('tab10', 10)
    class_colors = {0: base_cmap(0), 1: base_cmap(1)}

    for neigh_val in [True, False]:
        sub = dfx[dfx[neighbor_col] == neigh_val]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for lab in [0, 1]:
            s = sub[sub[label_col] == lab]
            if s.empty:
                continue
            vals = s[cloud_col].values.astype(float)
            if clip_cloud_to is not None:
                lo, hi = clip_cloud_to
                vals = np.clip(vals, lo, hi)
            ax.hist(vals, bins=bin_edges, density=density, alpha=alpha, color=class_colors[lab], label=f'class {lab}')

        #ax.set_yscale('log')
        ax.set_xlabel('avg_cloud_idx')
        ax.set_ylabel('density' if density else 'count')
        ax.set_title(f'Distribuzione avg_cloud_idx — neighboring={neigh_val} — margin>{margin_threshold}')

        # Annotazioni di supporto
        n_total = int(len(sub))
        ax.text(0.01, 0.98, f"N = {n_total}", transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        ax.legend(loc='best')
        fig.tight_layout()

        if save_prefix is not None:
            out = f"{save_prefix}_neighboring_{str(neigh_val).lower()}.png"
            try:
                fig.savefig(out, dpi=150)
            except Exception:
                pass

# endregion

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
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    data_for_color = cm_norm if normalize else counts
    im = ax_cm.imshow(data_for_color, interpolation="nearest", cmap=cmap,
                      vmin=0.0, vmax=(1.0 if normalize else data_for_color.max() or 1.0))

    # Assi e ticks
    ax_cm.set_title(title, fontsize=16, pad=16)
    ax_cm.set_ylabel("True label", fontsize=14)
    ax_cm.set_xlabel("Predicted label", fontsize=14)
    ax_cm.set_xticks([0, 1], labels=labels)
    ax_cm.set_yticks([0, 1], labels=labels)
    ax_cm.tick_params(axis="both", labelsize=12)

    # Annotazioni: conteggio + percentuale per cella
    for i in range(2):
        for j in range(2):
            count = int(counts[i, j])
            pct = cm_norm[i, j]
            text = f"{count}\n({pct:,.1%})"
            text_color = "white" if i == j else "black"
            ax_cm.text(j, i, text, ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    # Barra colore
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
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
        ax_txt.text(
            x_key,
            y,
            name,
            transform=ax_txt.transAxes,
            ha="left",
            va="top",
            fontsize=15,
        )
        ax_txt.text(
            x_val,
            y,
            val,
            transform=ax_txt.transAxes,
            ha="right",
            va="top",
            fontsize=15,
            fontweight="bold",
        )
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


# endregion


#region altre

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

# endregion
