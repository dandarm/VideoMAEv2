import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import warnings
warnings.filterwarnings('ignore')



def evaluate_clustering_vs_dims(
    embeddings: np.ndarray,
    dims=(5,10,15,20,30,50),
    y_true: np.ndarray  = None,
    pca_components: int = 100,
    umap_kwargs: dict = None,
    hdbscan_kwargs: dict = None,
    random_state: int = None
) -> pd.DataFrame:
    """
    Sweep su più dimensioni target per UMAP e valutazione HDBSCAN.
    Ritorna un DataFrame con metriche interne ed esterne (se y_true fornito).

    Params
    ------
    embeddings : array (n_samples, n_features)   # es. (2400, 1400)
    dims : iterable di int                       # dimensioni UMAP per il clustering
    y_true : array opzionale con etichette reali (per ARI/NMI, esclusi i punti rumore)
    pca_components : int                         # PCA preliminare per velocizzare/stabilizzare
    umap_kwargs : dict                           # override di default UMAP
    hdbscan_kwargs : dict                        # override di default HDBSCAN
    random_state : int|None                      # seed per riproducibilità; None = più parallelo/varianza

    Returns
    -------
    pd.DataFrame con colonne:
      ['dim', 'n_clusters', 'noise_frac', 'silhouette', 'davies_bouldin',
       'calinski_harabasz', 'avg_cluster_size', 'median_cluster_size',
       'ARI', 'NMI']
    """
    if umap_kwargs is None:
        umap_kwargs = dict(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=random_state)
    if hdbscan_kwargs is None:
        hdbscan_kwargs = dict(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')

    # PCA preliminare (veloce e spesso benefica)
    pca = PCA(n_components=min(pca_components, embeddings.shape[1]), random_state=random_state)
    X_pca = pca.fit_transform(embeddings)

    rows = []
    for d in dims:
        # 1) UMAP alla dimensione d (per clustering)
        umap_model = umap.UMAP(n_components=d, **umap_kwargs)
        X_d = umap_model.fit_transform(X_pca)

        # 2) HDBSCAN sullo spazio ridotto
        clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
        labels = clusterer.fit_predict(X_d)  # -1 = rumore

        # indicatori base
        unique = np.unique(labels)
        cluster_ids = [c for c in unique if c != -1]
        n_clusters = len(cluster_ids)
        noise_frac = float(np.mean(labels == -1))

        # distribuzione dimensioni cluster
        cluster_sizes = [np.sum(labels == c) for c in cluster_ids]
        avg_cs = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        med_cs = float(np.median(cluster_sizes)) if cluster_sizes else 0.0

        # 3) Metriche interne (richiedono >=2 cluster "validi")
        if n_clusters >= 2:
            # Silhouette sui soli punti non-rumore
            mask = labels != -1
            try:
                sil = silhouette_score(X_d[mask], labels[mask], metric='euclidean') if np.sum(mask) > 1 else np.nan
            except Exception:
                sil = np.nan

            # DB/CH su tutti i punti assegnati a cluster (esclude rumore)
            try:
                db = davies_bouldin_score(X_d[mask], labels[mask]) if np.sum(mask) > 1 else np.nan
            except Exception:
                db = np.nan
            try:
                ch = calinski_harabasz_score(X_d[mask], labels[mask]) if np.sum(mask) > 1 else np.nan
            except Exception:
                ch = np.nan
        else:
            sil = np.nan
            db = np.nan
            ch = np.nan

        # 4) Metriche esterne (opzionali): ARI/NMI su subset non-rumore
        ARI = np.nan
        NMI = np.nan
        if y_true is not None:
            mask = labels != -1
            if mask.any() and len(np.unique(labels[mask])) >= 2 and len(np.unique(y_true[mask])) >= 2:
                try:
                    ARI = adjusted_rand_score(y_true[mask], labels[mask])
                    NMI = normalized_mutual_info_score(y_true[mask], labels[mask])
                except Exception:
                    ARI = np.nan
                    NMI = np.nan

        rows.append(dict(
            dim=d,
            n_clusters=n_clusters,
            noise_frac=noise_frac,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            avg_cluster_size=avg_cs,
            median_cluster_size=med_cs,
            ARI=ARI,
            NMI=NMI
        ))

    return pd.DataFrame(rows).sort_values('dim').reset_index(drop=True)

import pandas as pd
import numpy as np

def repeat_clustering_evaluation(
    embeddings: np.ndarray,
    dims=(5,10,15,20,30,50),
    n_repeats: int = 5,
    #seeds: list[int|None] | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Ripete evaluate_clustering_vs_dims più volte con seed diversi
    e concatena i risultati in un unico DataFrame.

    Params
    ------
    embeddings : array (n_samples, n_features)
    dims : iterable di int (dimensioni target per UMAP)
    n_repeats : quante ripetizioni fare
    seeds : lista di seed (se None, genera automaticamente)
    kwargs : parametri aggiuntivi passati a evaluate_clustering_vs_dims

    Returns
    -------
    DataFrame con colonne originali + 'repeat' e 'seed'
    """
    #if seeds is None:
        # se vuoi includere run "non deterministici", metti alcuni None
    #    seeds = [None if i==0 else np.random.randint(0, 10_000) for i in range(n_repeats)]
    #elif len(seeds) < n_repeats:
    #    raise ValueError("La lista seeds è più corta di n_repeats")

    all_runs = []
    for i in range(n_repeats):
        #seed = seeds[i]
        print(f"Iter {i}...")
        df = evaluate_clustering_vs_dims(
            embeddings,
            dims=dims,
            random_state=None,
            **kwargs
        )
        df['repeat'] = i
        #df['seed'] = seed
        all_runs.append(df)

    return pd.concat(all_runs, ignore_index=True)

def plot_metrics_with_errorbars(results_multi, metrics=('silhouette','davies_bouldin','calinski_harabasz')):
    """
    Plotta metriche con media ± deviazione standard su più ripetizioni.
    results_multi: DataFrame con colonne ['dim', 'repeat', metriche...]
    metrics: lista di metriche da plottare
    """
    grouped = results_multi.groupby('dim')

    plt.figure(figsize=(15, 4))

    for i, metric in enumerate(metrics, start=1):
        means = grouped[metric].mean()
        stds = grouped[metric].std()

        plt.subplot(1, len(metrics), i)
        plt.errorbar(means.index, means.values, yerr=stds.values, fmt='-o', capsize=5)
        plt.xlabel("Dimensioni UMAP")
        plt.ylabel(metric)
        if metric == 'silhouette':
            plt.title("Silhouette (↑ meglio)")
        elif metric == 'davies_bouldin':
            plt.title("Davies-Bouldin (↓ meglio)")
        elif metric == 'calinski_harabasz':
            plt.title("Calinski-Harabasz (↑ meglio)")
        else:
            plt.title(metric)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("metrics_clustering.png")
    plt.show()

def plot_cluster_stats_with_errorbars(results_multi):
    """
    Plotta media ± std del numero di cluster e della frazione di rumore
    al variare delle dimensioni UMAP.
    """
    grouped = results_multi.groupby('dim')

    means_clusters = grouped['n_clusters'].mean()
    stds_clusters = grouped['n_clusters'].std()

    means_noise = grouped['noise_frac'].mean()
    stds_noise = grouped['noise_frac'].std()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Numero cluster (asse sinistro)
    color1 = 'tab:blue'
    ax1.errorbar(means_clusters.index, means_clusters.values,
                 yerr=stds_clusters.values, fmt='-o', capsize=5,
                 color=color1, label="N. cluster")
    ax1.set_xlabel("Dimensioni UMAP")
    ax1.set_ylabel("Numero cluster", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Frazione rumore (asse destro)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.errorbar(means_noise.index, means_noise.values,
                 yerr=stds_noise.values, fmt='-s', capsize=5,
                 color=color2, label="Rumore")
    ax2.set_ylabel("Frazione rumore", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Cluster e rumore (media ± std su ripetizioni)")
    fig.tight_layout()
    plt.savefig("num_clusters.png")
    plt.show()




if __name__ == "__main__":
    emb_file = 'output/val_embeddings/val_all_merged.npz'
    # 1. Caricamento del file `.npz` contenente l'array di embedding
    data = np.load(emb_file)           # Carica il file .npz
    embeddings = data['embeddings']            # Estrae l'array con chiave "embeddings"

    # Fai 5 ripetizioni con seed diversi
    lista_dim = range(5, 550, 100)
    results_multi = repeat_clustering_evaluation(
        embeddings,
        dims=lista_dim,
        n_repeats=10,
        pca_components=1000,
        hdbscan_kwargs=dict(min_cluster_size=15)
    )

    # Media e deviazione standard per ogni dimensione
    summary = results_multi.groupby('dim').agg(['mean','std'])
    print(summary[['silhouette','davies_bouldin','calinski_harabasz','n_clusters','noise_frac']])

    plot_metrics_with_errorbars(results_multi)

    plot_cluster_stats_with_errorbars(results_multi)

