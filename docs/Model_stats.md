# Model_stats.ipynb

## Overview
Notebook focalizzato sull’analisi quantitativa delle prestazioni del classificatore VideoMAE. Copre la creazione di un set di validazione a partire dal master dataset (con filtri temporali e di nuvolosità), l’esecuzione del modello per generare predizioni, la visualizzazione dei risultati e il calcolo di metriche come accuracy, FPR, FNR, POD, FAR, e confusion matrix. Include anche funzioni per esplorare l’andamento delle metriche nel tempo e per label specifiche.

## Preparazione del dataset di valutazione
### Vengono rieseguiti i codici per la creazione del dataset TODO: semplificare e sostituire con i metodi di BuildDataset
- Caricamento del master (`all_data_CL7_tracks_complete_fast.csv`) con `BuildDataset`, calcolo di `delta_time` e uso di `aggiorna_label_distanza_temporale` per escludere frame oltre la soglia di 12 ore.
- Split train/test basato sull’ordinamento temporale, con salvataggio delle tile video di test (`create_and_save_tile_from_complete_df`).
- Calcolo opzionale dell’indice medio di nuvolosità per ogni video (`calc_avg_cld_idx`) e filtraggio dei video “cloudy”, generando dataset dedicati.
- Normalizzazione dei path (rimozione del prefisso directory) e salvataggio del CSV finale (`val_set.csv`).

### Caricamento e filtro del CSV di validation
- Lettura del CSV attraverso `pd.read_csv`, possibilità di calcolare nuovamente l’indice di nuvolosità se l’opzione `args.cloudy` è abilitata.
- Configurazione del `DataManager` in modalità test con `args.test_path = "val_quick_test.csv"` (o dataset equivalenti).

## Inferenza del modello
- Preparazione del modello con `create_model` e caricamento del checkpoint.
- Raccolta delle predizioni tramite funzioni di `model_analysis` (`get_path_pred_label`, `create_df_predictions`). O nel caso più semplice: ho già raccolto le predizioni con lo script `inference_classification.py` e salvate in `output/inference_predictions.csv`
- Nel caso in cui sono state raccolte anche i logit: viene creato il `df_predictions` a partire dal file .npz dei logit


## Calcolo delle metriche
- Uso di `evaluate_binary_classifier` per generare accuracy, precision, recall, FPR, FNR, ecc.
- Visualizzazione della confusion matrix con `plot_confusion_and_results`.
- Ripetizione del calcolo su dataset diversi: anno intero 2023 (`inference_predictions_entire_year.csv`), dataset bilanciato (`..._balanced.csv`), dataset sbilanciato (`..._UNbalanced.csv`).

## Analisi per label specifiche delta_time
- Funzioni `confusion_counts_per_label` e `plot_metrics_over_time` per calcolare true positive/false positive/false negative aggregati per colonne di label semantiche e mostrarne l'andamento temporale.


## Output
- CSV arricchiti con predizioni, confusion matrix salvate su disco, grafici delle metriche per scenario.

## Note operative
- Il notebook utilizza molte funzioni definite altrove (es. `prepare_finetuning_args`, `BuildDataset`, `model_analysis`); assicurarsi che i moduli siano importabili.
- Le metriche per singola label richiedono che il DataFrame delle predizioni includa colonne label per ogni delta time.
