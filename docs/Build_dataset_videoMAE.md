# Build_dataset_videoMAE.ipynb

## Overview
Questo notebook raccoglie l'intero flusso di generazione dei dataset video, sia non supervisionati sia supervisionati, a partire dai prodotti Airmass RGB e dalle tracce dei cicloni fornite da Manos. Vengono richiamate numerose utility del progetto (`dataset.build_dataset`, `dataset.data_manager`, `view_test_tiles`, `arguments`), con l'obiettivo di costruire i DataFrame di base, suddividere in tile le immagini, assemblare sequenze temporali da 16 frame e salvare su disco sia i CSV descrittivi sia le cartelle con i video pronti per l'addestramento.

## Setup e costanti
Le prime celle caricano gli automatismi di ricarica (`%autoreload`), le librerie standard (Pathlib, Pandas, NumPy, datetime, Basemap) e aggiungono percorsi ausiliari a `sys.path`. Vengono quindi importate tutte le funzioni di supporto da `dataset.build_dataset` (per il calcolo delle scale in pixel, il raggruppamento per offset, la creazione e il salvataggio delle tile, la costruzione dei CSV finali, ecc.) e da `medicane_utils`. Si inizializzano tre directory chiave:
- `input_dir`: cartella con le immagini complete del Mediterraneo.
- `output_dir`: destinazione dei video supervisionati.
- `unsup_output_dir`: destinazione delle sequenze non supervisionate.

## Costruzione dataset non supervisionato
1. Caricamento del CSV `all_data_unsup.csv`, tipizzazione delle colonne critiche (`path`, `tile_offset_x/y`, `datetime`) e rimozione della colonna di indice.
2. Riduzione del campione (prima di eventuali prove) e costruzione dei gruppi temporali con `get_gruppi_date`, che restituisce liste di righe consecutive per ogni intervallo continuo di immagini.
3. Per ogni gruppo temporale: raggruppamento per offset di tile (`group_df_by_offsets`), costruzione delle sequenze video (`create_tile_videos` con `supervised=False`), salvataggio su disco (`create_and_save_tile_from_complete_df`) e accumulo in `all_videos`.
4. Concatenazione di tutti i DataFrame video e generazione del CSV finale con `create_final_df_csv`; il file salvato (`train_960_UNsupervised.csv`) contiene un record per ogni video con la lista dei frame originali, le coordinate del tile e metadati temporali.

## Flusso supervisionato
1. Caricamento della tabella `manos_CL7_pixel.csv` che contiene le tracce dei cicloni (coordinati sia in lat/lon sia in pixel) e generazione degli offset di tile con `calc_tile_offsets`.
2. Richiamo di `labeled_tiles_from_metadatafiles_maxfast` per etichettare ogni tile delle immagini complete rispetto alla presenza del ciclone. Il DataFrame risultante viene salvato come `all_data_CL7_tracks_complete_fast.csv` e rappresenta il “master” delle osservazioni supervisionate.
3. Analisi del master: filtraggio di righe positive, raggruppamento temporale (`get_gruppi_date`), verifica della lunghezza dei blocchi, suddivisione in gruppi di 20 tile per timestamp e prove di bilanciamento tra esempi positivi e negativi.
4. Costruzione dei video supervisionati bilanciati: per specifici periodi temporali si invoca `balance_time_group`, si concatenano i DataFrame ottenuti, si suddividono in train/test e si richiama `create_final_df_csv` per generare `train_supervised.csv` e `val_supervised.csv`. In aggiunta vengono esplorati approcci alternativi usando la classe `BuildDataset` (con metodi `load_master_df`, `get_sequential_periods`, `make_df_video`, `create_final_df_csv`).
5. Ulteriori esperimenti: ripartizione secondo ratio temporali manuali, salvataggio di dataset multipli (ad esempio `train_dataset_1954.csv`, `test_dataset_2802.csv`), bilanciamento custom tramite la funzione `balance_df_by_label` e split 70/20/10 con esportazione di `train_supervised.csv`, `val_supervised.csv` e `test_supervised.csv`.

## Validazioni e controlli
Il notebook dedica varie sezioni a controllare la coerenza dei dati:
- Visualizzazione dei gruppi temporali e del numero di tile per timestamp.
- Verifica delle fusioni tra DataFrame di predizioni e dataset video (`merge` su `path`).
- Test di coerenza con `DataLoader` PyTorch: costruzione del dataloader di validazione tramite `build_dataset`, raccolta di percorsi, label e predizioni (`get_only_labels`, `create_df_predictions`), merge con i dataset video e ordinamento per data.
- Analisi dello stato delle cartelle su disco (filtraggio di path inesistenti, conteggio di esempi positivi/negativi) e bilanciamento randomizzato tramite sampling.

## Output principali
- Cartelle contenenti i video supervisionati e non supervisionati, organizzate per tile e periodo temporale.
- CSV: `train_960_UNsupervised.csv`, `all_data_CL7_tracks_complete_fast.csv`, `train_supervised.csv`, `val_supervised.csv`, `test_supervised.csv`, oltre ai dataset intermedi creati per esperimenti specifici.

## Note operative
- Tutte le funzioni delegate (dal modulo `dataset.build_dataset`) assumono che le immagini siano già scaricate e disponibili in `../fromgcloud` e che le tracce di Manos siano coerenti con i timestamp delle immagini.
- Gli step di bilanciamento e filtraggio sono modulari: il notebook mostra diverse strategie (sampling casuale, selezione di periodi storici, filtri su nuvolosità) che possono essere riprese singolarmente.
- L’utilizzo della classe `BuildDataset` consente di replicare l’intero processo via codice “production ready”, mentre le celle iniziali offrono una versione esplicita adatta a debugging e analisi.
