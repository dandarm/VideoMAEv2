# Build_dataset_videoMAE.ipynb

## Overview
Questo notebook raccoglie l'intero flusso di generazione dei dataset video, sia non supervisionati sia supervisionati, a partire dai prodotti Airmass RGB e dalle tracce dei cicloni fornite da Manos. Vengono richiamate numerose utility del progetto (`dataset.build_dataset`, `dataset.data_manager`, `view_test_tiles`, `arguments`), con l'obiettivo di costruire i DataFrame di base, suddividere in tile le immagini, assemblare sequenze temporali da 16 frame e salvare su disco sia i CSV descrittivi sia le cartelle con i video pronti per l'addestramento.

## Load images
- la funzione `load_all_images` carica tutti i file dalla `input_dir` in una lista `metadata` di `PosixPath` e `datetime`

## Costruzione dataset non supervisionato  (TODO: da riguardare)
1. Caricamento del CSV `all_data_unsup.csv`, tipizzazione delle colonne critiche (`path`, `tile_offset_x/y`, `datetime`) e rimozione della colonna di indice.
2. Riduzione del campione (prima di eventuali prove) e costruzione dei gruppi temporali con `get_gruppi_date`, che restituisce liste di righe consecutive per ogni intervallo continuo di immagini.
3. Per ogni gruppo temporale: raggruppamento per offset di tile (`group_df_by_offsets`), costruzione delle sequenze video (`create_tile_videos` con `supervised=False`), salvataggio su disco (`create_and_save_tile_from_complete_df`) e accumulo in `all_videos`.
4. Concatenazione di tutti i DataFrame video e generazione del CSV finale con `create_final_df_csv`; il file salvato (`train_960_UNsupervised.csv`) contiene un record per ogni video con la lista dei frame originali, le coordinate del tile e metadati temporali.

## Flusso supervisionato
1. Caricamento di un dataframe di tracce Manos (es. `manos_CL7_pixel.csv`) che contiene le coordinate sia in lat/lon sia in pixel, e generazione degli offset di tile con `calc_tile_offsets`(in funzione dello stride).
2. Richiamo di `labeled_tiles_from_metadatafiles_maxfast` per etichettare ogni tile delle immagini complete rispetto alla presenza del ciclone. Il DataFrame risultante viene salvato come `all_data_CL7_tracks_complete_fast.csv` e rappresenta il “master” delle osservazioni supervisionate, perché contiene: path_immagine, offsets_x_y, coordinate_pixel, label.
3. Analisi del master: raggruppamento per segmenti temporali contigui (`get_gruppi_date`), verifica della lunghezza dei blocchi, suddivisione in gruppi di tile con lo stesso ofsset (`group_df_by_offsets`). 
4. Costruzione del dataframe dei video etichettati con `create_tile_videos`.
- Prove di bilanciamento tra esempi positivi e negativi (`balance_time_group`), 
- Si concatenano i DataFrame ottenuti per gruppi temporali diversi, si suddividono in train/test e si richiama `create_final_df_csv` per generare `train_supervised.csv`, `test_supervised.csv`  e `val_supervised.csv`, necessari per istanziare il DataLoader.  Questi 3 file csv sono l'output finale del flusso, contengono i path delle folder con i frame, che costituiscono i sample video del dataset, insieme con il relativo salvataggio delle folder.

5. In aggiunta viene mostrato lo stesso flusso usando la classe `BuildDataset` (con metodi `create_master_df` e varianti, o `load_master_df`, `get_sequential_periods`, `make_df_video`, `create_final_df_csv`),
e usando il metodo `get_data_ready`. Quindi l’utilizzo della classe `BuildDataset` consente di replicare l’intero processo via codice “production ready”, mentre le celle iniziali offrono una versione esplicita adatta a debugging e analisi.


6. Creazione dataset di un anno intero: usando il medoto `get_data_ready_full_year` e `create_master_df_year` che non limita il caricamento ai soli intervalli temporali attorno ai cicloni come contenuto nel df_tracks di Manos.


## Output principali (TODO: controllare quali csv si possono eliminare, non servono tutti, mantenere quelli più generici, in base a scelte sui training da ancora effettuare o quelli che hanno dato risultati migliori)
- Cartelle contenenti i video supervisionati e non supervisionati, organizzate per tile e periodo temporale.
- CSV: `train_960_UNsupervised.csv`, `all_data_CL7_tracks_complete_fast.csv`, `train_supervised.csv`, `val_supervised.csv`, `test_supervised.csv`, oltre ai dataset intermedi creati per esperimenti specifici.

## Note operative
- Tutte le funzioni delegate (dal modulo `dataset.build_dataset`) assumono che le immagini siano già scaricate e disponibili in `../fromgcloud` e che le tracce di Manos siano coerenti con i timestamp delle immagini.
- Gli step di bilanciamento e filtraggio sono modulari: il notebook mostra diverse strategie (sampling casuale, selezione di periodi storici, filtri su nuvolosità) che possono essere riprese singolarmente.

