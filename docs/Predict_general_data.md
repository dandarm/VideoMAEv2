# Predict_general_data.ipynb

## Overview
Il notebook documenta come costruire un set di inferenza a partire dal master dataframe supervisionato, eseguire il modello di classificazione VideoMAE su tali dati e trasformare le predizioni video in etichette per singoli frame/tile, fino alla creazione di animazioni che rappresentano il Mediterraneo completo. Include inoltre vari scenari di riutilizzo del modello su dataset già salvati.

## Preparazione del dataset di inferenza
1. Caricamento del master dataframe tramite `BuildDataset(type='SUPERVISED', master_df_path=file_master_df)` e `load_master_df()`.
2. Filtro temporale (ad es. solo eventi successivi a settembre 2023) con creazione di un sotto-DataFrame (`df_2023`).
3. Generazione dei video (`make_df_video`) senza bilanciamento per ottenere la struttura attesa dal classificatore; salvataggio dei riferimenti in `general_inference_set.csv`.
4. (Opzionale) visualizzazione rapida dei frame temporali con `create_labeled_images_with_tiles`.

## Esecuzione del modello
1. Configurazione degli argomenti (`prepare_finetuning_args`) in modalità test (`args.test_mode = True`) e assegnazione del percorso al CSV (`args.test_path`).
2. Creazione del `DataManager` in modalità classificazione e costruzione del `DataLoader`.
3. Inizializzazione del modello con `create_model` caricando il checkpoint desiderato (`args.init_ckpt`).
4. Ottenimento delle predizioni tramite `get_path_pred_label` e costruzione del DataFrame aggregato con `create_df_predictions`.

## Mappatura delle predizioni alle immagini originali
- Merge tra il DataFrame video (`data.df_video`) e le predizioni su `path` per arricchire ogni sequenza con la confidenza stimata.
- Conversione delle predizioni da livello video a livello immagine con `video_pred_2_img_pred`, che scompone la sequenza in frame/tile e produce la colonna `tmp_label`.
- (Opzione) riduzione della frequenza temporale (`sub_select_frequency`) per generare un sottoinsieme più leggero.
- Join con il master dataframe originale per recuperare coordinate, timestamp e label effettiva (`merge(..., how='left').drop(columns='label').rename(columns={'tmp_label':'label'})`).

## Creazione di video MED
- Utilizzo delle funzioni di rendering (`get_writer4animation`, `make_animation`) per comporre clip MP4 complete del Mediterraneo, utilizzando i frame arricchiti con predizioni.
- Supporto a dataset pre-esistenti: caricamento di CSV già generati (`train_CL10_148.csv`, ecc.), ripetizione della pipeline di predizione e gestione di dataset con tile mancanti (espansione via `expand_group` e offset calcolati con `calc_tile_offsets`).

## Sezioni aggiuntive
- Conversione delle predizioni per dataset già bilanciati o creati precedentemente, mantenendo coerenza con i DataFrame `train_m.master_df`.
- Considerazioni su come gestire tile mancanti: definizione di `df_offsets` e uso di `groupby` con `expand_group` per ricostruire tutte le posizioni.
- Uso di `make_animation` o `make_animation_parallel_ffmpeg` per salvare clip finali (es. `daniel_prediction_1.mp4`).

## Output principali
- CSV con predizioni video (`df_predictions`) e mapping immagine (`df_data_merg`).
- Video animati con sovrapposizione della probabilità stimata dal modello.

## Note operative
- Il notebook presume che le tile siano già state salvate in `output_dir` dalla pipeline di costruzione dataset.
- Per dataset storici è necessario aggiornare `args.test_path` e gli offset coerenti con il metodo di tiling usato.
- La sezione “old” contiene controlli su master dataset storici (`master_data_2020_wID.csv`) utili se si deve verificare la continuità fra versioni precedenti.
