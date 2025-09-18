# View_test_tiles.ipynb

## Overview
Notebook complesso dedicato all’ispezione del dataset di validazione/test, alla generazione di GIF del Mediterraneo, alla verifica delle predizioni e alla creazione di animazioni per singoli cicloni. Integra funzioni di `dataset.build_dataset`, `view_test_tiles`, `model_analysis` e `BuildDataset` per offrire una vista end-to-end su dati, modello e output visivi.

## Caricamento dati e metadati
- Creazione del dataloader di validation con `build_dataset` e `DataLoader` PyTorch.
- Lettura di `val_supervised.csv` per estrarre timestamp e combinazione con `all_data_wo_overlap_tiles.csv`, arricchendo le righe con lat/lon, offset e sorgente (`CL7`).
- Disponibilità di una versione estesa (`all_data_all_methods_tracks_complete.csv`) con colonne in formato `object` per gestire differenti fonti di etichette.

## Modello e predizioni
- Preparazione del modello VideoMAE via `create_model`, caricando il checkpoint configurato in `args` e abilitando la modalità test (`get_prediction = True`).
- Funzioni di supporto in `model_analysis` per estrarre label e predizioni (`predict_label`, `get_path_pred_label`, `create_df_predictions`).

## Visualizzazione delle tile
- Funzioni custom (`get_time_from_row`, `draw_tiles_and_center`, `create_gif_pil`, `display_video_clip`, ecc.) per rendere i frame delle tile, disegnare il centro del ciclone, aggiungere timestamp e generare GIF.
- Analisi approfondita dei dataset: filtri per tile specifiche, conteggio delle combinazioni offset, esplorazione dei nomi file, ecc.

## Creazione di animazioni dei cicloni
- Selezione di ID o nomi di cicloni (es. `Apollo`, `Blas`, `Daniel`, `Helios`, `Juliette`) dal dataset `more_medicanes.csv`.
- Uso di `BuildDataset.create_master_df_short` per ricostruire i frame di ogni ciclone e generare animazioni con `make_animation_parallel_ffmpeg`.
- Salvataggio dei frame di debug (`save_frames_parallel`) e render frame-by-frame con `compose_image`.

## Debug e strumenti aggiuntivi
- Estrazione di intervalli temporali dalle tracce (`get_intervals_in_tracks_df`), caricamento delle immagini corrispondenti e verifica della completezza del dataset.
- Numerosi block di debug per ispezionare DataFrame, contare gruppi e assicurarsi che le unioni tra dataset e predizioni siano corrette.

## Output
- GIF e video MP4 per cicli specifici, frame salvati su disco (`anim_frames_*`).
- DataFrame di supporto con predizioni e metadati arricchiti.

## Note operative
- Il notebook manipola path relativi e assoluti; verificare `output_dir`, `input_dir` e `ffmpeg` in PATH prima dell’esecuzione.
- Alcune sezioni sono pensate per esecuzioni lunghe; conviene lavorare su subset o cicloni specifici per debug rapido.
