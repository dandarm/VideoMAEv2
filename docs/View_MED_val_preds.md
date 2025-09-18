# View_MED_val_preds.ipynb

## Overview
Documento di analisi per la fase di validazione: abbina le predizioni del classificatore sui video di validation con le tile originali, arricchisce i dati con informazioni di nuvolosità e genera visualizzazioni/animazioni del Mediterraneo completo. Il notebook mostra come filtrare i video in base all’indice di copertura nuvolosa e come espandere le predizioni dalle sequenze aggregate ai singoli frame.

## Preparazione del dataset di validazione
- Caricamento degli argomenti (`prepare_finetuning_args`) e forzatura del device a CPU per la visualizzazione.
- Creazione del `DataManager` in modalità valutazione sul CSV di validation (`args.val_path`).
- Opzione per filtrare i video in validazione in base alla nuvolosità: lettura del CSV, calcolo della colonna `avg_cloud_idx` con `calc_avg_cld_idx` e salvataggio dei path conservati in `kept`.
- Ricostruzione del dataloader utilizzando eventuali CSV filtrati.

## Predizioni
- Configurazione del modello VideoMAE (`create_model`) con checkpoint dedicato (es. `checkpoint-best-90_25lug25.pth`).
- Raccolta dei percorsi, delle predizioni e delle label vere tramite `get_only_labels` e costruzione del `DataFrame` con `create_df_predictions`.
- Possibilità alternativa: caricare direttamente un CSV precomputato (`output/inference_predictions.csv`).

## Unione con il builder di validation
- Creazione di un `BuildDataset` di validazione tramite la funzione helper `make_validation_data_builder_from_manos_tracks`, che riproduce le sequenze video per i periodi indicati in `medicanes_new_windows.csv`.
- Merge tra `df_video` del builder e il DataFrame delle predizioni; ogni riga rappresenta un video tile con la lista dei frame originali (`orig_paths`) e informazioni di contesto (`neighboring`, coordinate offset).

## Espansione al livello immagine
- Iterazione su ogni riga per associare la predizione aggregata ai singoli frame (`records.append({...})`), generando `df_mapping` con colonne `path`, `predictions`, `tmp_label`, `tile_offset_x/y`, `neighboring`.
- Join con `master_df` per recuperare coordinate geografiche e altri metadati; sostituzione della colonna `label` con quella temporanea.
- Gestione delle tile mancanti: calcolo degli offset teorici (`calc_tile_offsets`) e funzione `expand_group` che riempie eventuali lacune mantenendo i valori costanti del gruppo (timestamp, prediction, ecc.).

## Visualizzazione
- Creazione di animazioni tramite `make_animation_parallel_ffmpeg` o `render_and_save_frame`, che generano file MP4 o singoli frame con overlay delle predizioni.
- Conteggio dei gruppi per debug e ispezione manuale (`groupby('path')`).

## Output
- DataFrame arricchiti (`df_mapping`, `df_data_merg`).
- Video e frame renderizzati con le predizioni di validation.

## Note operative
- Assicurarsi che `args.cloudy` sia coerente con la presenza/assenza dell’indice di nuvolosità nel CSV; in caso contrario la sezione di filtro viene saltata.
- Le funzioni `make_animation_parallel_ffmpeg` e `render_and_save_frame` richiedono che `ffmpeg` sia disponibile in PATH; il notebook lo aggiunge manualmente.
- Il flusso è facilmente adattabile a set di test, cambiando il CSV e ripetendo il merge.
