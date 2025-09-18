# Experiment_dataset.ipynb

## Overview
Notebook sperimentale dedicato alla manipolazione del master dataframe supervisionato, al relabeling in funzione della distanza temporale dai cicloni e alla costruzione di split train/test/validation per diverse configurazioni (CL10, medicanes estesi, anni completi). Funziona come laboratorio per testare funzioni di `BuildDataset` e `dataset.dataset_labeling_study`.

## Relabeling basato sul tempo
- Caricamento del master `all_data_CL7_tracks_complete_fast.csv` tramite `BuildDataset`.
- Invocazione di `calc_delta_time` per calcolare l’intervallo temporale rispetto al ciclone attivo.
- Uso di `aggiorna_label_distanza_temporale` per impostare un’etichetta ausiliaria che marca come `-1` i frame oltre 12 ore dal centro del ciclone, e sostituzione della colonna `label` con tale valore (dopo aver escluso le righe `-1`).
- Generazione del DataFrame video con `make_df_video`, bilanciamento opzionale, split temporale 70/30 tra train e test.

## Split basati sulle tracce CL10
- Import delle utility per creare intervalli e suddividerli (`train_test_cyclones_num_split`, `get_intervals_in_tracks_df`).
- Caricamento di `manos_CL10_pixel.csv`, divisione degli ID ciclone in train/test e valutazione della durata totale per ogni split.
- Invocazione di `BuildDataset` per costruire dataset supervisionati per entrambi gli split, con salvataggio dei CSV (`train_CL10`, `test_CL10`).

## Dataset di soli medicanes
- Lettura di `manos_medicanes_only.csv` e `more_medicanes_time_updated.csv`.
- Normalizzazione della colonna `id_final` assegnando ID numerici anche ai cicloni nominati (via `utils.str2num`).
- Filtraggio delle righe fuori dal range temporale dichiarato, salvataggio in `medicanes_new_windows.csv` e split 70/15/15 con `get_train_test_validation_df`.
- Creazione dei dataset `train_manos_w`, `test_manos_w`, `val_manos_w` tramite `BuildDataset` passando i rispettivi DataFrame delle tracce.

## Output
- DataFrame video (spesso con bilanciamento) e CSV associati: `train_CL10.csv`, `test_CL10.csv`, `train_manos_w.csv`, `test_manos_w.csv`, ecc.
- CSV aggiornati con nuove finestre temporali (`medicanes_new_windows.csv`).

## Note
- Il notebook mostra come riutilizzare `BuildDataset` sia in modalità “da CSV già pronto” sia passando direttamente le tracce e il percorso delle immagini.
- L’approccio di relabeling consente di sperimentare soglie temporali differenti cambiando il parametro `soglia` della funzione; gli step di verifica (maschere sul DataFrame, visualizzazione di intervalli) aiutano a controllare la correttezza.
