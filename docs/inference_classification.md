## Scopo del modulo
Script di inferenza per modelli di classificazione allenati con VideoMAE/TIMM. Configura l’ambiente distribuito, carica checkpoint pre-addestrati e offre flussi multipli: predizione standard, raccolta logits o embedding, con gestione automatica dell’aggregazione multi-rank e salvataggio dei risultati in CSV/NPZ.

## Flusso ad alto livello
```text
1. Parsare argomenti CLI ed espandere la configurazione tramite `prepare_finetuning_args`.
2. Eventualmente sostituire `val_path` e `init_ckpt` con input CLI.
3. Inizializzare seed, CUDA e risorse distribuite (rank, device, output/log dir).
4. Creare `DataManager` di validazione e relativo DataLoader supervisionato.
5. Istanziarne il modello TIMM e wrappare in DDP se necessario; impostare `eval()`.
6. In base alle flag: scegliere tra raccolta logits, embedding o predizioni standard.
7. Eseguire `validation_one_epoch_*` appropriata per ottenere metriche e raccolte dati.
8. Coordinare sincronizzazione di rank, merge di shard NPZ e pulizia file temporanei.
9. Generare DataFrame tramite `create_df_predictions` e salvare CSV finali (solo rank 0).
10. Loggare metriche di inferenza su `inference_metrics.txt` e stampare durata totale.
```

## API principali
- `all_seeds()`: rende deterministici random/cuda; senza ritorno.
- `launch_inference_classification(terminal_args)`: workflow completo di inferenza; `terminal_args` deve contenere `on`, `inference_model`, opzionalmente flag per logits/embedding; ritorna `None`.
- `__main__`: parser CLI con opzioni per CSV input, checkpoint, nomi output e flag `--get_logits/--get_embeddings` che invoca `launch_inference_classification`.

## Dipendenze
- `arguments.prepare_finetuning_args` per configurazione coerente col training.
- `dataset.data_manager.DataManager` per DataLoader di validazione.
- `engine_for_finetuning.validation_one_epoch_{collect, collect_logits, collect_embeddings}` per esecuzione forward e raccolta dati.
- `model_analysis` per costruzione DataFrame e merge/cleanup file NPZ.
- `utils` per risorse distribuite e scaler AMP.

## I/O e formati
- Input primario: CSV o dataset referenziati da `args.val_path` (formato non determinato dal codice analizzato).
- Checkpoint caricato via `args.init_ckpt` (Torch `state_dict`).
- Output: CSV con colonne determinate da `create_df_predictions` (tipicamente path, predizione, label), file NPZ per logits/embedding con schema `prefix_rankX_batchY_*.npz` e merge globale opzionale.
- Log: file `inference_metrics.txt` in JSON lines, directory `log_dir` per eventuali writer esterni.

## Punti di estensione/assunzioni
- Necessita che il checkpoint contenga strutture compatibili col modello TIMM scelto; mismatch generano errori nella `load_state_dict` interna (`auto_load_model`).
- Flag logits/embedding sono mutuamente esclusivi e generano `ValueError` se attivati insieme.
- Merge/cleanup NPZ assume permessi di scrittura condivisi e filesystem visibile da tutti i rank.
- Metriche considerano solo bilanciamento (`bal_acc`) fornito dalle funzioni engine; ulteriori metriche vanno implementate separatamente.

## Copertura
- ✓ `all_seeds`
- ✓ `launch_inference_classification`
- ✓ `__main__`
