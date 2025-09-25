## Scopo del modulo
Entry point per l'addestramento del modello di tracking del centro ciclone. Configura ambiente distribuito, costruisce DataLoader specializzati tramite `DataManager`, istanzia la rete di regressione da `models.tracking_model` e gestisce il ciclo train/val/logging con checkpointing del miglior modello.

## Flusso ad alto livello
```text
1. Parsare CLI (`parse_args`) e costruire configurazione estesa tramite `prepare_tracking_args`.
2. Inizializzare seed random, backend CUDA e contesto distribuito (rank, device, log dir).
3. Creare tre `DataManager` (train/test/val opzionale) e ottenere DataLoader di tracking.
4. Costruire il modello con `create_tracking_model`, trasferirlo sul device e wrappare in DDP.
5. Inizializzare loss `MSELoss` e ottimizzatore configurato da `create_optimizer`.
6. Iterare sulle epoche: eseguire `train_one_epoch` sul loader train.
7. Valutare con `evaluate` su test (e valida se presente), aggiornando `best_loss`.
8. Salvare checkpoint migliore nella cartella `output_dir` (solo rank 0).
9. Serializzare metriche per epoca in `log.txt` (JSON lines).
10. Stampare tempo totale di training formattato HH:MM:SS.
```

## API principali
- `launch_tracking(terminal_args)`: esegue l'intero workflow di training; `terminal_args` deve fornire almeno `on`; ritorna `None`.
- `parse_args()`: parser CLI minimale (`--on`) che restituisce `Namespace` utilizzato da `launch_tracking`.
- `__main__`: entry point che richiama `launch_tracking(parse_args())`.

## Dipendenze
- `arguments.prepare_tracking_args` per combinare argomenti di configurazione.
- `dataset.data_manager.DataManager` per DataLoader specifici di tracking.
- `models.tracking_model.create_tracking_model` (e opzionalmente `RegressionHead`) per costruire la rete.
- `engine_for_tracking.train_one_epoch/evaluate` per orchestrare cicli di training e valutazione.
- `utils` per gestione risorse distribuite e logging coordinato.

## I/O e formati
- Input: dataset di tracking strutturati secondo le aspettative di `DataManager.get_tracking_dataloader` (formato non determinato dal codice analizzato).
- Output: checkpoint `checkpoint-tracking-best.pth` in `args.output_dir` con stato modello/optimizer/epoch, log JSON lines in `log.txt`, stampe console informative.
- Non produce CSV; eventuali metriche aggiuntive vanno integrate nell'engine.

## Punti di estensione/assunzioni
- Il modello è trattato come regressore (verifica `RegressionHead`); eventuali head alternative richiedono `evaluate` compatibile.
- Salvataggio best parte dopo `start_epoch_for_saving_best_ckpt`; ensure configurazione adatta.
- Richiede che `DataManager` gestisca dataset di tracking; errori in configurazione percorsi generano eccezioni propagate.
- Attualmente usa loss MSE fissa; per obiettivi classificazione/ibridi servono modifiche interne.

## Copertura
- ✓ `launch_tracking`
- ✓ `parse_args`
- ✓ `__main__`
