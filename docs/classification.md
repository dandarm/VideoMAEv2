## Scopo del modulo
Modulo di lancio per il fine-tuning supervisionato di modelli VideoMAE/TIMM su task di classificazione cyclone/no-cyclone. Gestisce inizializzazione distribuita, orchestrazione dei DataLoader tramite `DataManager`, costruzione del modello, training con valutazione periodica e salvataggio dei checkpoint migliori. Centralizza logging e scalatura automatica degli iperparametri in funzione del world size, rendendo riutilizzabile lo script su cluster multi-GPU.

## Flusso ad alto livello
```text
1. Parsare CLI e costruire Args tramite `prepare_finetuning_args`.
2. Inizializzare semi, backend CUDA, risorse e processo distributo.
3. Istanziate tre `DataManager` (train/test/val) e creare i rispettivi DataLoader di classificazione.
4. Facoltativamente calcolare pesi di classe dal dataset di training.
5. Creare il modello TIMM, spostarlo sul device, wrappare in DDP se richiesto.
6. Inizializzare ottimizzatore, scaler AMP e scheduler coseno per LR/weight decay.
7. Opzionalmente riprendere stato da checkpoint (`auto_load_model`).
8. Per ogni epoca: eseguire `train_one_epoch`, quindi validare a intervalli `testing_epochs`.
9. Aggiornare best checkpoint e loggare statistiche su file JSON lines.
10. Al termine, riportare tempo totale e rilasciare risorse.
```

## API principali
- `all_seeds()`: imposta determinismo per numpy/torch/CUDA; nessun input, effetti collaterali su stato globale; ritorna `None`.
- `launch_finetuning_classification(terminal_args)`: entry point principale; accetta `argparse.Namespace` con almeno `on`; costruisce training loop completo; nessun valore di ritorno.
- `__main__`: parser CLI minimale (`--on`) che richiama `launch_finetuning_classification`.

## Dipendenze
- `arguments.prepare_finetuning_args` per caricare le configurazioni per il finetuning
- `dataset.data_manager.DataManager` per creazione dei dataloader supervisionati.
- `timm.create_model`, `optim_factory.create_optimizer`, `engine_for_finetuning.train_one_epoch/validation_one_epoch` per caricare il modello , l'optimizer (tipicamente AdamW), e il ciclo di addestramento.
- `utils` per gestione risorse, scheduler e checkpointing.

## I/O e formati
- Input: CSV/dataset definiti in `Args` consumati da `DataManager` (formato non determinato dal codice analizzato).
- Output: log `log.txt` in `args.output_dir` (JSON lines), eventuale checkpoint `checkpoint-best.pth`, stampe console riepilogative.
- Dati intermedi: se abilitato `use_class_weight`, calcolo di pesi (tensor float su device).

## Punti di estensione/assunzioni
- Si assume che `Args` definisca percorsi, dimensioni batch e iperparametri coerenti; invalidi generano eccezioni a runtime.
- Per attivare mixup/cutmix occorre configurare gli argomenti; lo script forza `mixup_active = False` dopo la stampa (comportamento hard-coded).
- La loss con pesi di classe è commentata: va sbloccata manualmente se necessaria.
- Checkpointing "best" parte da `start_epoch_for_saving_best_ckpt` per evitare early spikes.

## Copertura
- ✓ `all_seeds`
- ✓ `launch_finetuning_classification`
- ✓ `__main__`
