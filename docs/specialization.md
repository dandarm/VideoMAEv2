## Scopo del modulo
Script di addestramento per la fase di "specialization" (fine-tuning auto-supervisionato) del modello VideoMAE. Configura ambiente distribuito, carica un modello pre-addestrato tramite `get_model`, costruisce dataset dedicati con finestre e patch coerenti e gestisce l'intero ciclo di training/test con logging su TensorBoard e checkpointing periodico.

## SLURM - OPENMPI traning
Utilizzare lo script sbatch_job.sh per lanciare questo training di su un cluster di nodi distribuiti (TODO: modificare o creare nuovo script)

## Flusso ad alto livello
```text
1. Ricevere argomenti macchina tramite `prepare_args` e recuperarne iperparametri e percorsi.
2. Inizializzare risorse distribuite (rank, device, DDP) e creare directory di log.
3. Caricare il modello MAE con `get_model`, spostarlo sul device e wrappare in DDP.
4. Costruire dataset e dataloader di training/test con `get_dataset_dataloader` e `get_dataloader` (patch size derivata dal modello).
5. Calcolare numero di step per epoca e scalare learning rate/warmup/weight decay in funzione del world size.
6. Creare ottimizzatore e scaler AMP, quindi predisporre scheduler coseno per LR/WD.
7. Riprendere eventuali checkpoint (`auto_load_model`) e inizializzare logger TensorBoard.
8. Per ogni epoca: sincronizzare sampler, eseguire `train_one_epoch` con scheduler passo-passo e salvare checkpoint secondo `save_ckpt_freq`.
9. A intervalli `testing_epochs`, lanciare `test` sul set dedicato e appendere log in JSON lines.
10. Al termine, stampare tempo totale di addestramento.
```

## API principali
- `launch_specialization_training(terminal_args)`: orchestratore principale; `terminal_args` deve contenere almeno `on`; ritorna `None` dopo aver completato training/test.
- `__main__`: invoca `launch_specialization_training()` senza parser; nel codice corrente solleva `TypeError` perché manca l'argomento richiesto (da correggere esternamente).

## Dipendenze
- `arguments.prepare_args` per configurazione base.
- `utils` per risorse distribuite, scheduler, checkpoint, logging TensorBoard.
- `optim_factory.create_optimizer` per costruire l'ottimizzatore coerente con `Args`.
- `model_analysis.get_dataset_dataloader/get_dataloader` per dataset specializzati.
- `engine_for_pretraining.train_one_epoch/test` per loop di training e valutazione auto-supervisionata.

## I/O e formati
- Input: percorsi dati definiti in `args.data_path/test_path` (formati specifici non determinati dal codice analizzato); patch size dedotta dal modello.
- Output: `log.txt` (JSON lines), file TensorBoard in `log_dir`, checkpoint salvati tramite `utils.save_model` nella cartella `output_dir`.
- Richiede GPU e backend NCCL; presume visibilità condivisa dei dataset tra rank.

## Punti di estensione/assunzioni
- Scheduler step-level richiede `num_training_steps_per_epoch > 0`; dataset troppo piccoli rispetto al batch generano divisione per zero.
- Il modello deve esporre attributo `encoder.patch_embed.patch_size`; altri modelli necessitano adattatore.
- Per abilitare run single-GPU occorre impostare correttamente i flag distribuiti (`get_resources`).
- È necessario aggiungere un parser CLI o passare esplicitamente `terminal_args` prima di usare lo script stand-alone.

## Copertura
- ✓ `launch_specialization_training`
- ✓ `__main__`
