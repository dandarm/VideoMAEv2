# Plot_train_loss.ipynb

## Overview
Notebook che raccoglie numerosi esperimenti di plotting dei log di training. Permette di caricare file cronologici, visualizzare curve di loss/accuratezza, confrontare run differenti e valutare strategie di apprendimento (learning rate schedule, dataset bilanciati/sbilanciati, variazioni di seed).

## Flusso base
- Import di `collect_data`, `plot_training_curves` e `plot_fprfnr` dal modulo `plot_results`.
- Richiamo di `collect_data` su differenti file di log (`log.txt`, `log_lr_again.txt`, `log_weighted.txt`, `log_medicanes_600_lr*.txt`, ecc.) per estrarre sequenze di metriche.
- Uso ripetuto di `plot_training_curves` con/ senza scala logaritmica per visualizzare loss e accuracy su training/validation/test.
- Plot aggiuntivi come `plot_fprfnr` per analizzare FPR/FNR su dataset multipli.

## Sezioni tematiche
- **Run principali**: confronto del log base (`log.txt`) con varianti su learning rate e pesatura delle classi.
- **Studio del learning rate**: costruzione di array di LR e weight decay pianificati, conversione da step a epoche, plot dei valori medi per epoca e della curva step-by-step.
- **Studi storici**: grafici per run più vecchie (600/1000 medicanes) per vedere l’evoluzione delle performance negli esperimenti iniziali.

## Output
- Grafici Matplotlib (loss, accuracy, LR schedule) salvati automaticamente quando viene passato `plot_file_name`.

## Note operative
- Molte celle sono replicate per caricare log differenti; è possibile commentare quelle non necessarie per ridurre il tempo di esecuzione.
- I log devono essere presenti nella cartella `output` (e `output/old_logs`) con la struttura attesa da `collect_data`.
