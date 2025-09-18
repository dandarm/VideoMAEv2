# Plot_metrics.ipynb

## Overview
Notebook destinato all’analisi comparativa dei log di addestramento. Carica i file di log generati durante il training (unbalanced, balanced, weighted), estrae metriche di accuratezza e tassi di errore e produce grafici per confrontare False Positive Rate (FPR) e False Negative Rate (FNR) tra diversi esperimenti.

## Flusso
1. Import delle librerie (`matplotlib`, `numpy`) e delle funzioni `collect_data`, `plot_training_curves`, `plot_fprfnr` dal modulo `plot_results`.
2. Richiamo di `collect_data` su file di log diversi (`log_weighted.txt`, `log_train_unbalanced.txt`, `log_90_lrbest2.txt`) per ottenere tuple con informazioni di training e validation.
3. Costruzione di liste di metriche (`accs`, `fprs`, `fnrs`, ecc.) estraendo dai log le curve corrispondenti.
4. Plot di FPR e FNR per ciascun esperimento in due subplot adiacenti, con colori differenziati e versioni “balanced”/“unbalanced”.

## Output
- Grafico Matplotlib con il confronto dei tassi di errore nei vari setup di addestramento.

## Note
- I file di log devono esistere nella cartella `output/old_logs`.
- Il notebook può essere esteso aggiungendo nuovi esperimenti all’array `results` per includerli nei grafici.
