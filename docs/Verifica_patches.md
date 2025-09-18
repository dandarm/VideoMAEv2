# Verifica_patches.ipynb

## Overview
Notebook dedicato all’analisi delle ricostruzioni del modello VideoMAE specializzato (pretraining) e alla verifica delle patch utilizzate. Include tre blocchi principali: analisi dei log di training, generazione di GIF da sequenze ricostruite e ispezione delle maschere di patching e della dimensione del decoder.

## Analisi dei log
- Lettura del file `log_train1000video.txt`, estrazione di epoch, train loss e test loss, unione in un DataFrame e applicazione di forward-fill per colmare eventuali gap.
- Plot delle curve di perdita (train/test) per monitorare l’andamento dell’allenamento.

## Generazione di gif
- Caricamento di immagini da una cartella (`sequenced_imgs/`) e salvataggio di un’animazione GIF tramite `PIL.Image` per controllare visivamente la sequenza.

## Inferenza con il modello MAE
- Configurazione degli argomenti (`prepare_args`) per la modalità test con dataset non supervisionato (`test_dataset_2802.csv`).
- Creazione del `DataManager` specializzato (unsupervised) che restituisce triple (`images`, `bool_masked_pos`, `decode_masked_pos`).
- Esecuzione del modello VideoMAE specializzato (`specialized_model`) in `torch.no_grad` e raccolta di output e immagini originali.
- Visualizzazione di singoli frame (`visualize_frame`) e salvataggio di GIF dell’input e della ricostruzione (`save_animation_gif`).

## Studio delle maschere e delle dimensioni
- Valutazione di dimensioni e numero di patch (`bool_masked_pos.sum()`, `decode_masked_pos.sum()`) per verificare il rapporto tra patch codificate e decodificate.
- Calcolo di `patch_dim`, conteggio delle patch per lato e verifica della corrispondenza con la dimensione dell’output del decoder.

## Ricostruzione manuale
- Definizione della funzione `reconstruct_mae_batch`, che esegue denormalizzazione, patchify, applicazione del modello e ricostruzione delle patch, restituendo frame ricostruiti e immagini originali ristrette.
- Iterazione sul dataloader per salvare GIF della ricostruzione e dell’input, utile per confronti visivi.

## Output
- Grafici di perdita, GIF (`reconstruction.gif`, `my_original.gif`) e statistiche sulle maschere.

## Note
- Il notebook presuppone che il modello specializzato (`specialized_model`) sia già caricato nel contesto e che `prepare_args` sia disponibile.
- Le funzioni di ricostruzione possono essere riutilizzate per validare altri checkpoint MAE cambiando il path in `args.init_ckpt`.
