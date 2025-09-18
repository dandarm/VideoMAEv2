# View_tracking_tiles.ipynb

## Overview
Notebook pensato per ispezionare visivamente il dataset di tracking. Prende un video generato dalla pipeline `BuildTrackingDataset`, denormalizza i frame, disegna il centro del ciclone riportato nel CSV di tracking e mostra la sequenza come clip animata. Serve come strumento di QA rapido per le coordinate di tracking.

## Flusso operativo
1. **Import e configurazione**: vengono caricati `prepare_finetuning_args`, `BuildTrackingDataset`, `DataManager`, `MedicanesTrackDataset` e utility per la visualizzazione (`display_video_clip`).
2. **Preparazione dei dati**:
   - Lettura delle tracce aggiornate da `more_medicanes_time_updated.csv`.
   - Creazione dell’istanza `BuildTrackingDataset` con type `supervised`, che costruisce la lista di video tile contenente sia i frame sia le coordinate (in pixel) del centro ciclone.
   - Costruzione del `DataManager` in modalità valutazione per ottenere il `DataLoader` di tracking.
3. **Utility per parsing dei nomi cartella**: la funzione `extract_offsets` recupera gli offset (x, y) direttamente dal nome della cartella del video.
4. **Visualizzazione**:
   - Recupero di un campione (`track_ds[1190]`), denormalizzazione dei tensori usando le statistiche ImageNet e conversione a NumPy con layout `[T, H, W, C]`.
   - Disegno di un cerchio rosso sulle coordinate del ciclone per ciascun frame utilizzando `PIL.ImageDraw`.
   - Invocazione di `display_video_clip` per mostrare la sequenza risultante.

## Output
Il notebook mostra direttamente in Jupyter la clip annotata. Non produce file ma consente di validare la corrispondenza tra coordinate registrate nel CSV e posizione reale del ciclone sulla tile.

## Note
- Gli argomenti (`args`) e i percorsi (`input_dir`, `output_dir`, `csv_out`) devono puntare a dati già generati dalla pipeline di tracking.
- Modificando l’indice del campione (`track_ds[index]`) si possono esaminare rapidamente altri video.
