# Cloud_index.ipynb

## Overview
Il notebook esplora la stima della copertura nuvolosa nelle immagini Airmass RGB e come integrare tale informazione nei dataset video. Contiene esperimenti di thresholding, visualizzazioni interattive, calcolo dell’indice di nuvolosità per medicanes specifici e generazione di dataset separati “cloudy” e “clear-sky”.

## Analisi di un singolo frame
- Caricamento di un frame (`img_00010.png`), conversione in scala di grigi e calcolo di soglie tramite Otsu o percentile.
- Visualizzazione dei canali e della maschera binaria corrispondente usando Matplotlib.
- Uso delle utility `threshold_image` e `cloud_idx` per computare la frazione di pixel nuvolosi; salvataggio della maschera come immagine.

## Ispezione di clip campione
- Creazione di un dataloader di validation con `build_dataset`.
- Estrazione di un batch e visualizzazione dei frame (denormalizzati) con `display_video_clip`.
- Applicazione di `threshold_image` ai frame per mostrare le maschere di nuvolosità nel tempo.

## Analisi su un medicane (Juliette)
- Caricamento delle finestre temporali da `medicanes_new_windows.csv` e selezione del periodo relativo a Juliette.
- Generazione del DataFrame delle tile senza etichette (`create_df_unlabeled_tiles_from_metadatafiles`) e calcolo del cloud index per ciascuna tile (`add_cloud_idx_to_master_df`).
- Visualizzazione delle tile con overlay del valore di copertura nuvolosa, sia in formati statici sia tramite widget interattivi (dropdown per timestamp, slider per indice tile).

## Integrazione nel master dataset
- Caricamento del master supervisionato e invocazione di `make_df_video` per costruire gli esempi video.
- Calcolo dell’indice medio di nuvolosità per ogni video (`calc_avg_cld_idx`) e aggiunta della colonna `avg_cloud_idx`.
- Segmentazione del dataset in subset “cloudy” (indice > 0.2) e “clear sky”, con split train/test e salvataggio dei CSV (`cloudy_train_853.csv`, `cloudy_test_351.csv`).

## Output
- Immagini e GIF illustrative della copertura nuvolosa.
- CSV con l’indice medio di nuvolosità per video e dataset dedicati alla condizione cloud.

## Note operative
- Il notebook dipende da OpenCV (`cv2`) per il caricamento e il thresholding; assicura che le immagini siano accessibili in `output_dir`.
- I widget interattivi richiedono un ambiente notebook con supporto `ipywidgets`.
- I valori soglia possono essere modificati facilmente cambiando il percentile o i parametri nelle funzioni helper.
