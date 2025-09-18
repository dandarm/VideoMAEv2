# dataframe_creation_test_old.ipynb

## Overview
Notebook storico che documenta i primi esperimenti di creazione del master dataframe supervisionato. Mostra come caricare le tracce dei cicloni, associare le immagini Airmass RGB, suddividere in tile, calcolare le coordinate del ciclone in pixel e preparare i dataset sia supervisionati sia non supervisionati. Include anche routine di visualizzazione (GIF) per diversi medicanes.

## Pipeline principale
1. **Caricamento tracce**: lettura di `TRACKS_CL7.dat`, filtro temporale e spaziale (lat/lon) per conservare soltanto i cicloni nel Mediterraneo; arricchimento con colonne calendarizzate (anno, mese, giorno, ora) e marcatura dei medicanes tramite `medicane_validi.csv`.
2. **Lettura immagini**: ottenimento di una lista ordinata delle immagini da `../fromgcloud` con `load_all_images`.
3. **Selezione medicane**: estrazione delle immagini corrispondenti a cicloni specifici (Ianos, Trixie, Rolf) per verificare la corrispondenza tra tracce e frame.
4. **Creazione tile**: calcolo degli offset (`calc_tile_offsets`), generazione delle tile da una singola immagine (`create_tiles`) e memorizzazione delle coordinate (top-left) per ogni patch.
5. **Mapping coordinate -> pixel**: dato un timestamp, filtro delle tracce corrispondenti e conversione lat/lon in pixel con `coord2px`/`compute_pixel_scale`, validando l’individuazione del centro rispetto alla tile.

## Dataset non supervisionato
- Costruzione di un DataFrame con tutte le tile e i path (`df_data_unsup`) e salvataggio in `all_data_unsup.csv`.
- Raggruppamento per immagine, disegno delle tile e del centro del ciclone per visualizzazione (`draw_tiles_and_center`).

## Etichettatura supervisionata
- Uso delle funzioni `labeled_tiles_from_metadatafiles` e `get_tile_labels` per associare a ogni tile lo stato di presenza del ciclone.
- Salvataggio del master `all_data_full_tiles.csv` con tipizzazione accurata di tutte le colonne.
- Creazione di subset per medicanes individuali (Rolf, Ilona, Qendresa, Trixie, Numa, Ianos) e filtri orari.
- Generazione di GIF per ogni evento tramite `create_labeled_images_with_tiles` e un basemap predefinito.

## Output
- CSV: `all_data_unsup.csv`, `all_data_full_tiles.csv` e vari file derivati specifici per medicanes.
- GIF illustrative per verificare le tile estratte e la posizione del ciclone.

## Note
- Questo notebook rappresenta la base da cui è evoluta la pipeline successiva (`Build_dataset_videoMAE`); molte funzioni sono ora incapsulate in moduli riutilizzabili.
- Le routine di visualizzazione qui presentate sono utili per debug visivo quando si lavora con nuovi dati.
