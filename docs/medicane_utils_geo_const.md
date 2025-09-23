# medicane_utils/geo_const.ipynb

## Overview
Notebook di supporto per verificare le costanti geografiche e le conversioni lat/lon → pixel utilizzate nel progetto. Fornisce esempi di utilizzo di Basemap, calcola scale in chilometri, disegna le coste del Mediterraneo e mostra come individuare il pixel più vicino a coordinate geografiche date.

## Componenti principali
- Import di `Basemap`, `numpy`, `matplotlib` e `PIL` per la visualizzazione.
- Definizione dei limiti geografici (`latcorners`, `loncorners`) e creazione dell’oggetto `basemap_obj` con le proiezioni impostate nel modulo.
- Calcolo e stampa della bounding box in coordinate cartografiche (`Xmin`, `Xmax`, `Ymin`, `Ymax`) e delle scale pixel/km.
- Rendering di una figura con coste, paralleli e meridiani utilizzando Basemap e customizing dell’area di disegno.

## Conversioni coordinate → pixel
- Uso delle funzioni `get_lon_lat_grid_2_pixel` e `trova_indici_vicini` per ottenere griglie di lat/lon per ogni pixel di un’immagine (dimensioni 1290×420) e per trovare gli indici del pixel più vicino a una coppia (lon, lat).
- Correzione della coordinata y (invertendo l’asse verticale) per mappare correttamente le coordinate sul sistema di riferimento delle immagini.
- Disegno del punto trovato su un’immagine (`img_pil`) utilizzando `PIL.ImageDraw` per una verifica visiva.

TODO: in 'production' non viene usata `trova_indici_vicini`, ma la funzione ottimizzata `get_cyclone_center_pixel_vector` usata in Analyze_Manos_tracks.ipynb. Verificare quale è la migliore e tenere solo quella.

## Output
- Figure di debug con le coste e il punto selezionato.

## Note
- Il notebook serve come riferimento per chi implementa nuove funzioni di mapping o deve verificare che le conversioni lat/lon → pixel siano coerenti con il sistema adottato.
