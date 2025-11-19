# Conversione coordinate pixel ↔ chilometri

Questo documento raccoglie tutti i punti del codice che gestiscono il passaggio fra coordinate in pixel, coordinate geografiche (lat/lon) e distanze in chilometri. È pensato come guida operativa: ogni sezione indica il file e le righe da consultare per replicare la procedura.

## Riferimenti chiave
- `medicane_utils/geo_const.py:4-77` – definisce il dominio geografico, crea l'oggetto Basemap e fornisce le utility per ottenere la griglia lat/lon per ogni pixel tramite `get_lon_lat_grid_2_pixel` e per trovare il pixel più vicino (`trova_indici_vicini`).
- `dataset/build_dataset.py:117-194` – contiene le funzioni `compute_pixel_scale`, `coord2px`, `get_cyclone_center_pixel` e `inside_tile` usate per convertire lat/lon (o spostamenti espressi in metri/km) in coordinate pixel rispetto alle immagini 1290×420.
- `engine_for_tracking.py:23-90` – implementa il percorso inverso: `_pixels_to_latlon` associa a ogni coppia (x, y) globale la lat/lon corrispondente, mentre `_haversine_km` e `batch_geo_distance_km` trasformano differenze in pixel in distanze geodetiche in chilometri.
- `inference_tracking.py:37-114` – esempio concreto di conversione per singolo campione (`_compute_sample_record`), utile come blueprint per calcolare errori sia in pixel sia in chilometri.

## 1. Preparare la mappa lat/lon dei pixel

1. `medicane_utils/geo_const.py:9-45` crea un oggetto Basemap geostazionario centrato sul Mediterraneo e usa `makegrid(image_w=1290, image_h=420)` per generare `lon_grid` e `lat_grid`.
2. `get_lon_lat_grid_2_pixel` restituisce queste matrici (più le coordinate metriche X/Y) e viene richiamata sia lato dataset sia lato tracking per avere una lookup table consistente.
3. Se si lavora con coordinate note (lat/lon) e serve trovare il pixel più vicino, `trova_indici_vicini` (`medicane_utils/geo_const.py:49-77`) calcola la distanza quadratica su tutta la griglia e restituisce gli indici (colonna, riga) del pixel target.

## 2. Da chilometri/lat-lon a pixel

Quando si parte da coordinate fisiche (chilometri o lat/lon) e si vuole sapere a che pixel corrispondono:

1. `compute_pixel_scale` (`dataset/build_dataset.py:117-134`) proietta i corner dell'immagine nel sistema metrico di Basemap. Dividendo i pixel totali per l'estensione metrica, la funzione fornisce `px_scale_x` e `px_scale_y`, ossia quanti pixel corrispondono a un metro sui due assi. Moltiplica questi valori per 1000 per passare da km a pixel.
2. `coord2px` (`dataset/build_dataset.py:136-148`) usa gli scale factor per convertire un punto espresso in lat/lon:  
   - converte la coppia in coordinate metriche (`default_basem_obj(lon, lat)`),  
   - rimuove l'offset di origine (`Xmin`, `Ymin`),  
   - scala in pixel tramite `px_per_m_x`/`px_per_m_y`,  
   - ribalta l'asse Y (l'immagine è indicizzata dall'alto verso il basso).
3. Le pipeline moderne usano direttamente `get_cyclone_center_pixel` (`dataset/build_dataset.py:155-162`), che richiama `get_lon_lat_grid_2_pixel` una sola volta all'avvio e poi trova gli indici del pixel più vicino con `trova_indici_vicini`, evitando di ricalcolare le proiezioni.
4. `inside_tile` (`dataset/build_dataset.py:164-185`) mostra come questi pixel vengano confrontati con i bounding-box delle tile 224×224: è l'esempio più semplice di “km → pixel → logica applicativa”.

## 3. Da pixel a lat/lon

Il tracking lavora su coordinate pixel relative alla tile. Per riportarle nel sistema geografico:

1. `engine_for_tracking._get_lon_lat_grid` (`engine_for_tracking.py:23-27`) richiama `get_lon_lat_grid_2_pixel` e memorizza la griglia (cache LRU per non ricomputare).
2. `_pixels_to_latlon` (`engine_for_tracking.py:29-36`) arrotonda e clippa le coordinate pixel globali, inverte l'asse Y (`row_idx = IMAGE_HEIGHT - 1 - y_idx`) e indicizza `lat_grid` / `lon_grid` per ottenere le coordinate geografiche corrispondenti.
3. `_parse_tile_offsets` (`engine_for_tracking.py:51-56`) legge gli offset (x, y) dal nome della cartella/tile. Le coordinate predette da rete e ground-truth sono relative alla tile, quindi `batch_geo_distance_km` somma questi offset per ottenere posizioni globali prima di passare `_pixels_to_latlon`.

## 4. Da pixel a chilometri

Una volta ottenute lat/lon per predizione e target:

1. `_haversine_km` (`engine_for_tracking.py:39-48`) applica la formula dell’haversine con `EARTH_RADIUS_KM = 6371.0088` per calcolare la distanza geodetica.
2. `batch_geo_distance_km` (`engine_for_tracking.py:59-90`) mette tutto insieme:
   - converte tensori PyTorch in NumPy (`pred_np`, `target_np`);
   - ricava gli offset con `_parse_tile_offsets` e calcola le coordinate globali (`global_pred`, `global_target`);
   - richiama `_pixels_to_latlon` per ottenere lat/lon assoluti;
   - calcola la distanza media in chilometri con `_haversine_km`, scartando eventuali valori non finiti.
3. Durante l’inferenza, `_compute_sample_record` in `inference_tracking.py:37-114` replica gli stessi passi per ogni sample e salva sia l’errore in pixel (`err_px`) sia quello in chilometri (`err_km`), utile come riferimento pratico.

## 5. Procedure operative

### 5.1 Calcolare quanti pixel corrispondono a un offset in chilometri
1. Ottieni `Xmin, Ymin, px_scale_x, px_scale_y = compute_pixel_scale()` (`dataset/build_dataset.py:117-134`).
2. Moltiplica il delta in km per `1000 * px_scale_x` (asse X) o `1000 * px_scale_y` (asse Y) per avere l’offset in pixel. Il ribaltamento dell’asse Y va gestito come in `coord2px` prima di confrontare le coordinate.

### 5.2 Convertire una coppia pixel globali in lat/lon e distanza
1. Somma gli offset della tile alle coordinate relative (vedi `_parse_tile_offsets` e `global_pred = pred_np + offsets` in `engine_for_tracking.py:74-83`).
2. Passa la coppia risultante a `_pixels_to_latlon` (`engine_for_tracking.py:29-36`) per ottenere latitudine e longitudine.
3. Se devi confrontare due punti, invia entrambe le coppie lat/lon a `_haversine_km` e, se necessario, riusa `batch_geo_distance_km` per gestire vettori interi.

### 5.3 Pipeline end-to-end (pixel ↔ km)
Per misurare l’errore del tracker in chilometri:
1. Carica le predizioni e i target (in pixel relativi alla tile).
2. Ricava gli offset dai nomi dei video/tile (`engine_for_tracking.py:51-56`) e ottieni le coordinate globali.
3. Mappa i pixel globali in lat/lon con `_pixels_to_latlon`.
4. Usa `_haversine_km` per ottenere la distanza in km. Se vuoi l’errore medio del batch, affidati direttamente a `batch_geo_distance_km`.

## 6. Note e best practice
- La griglia lat/lon è valida solo per immagini 1290×420; se cambiano dimensioni occorre rigenerarla chiamando `get_lon_lat_grid_2_pixel` con i nuovi parametri sia lato dataset che lato tracking.
- Quando lavori con offset manuali o affini, assicurati di invertire l’asse Y nello stesso modo in cui fanno `coord2px` e `_pixels_to_latlon`, altrimenti il punto risulterà ribaltato verticalmente.
- Per debug veloci è spesso sufficiente usare `_compute_sample_record` (`inference_tracking.py:37-114`), che incapsula tutti i passaggi e restituisce sia coordinate in pixel che lat/lon, oltre a `err_km`.
- Se devi processare molte coordinate lat/lon simultaneamente (es. durante il preprocessing dei CSV Manos), puoi vettorizzare `get_cyclone_center_pixel` oppure riutilizzare il blocco di codice mostrato nel notebook `Analyze_Manos_tracks.ipynb` (funzione `get_cyclone_center_pixel_vector` citata in `docs/Analyze_Manos_tracks.md`), mantenendo però le stesse costanti e ribaltamenti descritti qui.

Seguendo i riferimenti indicati sopra è possibile implementare o verificare qualsiasi conversione pixel ↔ km mantenendo allineata la pipeline con il codice di riferimento del repository.
