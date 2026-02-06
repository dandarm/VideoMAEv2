# Tracking From Folder

Questo documento descrive lo script `track_from_folder.py`, che esegue l’inferenza di **tracking** a partire da una cartella di *video tile* (una subfolder per tile, con ~16 frame), salva un CSV di predizioni e opzionalmente genera i video tile annotati.

## Cosa fa lo script

1. **Legge i video tile**:
   - `--input_dir` deve contenere **solo cartelle**, una per tile.
   - ogni cartella contiene i frame `img_00001.png`, `img_00002.png`, ...
   - il **nome della cartella** deve avere formato: `DD-MM-YYYY_HHMM_offsetX_offsetY`
     - esempi: `07-11-2011_1900_426_0`, `30-11-2020_0700_1065_0`
2. **Esegue l’inferenza tracking**:
   - usa il modello indicato da `--model_path`.
   - salva le predizioni in `tracking_inference_predictions.csv` (o nome custom).
3. **(Opzionale) Genera i video tile annotati**:
   - attiva con `--make_video`.
   - per ogni tile crea una cartella con i frame annotati e un MP4.
   - predizione in **rosso**, GT in **verde** (se disponibile).

## Argomenti principali

- `--input_dir`: cartella che contiene le subfolder dei video tile.
- `--model_path`: checkpoint del modello di tracking.
- `--output_dir`: cartella output per CSV e metriche (default: `./output`).
- `--manos_file`: CSV Manos con GT (default: `medicane_data_input/medicanes_new_windows.csv`).
  - Se il file **non esiste**, l’output è solo con la traccia predetta (nessun GT).
- `--make_video`: genera i video tile annotati.
- `--ffmpeg_path`: (opzionale) path da aggiungere a `PATH` per trovare `ffmpeg`.
- `--on`: preset macchina per `arguments.py` (es. `leonardo`).

## Output

- CSV: `<output_dir>/tracking_inference_predictions.csv`
- Metriche: `<output_dir>/inference_tracking_metrics.txt`
- Video annotati (se `--make_video`):
  - frame: `<output_dir>/<tile_folder_name>/frame_00000.png`, ...
  - mp4: `<output_dir>/<tile_folder_name>.mp4`

## Esempio

```bash
python track_from_folder.py \
  --input_dir ../airmassRGB/supervised_tiles2track \
  --output_dir ../airmassRGB/supervised_tiles_tracked \
  --model_path ./output/checkpoint-tracking-best.pth \
  --manos_file medicane_data_input/medicanes_new_windows.csv \
  --make_video \
  --ffmpeg_path /percorso/ffmpeg
```

## Note

- La **GT** viene associata così:
  - dal nome della cartella si estraggono **data/ora** e **offset**.
  - nel CSV Manos (`time`, `x_pix`, `y_pix`) si prende la riga con `time` arrotondata all’ora e con punto `(x_pix, y_pix)` **dentro la tile** definita da `offsetX/offsetY`.
  - se non c’è match, la riga resta senza GT → output solo predizioni.
- `err_km` richiede che i nomi delle tile includano gli offset (come nello standard). Se non sono parsabili, `err_km` resta vuoto.
