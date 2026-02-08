# Predict And Track From Folder

Questo documento descrive lo script `predict_and_track_from_folder.py`, che **unisce** la pipeline di classificazione e tracking partendo da una cartella di immagini Airmass RGB. Lo script genera le video tile, fa classificazione, esegue tracking sulle tile positive e produce **un CSV finale per timeframe** + il **video Mediterraneo con tracce** (stile notebook `View_MED_tracking_preds`).

## Cosa fa lo script

1. **Crea le video tile** da `--input_dir` dentro `--output_dir`.
2. **Classificazione** sulle tile generate.
3. **Tracking** solo sulle tile classificate positive.
4. **CSV finale per timeframe** con colonne `datetime, has_cyclone, pred_lat, pred_lon`.
5. **Video Mediterraneo** con tracce PRED (rosso) e GT (verde) quando disponibile.

## Argomenti principali

- `--input_dir`: cartella di immagini (PNG con date nei nomi file).
- `--output_dir`: cartella dove salvare le tile e i CSV.
- `--classification_model_path`: checkpoint del modello di classificazione.
- `--tracking_model_path`: checkpoint del modello di tracking.
- `--split_by_subfolder`: se presente, processa ogni subfolder come sequenza separata.
- `--manos_file`: CSV Manos (opzionale) per etichette/GT.
- `--make_video`: genera il video Mediterraneo.
- `--only_video`: genera solo il video da PNG esistenti (salta rendering frame).
- `--ffmpeg_path`: (opzionale) path da aggiungere a `PATH` per trovare `ffmpeg`.
- `--on`: preset macchina per `arguments.py` (es. `leonardo`).

## Output

- **Classificazione (temp)**: `<output_dir>/_tmp_inference_predictions.csv`
- **Tracking per-tile (temp)**: `<output_dir>/_tmp_tracking_inference_predictions_tiles.csv`
- **CSV finale per timeframe**: `<output_dir>/tracking_inference_predictions.csv`
  - colonne: `datetime`, `has_cyclone`, `pred_lat`, `pred_lon`
- **Video Mediterraneo** (se `--make_video`):
  - `nomefile=<video_name>.mp4` nella root di esecuzione
  - frame intermedi in `./anim_frames_<video_name>/`

## Esempio

```bash
python predict_and_track_from_folder.py \
  --input_dir ../fromgcloud/2023 \
  --output_dir ../airmassRGB/supervised \
  --classification_model_path ./output/checkpoint-best-classification.pth \
  --tracking_model_path ./output/checkpoint-tracking-best.pth \
  --manos_file medicane_data_input/medicanes_new_windows.csv \
  --make_video \
  --ffmpeg_path /percorso/ffmpeg
```

## Note

- Le tile generate hanno nomi del tipo `DD-MM-YYYY_HHMM_offsetX_offsetY`. Questo formato è necessario per associare correttamente il tracking e la GT.
- Il **CSV finale per timeframe** sceglie:
  - la tile con **GT presente**, se `--manos_file` esiste;
  - altrimenti la **prima tile positiva** del timeframe.
  - se non ci sono tile positive per un timeframe, `has_cyclone=0` e `pred_lat/pred_lon=NaN`.
- Il **video Mediterraneo** non usa le tile PNG su disco: ricompone il mosaico a partire dai frame originali e disegna le tracce “al volo”, come nel notebook `View_MED_tracking_preds`.
- Lo script richiede **CUDA** per inferenza di classificazione e tracking.
