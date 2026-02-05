# Inference From Folder

Questo documento descrive lo script `predict_from_folder.py`, che esegue l’inferenza di classificazione a partire da una cartella di immagini (con data nel nome file), genera il CSV per `VideoMAE` e opzionalmente crea il video del Mediterraneo.

## Cosa fa lo script

1. **Costruisce il dataset di inferenza**:
   - legge le immagini dalla cartella `--input_dir` (anche con subfolder)
   - deduce l’intervallo temporale dai nomi file
   - genera i video/tile e il CSV per l’inferenza (`--dataset_csv`)
2. **Esegue l’inferenza**:
   - usa il modello indicato da `--model_path`
   - esegue una validazione con `no_grad` (come in `classification.py`)
   - salva le predizioni in `--preds_csv`
3. **(Opzionale) Crea il video del Mediterraneo**:
   - abilita con `--make_video`
   - usa `ffmpeg` per comporre l’MP4 finale

## Argomenti principali

- `--input_dir`: cartella di immagini (PNG con date nei nomi file)
- `--output_dir`: cartella di output per le tile video
- `--model_path`: checkpoint del modello
- `--split_by_subfolder`: se presente, tratta ogni subfolder come sequenza separata
- `--manos_file`: CSV Manos per etichette (opzionale)
- `--make_video`: genera il video Mediterraneo
- `--ffmpeg_path`: (opzionale) path da aggiungere a `PATH` per trovare `ffmpeg`
- `--only_video`: crea solo il video da frame PNG già esistenti (salta il rendering)
- `--on`: preset macchina per `arguments.py` (es. `leonardo`)

## Esempio (ambiente non HPC)

```bash
python predict_from_folder.py \
  --input_dir ../fromgcloud/2023 \
  --output_dir ../airmassRGB/supervised \
  --model_path ./output/checkpoint-best-lr-again2.pth \
  --manos_file medicane_data_input/medicanes_new_windows.csv \
  --make_video \
  --ffmpeg_path /percorso/ffmpeg \
  --only_video
```

## Esempio (HPC con Slurm)

```bash
sbatch inference_from_folder.sh
```
