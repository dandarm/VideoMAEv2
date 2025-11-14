# View_MED_tracking_preds.ipynb

## Obiettivo
- Il notebook genera un video del Mediterraneo che mostra ogni tile ground truth/predetta affiancata dalla traccia del centro ciclone prodotta dal modello di tracking, utilizzando le inferenze salvate in `output/tracking_inference_predictions.csv` (`View_MED_tracking_preds.ipynb:7`).

## Setup e dipendenze
- In apertura viene caricato `autoreload` e il PATH viene esteso con il binario locale `ffmpeg-7.0.2-amd64-static`, requisito fondamentale per arrivare all’MP4 finale (`View_MED_tracking_preds.ipynb:67` e `View_MED_tracking_preds.ipynb:71`).
- Le utility principali sono importate da `arguments`, `dataset.build_dataset`, `dataset.data_manager` e `view_test_tiles`, includendo palette, parser di liste (`safe_literal_eval`) e la funzione di rendering `make_animation_parallel_ffmpeg` (`View_MED_tracking_preds.ipynb:90`).
- Gli argomenti del fine-tuning sono caricati via `prepare_finetuning_args()` per poter risolvere path di input/output e individuare eventuali fallback dei CSV di inferenza (`View_MED_tracking_preds.ipynb:116`).
- Il CSV di tracking viene cercato in `output/tracking_inference_predictions.csv` con fallback su `args.output_dir`, e subito dopo viene forzato il colore rosso per la classe `PRED` all’interno di `PALETTE`, così da usare un canale consistente all’atto del disegno della traccia sul video (`View_MED_tracking_preds.ipynb:123` e `View_MED_tracking_preds.ipynb:137`).

## Costruzione del dataset di tracking
- `prepare_tracking_args()` inizializza l’insieme di iper-parametri per la pipeline di regression e alimenta `make_tracking_data_builder_from_csv`, che ricostruisce le sequenze video e i metadati Manos a partire dal CSV `medicane_data_input/medicanes_new_windows.csv`, dal sottoinsieme `test_tracking_selezionati.csv` e dalle cartelle `../fromgcloud` / `../airmassRGB/supervised/` (`View_MED_tracking_preds.ipynb:204`).
- L’oggetto `test_builder` fornisce sia `df_video` (informazioni aggregate per ciascun video tile) sia `master_df` (metadati per singolo frame) che saranno usati per motivare l’espansione ai frame (`View_MED_tracking_preds.ipynb:205`).

## Join con le predizioni del tracker
- Il CSV prodotto da `tracking.py` o da `inference_tracking` viene letto come `df_predictions`, normalizzando il campo `path` e aggiungendo `path_basename` per poterlo allineare ai nomi di frame presenti nel builder (`View_MED_tracking_preds.ipynb:231`).
- Il merge tra `df_video` e le predizioni avviene su `path` vs `path_basename`, mantenendo gli offset delle tile e gli errori pixel/km, rinominando le colonne per distinguere l’offset video, quello stimato dal modello e il percorso della predizione (`View_MED_tracking_preds.ipynb:237`).
- A valle del merge, `df_joined` contiene per ogni video clip: lista di frame originali (`orig_paths`), vicinato (`neighboring`), id ciclone e la coppia di coordinate predette/target in pixel, lat/lon (vedere la stampa delle colonne in `View_MED_tracking_preds.ipynb:255`).

## Espansione a livello frame/tile
- Il blocco `records = []` cicla ogni riga di `df_joined`, normalizzando `orig_paths` tramite `_to_list`/`safe_literal_eval`, scegliendo la label/prediction da propagare e salvando le coordinate globali previste e di ground truth per ogni frame (`View_MED_tracking_preds.ipynb:664`).
- `_parse_datetime_from_name` recupera timestamp sia dal frame che dal video di appartenenza; `_select_best_row` minimizza la differenza temporale per agganciare frame e clip quando esistono duplicati o finestre sovrapposte (`View_MED_tracking_preds.ipynb:682`).
- `expand_mediterranean_dataframe` effettua il join con `master_df`, rinominando automaticamente le colonne `path/tile_offset_x/tile_offset_y` se necessario, e usa `calc_tile_offsets(stride_x=213, stride_y=196)` per enumerare tutte le tile teoriche da riempire (`View_MED_tracking_preds.ipynb:780`).
- L’inner helper `expand_group` garantisce che ogni frame includa la colonna `filling_missing_tile` impostata a `True` quando una tile manca e deve essere colorata di grigio, requisito imposto anche da `make_animation_parallel_ffmpeg` (`View_MED_tracking_preds.ipynb:823`).

## Preparazione delle tracce
- Dopo l’espansione, il DataFrame viene ordinato per `datetime` così da rispettare la sequenza temporale del video (`View_MED_tracking_preds.ipynb:1146`).
- La funzione `ensure_list` porta `x_pix`, `y_pix` e `source` a liste vere anche quando provengono da stringhe serializzate o da valori singoli (`View_MED_tracking_preds.ipynb:1158`).
- `append_track_row` inserisce in testa alle liste i punti `GT` e `PRED` quando sono presenti `track_target_*` o `track_pred_*`, assicurando che la traccia del modello e quella di riferimento vengano disegnate in ordine coerente durante il rendering (`View_MED_tracking_preds.ipynb:1183`).
- Il DataFrame `tracked_frames` è una copia ordinata di `expanded_df` con `disable_tile_boxes=True` per concentrarsi sulle tracce anziché sui bounding box delle tile, e rappresenta la sorgente finale per la generazione dei frame (`View_MED_tracking_preds.ipynb:1314`).

## Rendering dei frame e creazione del video
- Il notebook invoca `make_animation_parallel_ffmpeg(tracked_frames, output_folder='./anim_frames_tracking', nomefile='test_tracking_predictions')`, generando prima i PNG e poi il MP4 finale (`View_MED_tracking_preds.ipynb:1436`).
- `make_animation_parallel_ffmpeg` (in `view_test_tiles.py`) verifica la presenza della colonna `filled_gray`, invoca `save_frames_parallel` per distribuire la generazione delle immagini su più CPU e crea un file `frames.txt` che elenca i frame con durata fissa (`view_test_tiles.py:1037` e `view_test_tiles.py:1050`).
- Il comando ffmpeg usato concatena i PNG in ordine, forza `-framerate 10`, codec `libx264`, `-crf 18` e `-pix_fmt yuv420p` così da ottenere un video riproducibile ovunque; la funzione ripete la chiamata se il file non esiste ancora (`view_test_tiles.py:1038`).
- Il risultato è salvato come `anim_frames_tracking_test_tracking_predictions/*.png` più `test_tracking_predictions.mp4` nella root del notebook, pronto per l’ispezione o per essere allegato a report ed esperimenti.

## Come replicare / adattare
1. Assicurarsi che `output/tracking_inference_predictions.csv` sia aggiornato (può provenire da `tracking.py` o `tracking_inference.py`).
2. Aggiornare eventualmente `manos_csv` o `selected_csv` per cambiare la finestra temporale o il subset di cicloni (`View_MED_tracking_preds.ipynb:205`).
3. Rieseguire le celle in ordine: setup, builder, merge, espansione, preparazione tracce, rendering.
4. Se si desidera cambiare colori o stili della traccia, modificare `PALETTE` prima di ricreare `tracked_frames` per propagare i cambiamenti nell’animazione (`View_MED_tracking_preds.ipynb:137`).

## Output attesi
- DataFrame intermedi (`df_mapping`, `expanded_df`, `tracked_frames`) per debug e per esportare snapshot dei metadati.
- Directory `anim_frames_tracking_test_tracking_predictions` con tutti i frame PNG resi.
- File finale `test_tracking_predictions.mp4` che contiene: mosaico di tile Mediterraneo, overlay delle tile mancanti in grigio, tracce GT (verde di default) e PRED (rosso forzato), più i timestamp e i testi disegnati da `view_test_tiles`.
