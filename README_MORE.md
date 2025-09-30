# VideoMAEv2 — Guida Approfondita

Questa guida integra il `README.md` principale fornendo una mappa ragionata di tutte le procedure documentate nella cartella [`docs/`](docs). Riassume come preparare il dataset, addestrare e valutare i modelli VideoMAEv2 e come utilizzare i notebook di analisi e visualizzazione disponibili nel repository.

## 1. Ambiente ed esecuzione di base
- **Installazione** – le dipendenze Python e PyTorch consigliate (con varianti 1.10/2.0) sono elencate in [`docs/INSTALL.md`](docs/INSTALL.md). Il flusso tipico usa Conda per creare l’ambiente `videomae`, installa PyTorch compatibile con CUDA e poi esegue `pip install -r requirements.txt`.
- **Struttura del repository** – le call tree in [`docs/training_call_tree.md`](docs/training_call_tree.md) e negli SVG associati (ad es. `training_call_tree.svg`, `classification_call_tree.svg`) sintetizzano le dipendenze fra script CLI e motori di training/inferenza (da estendere e integrare in forma meno formale con il presente doc)

## 2. Preparazione dei dati
- **Analisi delle tracce di Manos** – [`docs/Analyze_Manos_tracks.md`](docs/Analyze_Manos_tracks.md) spiega come aggregare i file `TRACKS_CL*.dat`, introdurre ID coerenti, trasformare coordinate geografiche in pixel e aggiornare finestre temporali più adeguate per i medicanes.
- **Costruzione dei dataset video** – [`docs/Build_dataset_videoMAE.md`](docs/Build_dataset_videoMAE.md) documenta l’intero flusso (tile extraction, sequenze da 16 frame, bilanciamento, salvataggio di ciascun video in folder) sia in modalità supervisionata sia auto-supervisionata, con focus sulle utility di `dataset.build_dataset` e sulla classe `BuildDataset`.
- **Generatore CLI dei dataset** – lo script [`docs/make_dataset_from_rgb.md`](docs/make_dataset_from_rgb.md) funge da dispatcher per creare il master dataframe, salvare le video tiles, subset “cloudy”, dataset annuali o tracciati Manos, selezionando la pipeline desiderata via flag mutuamente esclusivi.
- **Studio dell’indice di nuvolosità** – il notebook [`docs/Cloud_index.md`](docs/Cloud_index.md) descrive come stimare e annotare la copertura nuvolosa per tile e video, producendo dataset separati (cloudy/clear-sky) utili per training o analisi.
- **Esperimenti di relabeling e split** – [`docs/Experiment_dataset.md`](docs/Experiment_dataset.md) mostra scenari per limitare temporalmente le etichette, creare split basati su classi CL10 o medicanes e costruire dataset annuali usando `BuildDataset`.
- **Aggiornamento manuale delle finestre temporali** – [`docs/Video_cyclones_cut.md`](docs/Video_cyclones_cut.md) fornisce un’interfaccia interattiva per definire nuovi start/end dei cicloni e salvare `new_cyc_limits.csv`.
- **Supporto alle conversioni geografiche** – [`docs/medicane_utils_geo_const.md`](docs/medicane_utils_geo_const.md) documenta le costanti cartografiche, la proiezione Basemap e le funzioni di mapping lat/lon → pixel.

## 3. Pipeline di training
- **Pre-training auto-supervisionato** – [`docs/PRETRAIN.md`](docs/PRETRAIN.md) illustra gli script Slurm/torch.distributed per addestrare VideoMAEv2 con masking su dataset ibridi, spiegando gli iperparametri principali (`mask_ratio`, `decoder_depth`, `batch_size`, ecc.).
- **Specialization (MAE fine-tuning auto-supervisionato)** – [`docs/specialization.md`](docs/specialization.md) dettaglia lo script `specialization.py`, che riprende checkpoint MAE pre-addestrati, costruisce dataset custom con patch coerenti e gestisce training/test.
- **Fine-tuning supervisionato** – [`docs/classification.md`](docs/classification.md) entra nel merito dello script `classification.py`, illustrandone pipeline CLI → DataLoader → training loop → checkpointing.
- **Tracking del centro ciclone** – [`docs/TRACKING.md`](docs/TRACKING.md) descrive dataset, modello e loss per la regressione delle coordinate, mentre [`docs/tracking.md`](docs/tracking.md) documenta lo script `tracking.py` che orchestrano DataLoader, DDP, MSE loss, salvataggio checkpoint.

## 4. Inferenza e post-processing
- **Script di inferenza** – [`docs/inference_classification.md`](docs/inference_classification.md) dettaglia `inference_classification.py`, capace di produrre CSV di predizioni, logits o embedding aggregati da run distribuite, con gestione dei file NPZ multi-rank.
- **Analisi delle predizioni su scala Mediterranea** – i notebook [`docs/Predict_general_data.md`](docs/Predict_general_data.md), [`docs/View_MED_val_preds.md`](docs/View_MED_val_preds.md) e [`docs/View_test_tiles.md`](docs/View_test_tiles.md) spiegano come costruire set di inferenza custom, fondere le predizioni con i master dataframe, proiettarle sui frame originali e generare GIF/MP4 dell’intero bacino con overlay delle label. – [`docs/View_tracking_tiles.md`](docs/View_tracking_tiles.md) mostra come visualizzare rapidamente le clip annotate con i centri ciclone per verificare la qualità del dataset di regressione.
- **Utility per la raccolta e verifica delle ricostruzioni MAE** – [`docs/Verifica_patches.md`](docs/Verifica_patches.md) include procedure per analizzare log di pretraining, creare GIF delle ricostruzioni, studiare le maschere di patching e validare la pipeline MAE specializzata.

## 5. Valutazione e metriche
- **Metriche meteo-specifiche** – [`docs/metrics.md`](docs/metrics.md) riassume gli indicatori usati (POD/Recall, FAR, CSI, HSS, Balanced Accuracy) e il relativo significato nel contesto di eventi rari.
- **Notebook di valutazione** – [`docs/Model_stats.md`](docs/Model_stats.md) guida alla generazione di dataset di validation specifici, al calcolo di accuracy/FPR/FNR/POD/FAR e all’analisi per label temporali; integra funzioni di `model_analysis` e `utils` per la pipeline completa.
- **Analisi temporale e confronto run** – [`docs/Plot_compare_metrics.md`](docs/Plot_compare_metrics.md) e [`docs/Plot_train_loss.md`](docs/Plot_train_loss.md) mostrano come importare log multipli, confrontare FPR/FNR, visualizzare loss/accuracy e studiare schedule di learning rate in diversi esperimenti.

## 6. Risorse aggiuntive
- **Notebook di supporto ai dataset** – [`docs/Predict_general_data.md`](docs/Predict_general_data.md) e [`docs/Experiment_dataset.md`](docs/Experiment_dataset.md) fungono anche da reference per trasformare predizioni video in mappe a frame e per generare split custom (es. anni completi o dataset bilanciati).
- **Statistiche e visual analytics** – l’insieme dei notebook `View_*` fornisce strumenti per QA visivo, mentre `Plot_*` e `Model_stats` coprono la parte di analytics numeriche.
- **Diagrammi** – gli SVG (`classification_call_tree.svg`, `tracking_call_tree.svg`, `specialization_call_tree.svg`) offrono uno sguardo grafico su import e dipendenze degli script principali (ancora in via di miglioramento).

## 7. Percorso consigliato
1. **Preparare l’ambiente** con le istruzioni di installazione.
2. **Costruire o aggiornare i dataset** partendo dalle tracce di Manos e, se necessario, stimando la nuvolosità o ridefinendo le finestre temporali.
3. **Addestrare i modelli** seguendo le pipeline: pretraining/specialization → fine-tuning → tracking.
4. **Eseguire inferenza e analisi** usando gli script CLI e i notebook di visualizzazione per validare le predizioni sul Mediterraneo.
5. **Valutare e confrontare le run** tramite i notebook di metriche e plotting per scegliere i checkpoint migliori.

Questa guida costituisce la porta d’accesso ai notebook e agli script di `docs/`, aiutando a orientarsi tra le diverse fasi di preparazione dati, addestramento, inferenza e analisi del progetto VideoMAEv2.
