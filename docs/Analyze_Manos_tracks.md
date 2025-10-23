# Analyze_Manos_tracks.ipynb


## Overview
Questo notebook è dedicato all’analisi esplorativa delle tracce dei cicloni fornite da Manos (dataset `TRACKS_CL*.dat`). Vengono unificati i file per classi di ciclone (CL2–CL10), si effettua una verifica della consistenza e copertura temporale, sono calcolate le tracce con coordinate in pixel rispetto al riferimento dato dalle immagini Airmass RGB con un dato riferimento spaziale, e sono prodotti sottoinsiemi focalizzati sui cicloni maggiormente definiti, aventi durata maggiore e strutture più simili ai Medicane, e sui Medicanes stessi. Infine vengono generati file di tracce CSV aggiornati con nuove finestre temporali più aderenti alle fasi in cui il ciclone è chiaramente sviluppato e mostra una dinamica con una rotazione evidente.


## Caricamento dei file CL e verifiche preliminari
- Vengono raccolti i file `TRACKS_CL§.dat` sostituendo il simbolo § con gli indici da 2 a 10 e trasformandoli in DataFrame tramite `load_cyclones_track_noheader`.
- Viene contato il numero di cicloni unici (`id_cyc`) e si confronta la presenza di specifici ID tra classi differenti per capire se gli identificativi siano univoci o meno, e confronti di coordinate (lat, lon, time) tra file CL permettono di misurare eventuali differenze nelle tracce replicate tra i vari CL; vengono contate le righe sovrapposte per valutare la coerenza. -> Risulta che lo stesso ciclone può avere diversi id nei diversi files CL, e anche le tracce posso differire leggermente


## Assemblaggio del tracks dataframe
- Viene creato un DataFrame cumulativo che include tutte le righe dei diversi CL, annotando l’origine (`source`) e garantendo un indice unico.
- Vengono manipolati (e generati ex novo) gli identificatori numerici univoci (`id_cyc_unico`, `id_final`, `idorig`), utilizzando la funzione `decodifica_id_intero` per avere codifiche coerenti e distinguere i cicloni provenienti da ERA5.
- Dalla versione corrente la colonna di pressione (`pressure`) dei file `TRACKS_CL*.dat` è preservata lungo l’intera catena di DataFrame, così da poterla usare in analisi successive o durante la costruzione dei dataset.
- Si procede a un taglio spaziale sulle coordinate specificate in `(latcorners, loncorners)` con la funzione `select_MEDI_area`, e varie selezioni temporali a fini statistici
- Sono calcolate le statistiche di durata per ogni ciclone (con istogrammi). 
- Vengogo riconteggiati i cicloni unici per le varie selezioni effettuate


## Calcolo delle coordinate in pixel: trasformazione da lon,lat a x,y
- Si usa `get_lon_lat_grid_2_pixel` e `get_cyclone_center_pixel_vector` per la trasformazione
- Si aggiungono le colonne `x_pix` e `y_pix` al dataframe, e si salva in `all_manos_CL_pixel.csv`
- Si filtrano via quelli che durano meno di 3 giorni per fare statistiche di studio
- codice per filtrare via dai metadata_files tutti i file immagine che non rientrano negli intervalli temporali di copertura date dal df tracks di Manos (poi inglobato in una funzione dentro DataBuilder? TODO: verificare)


## Si salvano versioni normalizzate per classi specifiche 
- `manos_CL10_pixel.csv`, `manos_CL7_pixel.csv`


## Integrazione con ERA5 e selezione Medicanes
- Si carica `era5_medicanes.csv` e si uniscono le tracce ERA5 con quelle CL7, arricchendo il dataset con nomi noti dei medicanes, e si esportano diversi CSV: `manos_medicanes_only.csv`, `more_medicanes.csv`.
- Tramite liste di ID/nome predefinite si compongono subset di medicanes d’interesse, sia numerici (CL7) sia nominati (ERA5), assicurandosi che la colonna `id_final` sia valorizzata per tutti i casi.

## Aggiornamento delle finestre temporali
- Viene caricata `new_cyc_limits.csv`, contenente nuove indicazioni di start/end per alcuni cicloni, e si applica una procedura di merge basata su chiavi derivate (`id_key`).
- Il DataFrame aggiornato sostituisce `start_time` ed `end_time` con i valori nuovi quando presenti, mantenendo i dati originali per i restanti casi.
- Il risultato finale è salvato come `more_medicanes_time_updated.csv`, base per successive pipeline di costruzione dataset e visualizzazione.

## Output e note
- CSV esportati: 
    - `manos_CL10_pixel.csv`: tracce CL10 (pixel coordinates)
    - `manos_CL7_pixel.csv`:  tracce CL10 (pixel coordinates)
    - `manos_medicanes_only.csv`: contiene solo Medicanes  
    - `more_medicanes.csv`:  sono stati aggiunti ulteriori id al dataset
    - `more_medicanes_time_updated.csv`: sono state cambiate le finestre temporali, ridotte al periodo in cui una rotazione è chiaramente visibile
- Il notebook fornisce analisi diagnostic (conteggi, confronto di coordinate) utili per comprendere la qualità delle tracce originali e per documentare differenze fra classi.
- Le funzioni di supporto qui impiegate (per generare chiavi ibride, calcolare ID numerici, valutare durate) possono essere riaffiorate quando si costruiscono dataset supervisionati o si filtrano periodi specifici.
