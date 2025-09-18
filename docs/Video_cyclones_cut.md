# Video_cyclones_cut.ipynb

## Overview
Notebook interattivo per tagliare e annotare videoclips di cicloni. Consente di selezionare gli istanti di inizio e fine direttamente da un player video o da una sequenza di frame generata precedentemente, e di salvare i nuovi limiti temporali in un CSV riutilizzabile (`new_cyc_limits.csv`).

## Componenti principali
- Import di librerie per la UI (`ipywidgets`), manipolazione file (`Path`, `re`, `json`), gestione immagini (`PIL.Image`) e animazione (`matplotlib.animation`).
- Raccolta dei frame da una cartella (`anim_frames_{id}`) e ordinamento numerico per ricostruire la sequenza originale.
- Creazione di un’animazione Matplotlib che carica ogni frame on-demand attraverso `FuncAnimation`, utile come preview rapida.
- Implementazione di una cache LRU (`load_jpeg`) per ridurre il costo di caricamento quando si lavora con molti frame.

## Interfaccia per la selezione delle date
- Widget HTML/JavaScript personalizzati: generazione di ID univoci per video e pulsanti (`Segna INIZIO`, `Segna FINE`), aggancio degli eventi al player video e sincronizzazione con widget Python `FloatText`.
- Verifica dell’esistenza dei file selezionati (frame di start/end) e segnalazione di errori se mancano.
- Conversione dei timestamp selezionati in `datetime` e appending in una lista `lista_nuove_date`.

## Salvataggio dei nuovi limiti
- Costruzione di `df_new_limits` a partire dai dizionari raccolti e scrittura su disco (`new_cyc_limits.csv`) per alimentare pipeline successive (ad esempio aggiornare `more_medicanes_time_updated.csv`).

## Output
- CSV `new_cyc_limits.csv` contenente coppie (nome ciclone, start, end) aggiornate manualmente.

## Note
- Il notebook include più varianti dell’interfaccia (alcune commentate) per provare approcci diversi; è sufficiente eseguire la versione completa.
- Per il player video è necessario che `src_url` punti a un file servito da Jupyter (`/files/...`).
