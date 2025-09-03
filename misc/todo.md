# Todo List per il progetto VideoMAEv2 su dataset satellitari

## Dataset

- Aumentare le classi per le tile vicine spazialmente (limitrofe), tile nuvolose e clear sky

- Tiling dinamico che dipenda dalla posizione del ciclone, con la classe aggiuntiva (questo risolverebbe anche il problema dello spostamento temporale del centro del ciclone, per cui una tile passa da 1 a 0 o viceversa con minimi cambiamenti visivi dovuti al cambiamento istantaneo del centro da una tile a quella vicina)

- Dataset di test di un anno intero

- Crossfold validation 

## Analisi e Statistiche

- Guardare i valori logits prima della softmax e plottare i risultati corrispondenti a tutto il dataset per scoprire eventuali distribuzioni. Ricondurre tutti i valori ai casi di \
  - clear sky
  - tile medicane
  - tile limitrofe
  - cloudy tiles lontane

    e vedere se rimangono valori distribuiti diversamente da questi 4 casi.
  - Eventulamente analizzare i valori logits per le video tiles cloudy in funzione del cloud index
- Interpretare i falsi positivi in base alla distanza dal vero positivo -> se sono tile limitrofe non è così grave: escluderli o contarli a parte da quelli lontani o comunque vicini ma senza nuvole.



- Documentazione\
  completare il readme.md e agent.md

