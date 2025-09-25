# Todo List per il progetto VideoMAEv2 su dataset satellitari

# PRESENTAZIONE
rifare la conf matrix con font più grandi , e RECALL -> POD   

fare video inference di Ianos

## Dataset
- scaricare tutto il 2023

- verificare se nelle finestre temporali dei medicane ci sono altri cicloni contemporanei, e considerarli correttamente

- Aumentare le classi per le tile vicine spazialmente (limitrofe), tile nuvolose e clear sky

- Tiling dinamico che dipenda dalla posizione del ciclone, con la classe aggiuntiva (questo risolverebbe anche il problema dello spostamento temporale del centro del ciclone, per cui una tile passa da 1 a 0 o viceversa con minimi cambiamenti visivi dovuti al cambiamento istantaneo del centro da una tile a quella vicina)

- Crossfold validation 

- training senza clear sky


## Tracking
  - plot della traccia predetta (punto rosso) e traccia ground truth (punto verde)
  - stima della metrica MSE in km
  - renbdere la loss meno stringente, o accettare un risultato più approssimativo


## View MEDI
CLOUDY: vedo che se la tile esclusa aveva una label positiva e quindi un riquadro verde, questo non viene più plottato, possiamo mantenere tutte le feature indipendentemente? ma allora non le devo togliere per poi riespanderle altrimenti ho proprio perso l'informazione sulla tile, anche quando conteneva un ciclone.



  completare il readme.md e agent.md

- Video inference di tutti i medicane case studies in validation



