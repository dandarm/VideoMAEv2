# Todo List per il progetto VideoMAEv2 su dataset satellitari

## Dataset

- Aumentare le classi per le tile vicine spazialmente (limitrofe), tile nuvolose e clear sky

- Tiling dinamico che dipenda dalla posizione del ciclone, con la classe aggiuntiva (questo risolverebbe anche il problema dello spostamento temporale del centro del ciclone, per cui una tile passa da 1 a 0 o viceversa con minimi cambiamenti visivi dovuti al cambiamento istantaneo del centro da una tile a quella vicina)

- Dataset di test di un anno intero
    - Devo prendere un master_df caricato su tutte le immagini, non sulle righe di un manos track, perché altrimenti andrei a prendere solo gli intervalli relativi a cicloni,
      invece voglio tutto un anno, anche quando non ci sono cicloni

- Crossfold validation 


## View MEDI
vedo che se la tile esclusa aveva una label positiva e quindi un riquadro verde, questo non viene più plottato, possiamo mantenere tutte le feature indipendentemente? ma allora non le devo togliere per poi riespanderle altrimenti ho proprio perso l'informazione sulla tile, anche quando conteneva un ciclone.


- Documentazione\
  completare il readme.md e agent.md



# lost codex chats
$HOME/.codex/sessions/2025/09/09