## Scopo del modulo
Utility CLI per generare varianti di dataset RGB a partire da sorgenti Medicanes. Funziona da dispatcher verso builder specializzati (`make_sup_dataset`, `make_relabeled_dataset`, ecc.) selezionati tramite flag, impostando automaticamente sorgenti e destinazioni in base alla macchina (`--on`). Permette di creare DataFrame di riferimento, versioni riequilibrate e dataset di tracking.

## Flusso ad alto livello
```text
1. Definire parser CLI con flag mutuamente esclusivi che descrivono l'operazione richiesta.
2. Risolvere input/output base a seconda dell'ambiente (`leonardo` o `ewc`).
3. Se `--master_df`: invocare `make_master_df` per generare un dataframe consolidato.
4. Altrimenti, se `--relabeled_df`: eseguire `make_relabeled_dataset` (eventuale opzione `--cloudy`).
5. Se `--cloudy`: ricreare dataset con filtraggio basato sull'indice di nuvolosità
6. Se `--manos_tracks`: costruire dataset supervisionato da tracce Manos (`make_dataset_from_manos_tracks`).
7. Se `--tracking_manos`: creare dataset di tracking con split standard (`make_tracking_dataset_from_manos_tracks`).
8. Se `--all_year`: generare dataset completo per l'anno indicato (`make_dataset_from_entire_year`).
9. In assenza di flag specifici: default su `make_sup_dataset`.
```

## API principali
- `__main__`: unico entry point; elabora argomenti CLI, calcola root di input/output e richiama il costruttore di dataset pertinente; nessun valore di ritorno.

## Dipendenze
- `dataset.build_dataset` per tutte le routine di costruzione/trasformazione dei dataset.
- `argparse` per la definizione dell'interfaccia a linea di comando.

## I/O e formati
- Input: directory di tiles video sotto `$FAST/Medicanes_Data/from_gcloud` (leonardo) o percorsi analoghi per ewc; formati precisi dei CSV richiesti dai builder non determinati dal codice analizzato.
- Output: directory target (es. `$FAST/airmass/`, `../airmassRGB/supervised/`) popolata dai file/CSV prodotti dalle funzioni invocate; la struttura specifica dipende dall'helper scelto.

## Punti di estensione/assunzioni
- Le flag sono implementate con `elif`, quindi solo la prima condizione vera viene eseguita; combinazioni multiple non sono supportate.
- Richiede che le variabili d'ambiente (`$FAST`) siano valorizzate nel contesto `leonardo`.
- Per nuovi flussi è sufficiente importare il builder e aggiungere un ramo ulteriore prima del default.
- Non gestisce errori o validazioni dei percorsi: eventuali mancanze generano eccezioni propagate dagli helper.

## Copertura
- ✓ `__main__`
