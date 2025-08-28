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



## **Metriche**

Introdurre le metriche POD, FAR, CSI, HSS, e se nessuna corrisponde all'F1Score, anche quest'ultimo.\
Usare le seguenti definizioni, raccordarle a quelle usate in sklearn e già calcolate: False positives, false negatives, True positives, True negatives



- false alarm ratio (FAR).\
  A verification measure of categorical forecast performance equal to the number of false alarms divided by the total number of event forecasts. With respect to the 2x2 verification problem example outlined in the definition of contingency table, FAR= (B)/(A+B).

- probability of detection (POD).

  A verification measure of categorical forecast performance equal to the total number of correct event forecasts (hits) divided by the total number of events observed. Simply stated, it is the percent of events that are forecast. With respect to the 2x2 verification problem example outlined in the definition of contingency table, POD= (A)/(A+C).

- critical success index (CSI).

  Also called the threat score (TS), is a verification measure of categorical forecast performance equal to the total number of correct event forecasts (hits) divided by the total number of storm forecasts plus the number of misses (hits + false alarms + misses). The CSI is not affected by the number of non-event forecasts that verify (correct rejections). However, the CSI is a biased score that is dependent upon the frequency of the event. For an unbiased version of the CSI, see the Gilbert skill score (GS). With respect to the 2x2 verification problem example outlined in the definition of contingency table, CSI= (A)/(A+B+C).

- Heidke skill score (HSS).

  A skill corrected verification measure of categorical forecast performance similar to the success ratio (SR) but which takes into account the number of correct random forecasts (chance hits + chance correct rejections). The HSS is equal to the total number of correct forecasts minus the correct random forecasts (hits + correct rejections - correct random forecasts) divided by the total number of forecasts minus the correct forecasts due to chance (hits + false alarms + misses + correct rejections - correct random forecasts). With respect to the 2x2 verification problem example outlined in the definition of contingency table, HSS= (A+D-E)/(A+B+C+D-E), where E= correct random forecasts. This skill score falls within a (-1, +1) range. No incorrect forecasts give a score of +1, no correct forecasts give a score of -1, and either no events forecast or no events observed give a score of 0. For rare event forecasts, the HSS in the limiting case approaches 2A/(2A+B+C), a simple function of the Critical Success Index (CSI).

- 2X2 Contingency Table definition:

                                                               Event Observed\
  \-----------------------------------------------------------------

                                                                Yes              No

                                     |

                                     |    Yes                 A                 B

  Event Forecast            |

                                     |

                                     |     No                 C                 D

                                     |





- Documentazione\
  completare il readme.md e agent.md

