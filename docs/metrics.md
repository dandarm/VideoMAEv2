# Performance Metrics for Rare Event Detection

In binary classification problems, especially in meteorology and environmental sciences, the evaluation of performance cannot rely solely on **accuracy**, which may be misleading in the presence of highly imbalanced datasets. Instead, skill scores specifically designed for rare-event detection are preferable.  

Below, we introduce the main verification metrics used for this task, derived from the confusion matrix.

---

## Confusion Matrix

We consider the standard 2×2 contingency table, shown with both **machine learning notation** and the **meteorological convention**:

<img src="confusion_matrix_square.png" alt="Project Icon" width="300" />


Where:  
- **H (Hits)** = correctly predicted events  
- **M (Misses)** = events not predicted (false negatives)  
- **F (False Alarms)** = predicted events that did not occur (false positives)  
- **C (Correct Negatives)** = correctly predicted non-events  

---

## Probability of Detection (POD) / Recall

The **POD** (also called **Recall** or **Sensitivity**) quantifies the fraction of observed positives that were correctly predicted. It is particularly important when missing positive events (false negatives) is unacceptable.

$\text{POD} = \frac{TP}{TP + FN} = \frac{H}{H + M}$

- **Range**: \([0,1]\)  
- **Best value**: 1 (perfect detection)

---

## False Alarm Ratio (FAR)

The **FAR** measures the proportion of predicted positives that were in fact false alarms. 

Among all predicted events, it is the fraction that did not actually occur.  


$\text{FAR} = \frac{FP}{TP + FP} = \frac{F}{H + F}$


- **Range**: \([0,1]\)  
- **Best value**: 0 (no false alarms)  
---

## Critical Success Index (CSI) / Threat Score

The **CSI** (also known as the **Threat Score**) evaluates the accuracy of event prediction while ignoring true negatives. Particularly suited for rare-event problems where TN (or C) dominates the confusion matrix. 

It is the fraction of correctly detected events out of all events that were either forecast or observed.  

$\text{CSI} = \frac{TP}{TP + FP + FN} = \frac{H}{H + F + M}$


- **Range**: \([0,1]\)  
- **Best value**: 1  

---

## Heidke Skill Score (HSS)

The **HSS** provides a skill-adjusted measure of accuracy, comparing the model against random chance. It accounts for both correct detections and correct rejections.

Measures the accuracy of predictions relative to random chance. 

$
\text{HSS} = \frac{2 (TP \cdot TN - FN \cdot FP)}{(TP + FN)(FN + TN) + (TP + FP)(FP + TN)}
= \frac{2 (H \cdot C - M \cdot F)}{(H+M)(M+C) + (H+F)(F+C)}$

- **Range**: \($[- \infty, 1]$\)  
- **Best value**: 1 (perfect forecast)   

---

## Balanced Accuracy (BA)

The **Balanced Accuracy** is the average of the hit rate on the positive class (**Recall/POD**) and the hit rate on the negative class (**Specificity**).

Ensures equal weight to both classes, useful under class imbalance.  

$
\text{BA} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)
= \frac{1}{2}\left(\frac{H}{H + M} + \frac{C}{C + F}\right)$

- **Range**: \([0,1]\)  
- **Best value**: 1 

---

## Summary

- **POD (Recall)**: Prioritizes not missing positive events (minimizing misses \(M\)).  
- **FAR**: Quantifies false alarms among predicted positives.  
- **CSI**: Balances hits, misses, and false alarms, robust for rare-event detection.  
- **HSS**: Provides a skill score relative to random chance, including both positives and negatives.  
- **Balanced Accuracy**: Symmetrically weights performance on positive and negative classes, mitigating class imbalance effects.  

These metrics together provide a comprehensive evaluation framework for binary classification under strong class imbalance, with particular emphasis on rare-event detection such as in meteorological applications.
