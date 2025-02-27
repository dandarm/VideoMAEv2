#!/bin/bash

# Attiva l'ambiente conda
source /home/daniele/anaconda2024/bin/activate /home/daniele/anaconda2024/envs/videomae

# Esegui lo script di classificazione
python classification.py

# Disattiva l'ambiente conda
conda deactivate
