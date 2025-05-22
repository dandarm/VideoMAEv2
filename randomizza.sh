#!/bin/bash

# Controlla se è stato passato un file come argomento
if [ $# -ne 1 ]; then
    echo "Uso: $0 <input_file>"
    exit 1
fi

INPUT="$1"
N=70000  # <-- Qui puoi impostare il numero di righe da estrarre a caso

# Crea nomi per gli output
OUTPUT1="UNtrain.csv"
OUTPUT2="UNtest.csv"
TEMP="shuffled.txt"

# Controlla che il file esista
if [ ! -f "$INPUT" ]; then
    echo "File non trovato: $INPUT"
    exit 1
fi

# Procedura
tail -n +2 "$INPUT" | shuf > "$TEMP"

head -n 1 "$INPUT" > "$OUTPUT1"
head -n "$N" "$TEMP" >> "$OUTPUT1"

head -n 1 "$INPUT" > "$OUTPUT2"
tail -n +$((N+1)) "$TEMP" >> "$OUTPUT2"

rm "$TEMP"

echo "Creati: $OUTPUT1 e $OUTPUT2"
