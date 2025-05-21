#!/usr/bin/env python3
import os
import sys
from time import perf_counter

def main():
    start_time = perf_counter()
    directory = os.path.expandvars("$FAST/airmass/")  # Sostituisci
    input_file = "./all_data_unsup.csv"              # Sostituisci
    output_file = "./filtered_data.csv"          # Sostituisci

    # Fase 1: Indicizzazione file nella cartella (20k file)
    print("[1/3] Indicizzazione file...", flush=True)
    existing_files = set()
    for entry in os.scandir(directory):
        if entry.is_file():
            existing_files.add(entry.name.lower())  # Case-insensitive se necessario

    # Fase 2: Processamento file di input (1M righe)
    print(f"[2/3] Processamento {input_file}...", flush=True)
    processed = 0
    found = 0
    batch = []

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            processed += 1
            filename = line.split(',', 1)[0].strip()
            
            # Ottimizzazione: controllo case-insensitive
            if filename.lower() in existing_files:
                fout.write(line)
                found += 1

            # Progresso ogni 50k righe
            if processed % 50000 == 0:
                print(f"\rRighe processate: {processed} | Trovati: {found}", end='', flush=True)

    # Fase 3: Statistiche
    elapsed = perf_counter() - start_time
    print(f"\n[3/3] Completato in {elapsed:.2f} secondi")
    print(f"• File nella cartella: {len(existing_files)}")
    print(f"• Righe processate: {processed}")
    print(f"• Righe valide trovate: {found}")
    print(f"• Output salvato in: {output_file}")

if __name__ == "__main__":
    main()
