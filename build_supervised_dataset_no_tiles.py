import os
from pathlib import Path
import shutil
import csv
import re
from collections import defaultdict
from datetime import datetime
import pandas as pd

from build_dataset import extract_dates_pattern_airmass_rgb_20200101_0000

    
def split_into_subfolders_and_track_dates(images, output_dir, num_frames=16):    
    subfolder_info = []
    num_total_files = len(images)
    # quanti blocchi di 16 consecutivi
    num_subfolders = num_total_files // num_frames

    for i in range(num_subfolders):
        subfolder_name = f"part{i+1}"
        subfolder_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        start_idx = i * num_frames
        end_idx = start_idx + num_frames

        # Copia effettiva dei 16 frame nella sottocartella
        for idx, file in enumerate(images[start_idx:end_idx]):
            new_name = os.path.join(subfolder_path, f"img_{idx+1:05d}.png")
            shutil.copy(file, new_name)

        # Esempio: estrai la data dal frame centrale
        mid_idx = start_idx + num_frames // 2
        dt = extract_dates_pattern_airmass_rgb_20200101_0000(images[mid_idx].name)
        
        # Salviamo in una lista: cartella e data
        subfolder_info.append({"folder": subfolder_path, "date": dt})

    return subfolder_info


def create_supervised_csv_from_info(subfolder_info, medicane_csv, out_csv):
    import csv
    from datetime import datetime
    import pandas as pd

    # Carica gli intervalli di medicane
    intervals = load_medicane_intervals(medicane_csv)


    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path","start","end","label"])

        for item in subfolder_info:
            folder_path = item["folder"]
            dt = item["date"]
            if dt is None:
                # Se non riesci a parsare la data, skip o label = -1
                continue
            label = 1 if is_in_medicane(dt, intervals) else 0
            writer.writerow([folder_path, 1, 16, label])

    print(f"Creato CSV supervisionato in: {out_csv}")



def load_medicane_intervals(medicane_csv):
    """
    Legge un file CSV con le date di inizio/fine dei Medicane.
    Esempio: col start_date, end_date in formato 'YYYY-MM-DD HH:MM'
    """
    
    intervals = []
    df = pd.read_csv(medicane_csv)
    for _, row in df.iterrows():
        start_dt = datetime.strptime(row['Start_Date'], "%Y-%m-%d")
        end_dt   = datetime.strptime(row['End_Date'],   "%Y-%m-%d")
        intervals.append((start_dt, end_dt))
    return intervals

def is_in_medicane(date_to_check, intervals):
    return any(start <= date_to_check <= end for (start, end) in intervals)




if __name__ == '__main__':
    
    input_dir = "./from_gcloud"  # Cambia questo percorso
    output_dir = "./airmassRGB"  # Percorso per salvare i CSV
    os.makedirs(output_dir, exist_ok=True)
    filenames = get_files_from_folder(folder=input_dir, extension="png")

    file_metadata = []
    for fname in filenames:
        start_dt = extract_dates_pattern_airmass_rgb_20200101_0000(fname.name)
        file_metadata.append((fname, start_dt))

    sorted_files = sorted(file_metadata, key=lambda x: x[1])  # Ordina per start_dt
    #random_fnames =  [item[0] for item in file_metadata]
    sorted_filenames = [item[0] for item in sorted_files]

    subfolder_info = split_into_subfolders_and_track_dates(sorted_filenames, output_dir)
    
    folder_root = output_dir      # cartella con subfolder part1, part2 ...
    medicane_csv = "./medicane_validi.csv"    # le date dei medicane
    out_csv = "./dataset.csv"
    create_supervised_csv_from_info(subfolder_info, medicane_csv, out_csv)
    
    