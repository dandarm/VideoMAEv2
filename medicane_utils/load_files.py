import re
from datetime import datetime
from pathlib import Path
import pandas as pd

#### legge il file con le date dei medicane
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



########### legge i track di MANOS
def load_cyclones_track_noheader(path_tracks):
    """
    Carica righe del file TRACKS_CL7 (senza header). Ogni riga ha 8 campi:
      id, LON, LAT, year, month, day, hour, pressure
    Restituisce una lista di dict:
        { 'id_cyc': str, 'LON': float, 'LAT': float, 'time': datetime }
    """
    rows = []
    with open(path_tracks, 'r') as f:
        for line in f:
            # es. "00000001 -013.991 0032.717 1979 01 08 16 01012.31\n"
            parts = line.split()
            if len(parts) < 8:
                continue
            id_str = parts[0]
            lon_str = parts[1].replace(',', '.')
            lat_str = parts[2].replace(',', '.')
            year = int(parts[3])
            month= int(parts[4])
            day  = int(parts[5])
            hour = int(parts[6])
            # pressure = parts[7]  # ignorato

            lat = float(lat_str)
            lon = float(lon_str)

            time_stamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)

            rows.append({
                'id_cyc': id_str,
                'lat':    lat,
                'lon':    lon,
                'time':   time_stamp,
            })
    return pd.DataFrame(rows)




def get_files_from_folder(folder, extension):
    folder = Path(folder)
    files = list(folder.rglob(f"*.{extension}"))
    return files
    
def extract_dates_pattern_airmass_rgb_20200101_0000(filename):
    """
    Estrae le date di inizio e fine acquisizione dal nome del file.
    
    Esempio di nome file:
    airmass_rgb_20200101_0000.png
    """
    pattern = r"^airmass_rgb_(\d{8})_(\d{4})\.png$"
    match = re.match(pattern, filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMM
        datetime_str = f"{date_str}{time_str}"
        dt = datetime.strptime(datetime_str, '%Y%m%d%H%M')
        return dt
    else:
        return None