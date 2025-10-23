import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from medicane_utils.geo_const import latcorners, loncorners#, x_center, y_center, default_basem_obj

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
    Restituisce un DataFrame con colonne:
        id_cyc, lat, lon, time, pressure
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
            pressure_str = parts[7].replace(',', '.')

            lat = float(lat_str)
            lon = float(lon_str)
            try:
                pressure = float(pressure_str)
            except ValueError:
                pressure = float('nan')

            time_stamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)

            rows.append({
                'id_cyc': id_str,
                'lat':    lat,
                'lon':    lon,
                'time':   time_stamp,
                'pressure': pressure,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
    return df




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
    


def get_all_cyclones():
    tracks_file = "./TRACKS_CL7.dat"  
    df_tracks = load_cyclones_track_noheader(tracks_file)
    df_tracks['time'] = pd.to_datetime(df_tracks['time'])

    # limit to > 2011 cyclones
    df_tracks = df_tracks[df_tracks['time'] > datetime(2011, 1, 1)]

    # limit to Mediterranean cyclones
    tracks_df_coord = df_tracks[
        (df_tracks['lat'] >= latcorners[0]) & (df_tracks['lat'] <= latcorners[1]) &
        (df_tracks['lon'] >= loncorners[0]) & (df_tracks['lon'] <= loncorners[1])
    ]

    # load Medicanes
    df_med = pd.read_csv('medicane_validi.csv')
    df_med['Start_Date'] = pd.to_datetime(df_med['Start_Date'])
    df_med['End_Date'] = pd.to_datetime(df_med['End_Date'])

    # join cyclones and medicanes
    tracks_df_coord['Medicane'] = None

    # Per ogni intervallo nel df_piccolo, assegna il nome alle righe che rientrano
    for i, row in df_med.iterrows():
        start = row['Start_Date']
        end = row['End_Date'] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # include tutto il giorno finale
        nome = row['Medicane']
        #print(start, end, nome, end=' ')
        
        m_start = tracks_df_coord['time'] >= start
        m_end = tracks_df_coord['time'] <= end
        mask = m_start & m_end
        #print(m_start.sum(), m_end.sum(), mask.sum())
        #display(tracks_df_coord[mask])
        tracks_df_coord.loc[mask, 'Medicane'] = nome

    av_med = tracks_df_coord[~tracks_df_coord['Medicane'].isna()]['Medicane'].unique()
    print(f"Available Medicanes for training: {av_med}")

    return tracks_df_coord


def decodifica_id_intero(id_unico):
    """Ripristina l'id originale del file CLn di Manos"""
    origine = id_unico // 1_000_000
    id_cyc = id_unico % 1_000_000
    source = f"CL{origine}"
    return source, id_cyc



def load_all_images(input_dir):
    filenames = get_files_from_folder(folder=input_dir, extension="png")
    #print(f"Trovati {len(filenames)} files")

    file_metadata = []
    for fname in filenames:
        start_dt = extract_dates_pattern_airmass_rgb_20200101_0000(fname.name)
        file_metadata.append((fname, start_dt))

    sorted_metadata = sorted(file_metadata, key=lambda x: x[1])  # Ordina per start_dt
    #random_fnames =  [item[0] for item in file_metadata]
    #sorted_filenames = [item[0] for item in sorted_metadata]
    print(f"{len(sorted_metadata)} files loaded.")

    return sorted_metadata

def get_intervals_in_tracks_df(tracks_df):
    dff = tracks_df.sort_values('time')

    # Calcola la differenza tra righe successive
    expected_freq = pd.Timedelta(hours=2)
    time_diff = dff['time'].diff()

    # Ogni volta che c'è un "buco", parte un nuovo gruppo
    dff['gruppo'] = (time_diff >= expected_freq).cumsum()

    # Ora trovi gli intervalli min e max per ogni gruppo
    intervalli = dff.groupby('gruppo')['time'].agg(['min', 'max']).reset_index(drop=True)

    return intervalli

def load_all_images_in_intervals(input_dir: str, intervals: pd.DataFrame):
    """
    Carica tutti i file .png in input_dir, li ordina per data estratta dal nome
    e filtra mantenendo solo quelli il cui timestamp rientra in uno degli intervalli.

    Args:
        input_dir: path della cartella contenente i .png
        intervals: DataFrame con colonne 'min' e 'max' di tipo datetime

    Returns:
        List of tuples (Path, datetime) ordinata per datetime, solo per i file validi.
    """
    # Prepariamo un IntervalIndex per membership test veloce
    interval_index = pd.IntervalIndex.from_arrays(
        intervals['min'], intervals['max'], closed='both')

    # 1) raccogliamo tutti i file
    filenames = get_files_from_folder(folder=input_dir, extension="png")
    file_metadata = []
    for fname in filenames:
        raw_dt = extract_dates_pattern_airmass_rgb_20200101_0000(fname.name)
        frame_dt = normalize_timestamp(raw_dt)
        if pd.isna(frame_dt):
            print(type(frame_dt), "non è un timestamp valido, skip")
            continue  # skip file non conforme al pattern
        # 2) verifichiamo se frame_dt è in uno degli intervalli
        if in_any_interval_via_index(frame_dt, interval_index):
            file_metadata.append((fname, frame_dt))

    # 3) ordiniamo per data
    sorted_metadata = sorted(file_metadata, key=lambda x: x[1])
    return sorted_metadata



def normalize_timestamp(dt):
    # se dt è già Timestamp o datetime, diventa Timestamp; altrimenti ValueError/NaT
    try:        
        return pd.to_datetime(dt)
    except Exception:
        return pd.NaT
    
#def is_valid_timestamp(dt):
#    return isinstance(dt, pd.Timestamp) and not pd.isna(dt)

def in_any_interval_via_index(dt, interval_index):
    # get_indexer su lista di un solo elemento 
    # (dt è un singolo pd.Timestamp)
    idx = interval_index.get_indexer([dt])[0]
    return idx != -1
