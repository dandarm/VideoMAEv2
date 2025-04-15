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