from pathlib import Path
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from common_tools import df_seviri_spectra, load_date_medicanes
#date_medicanes = load_date_medicanes(relative='../')

def convert_data(sample_date_str):
    sample_date = datetime.strptime(sample_date_str, "%Y%m%d/%H:%M")
    return sample_date

def check_date_in_range(date_medicanes, date_to_check):
    # TODO: restituire anche il nome del medicane
    is_in_range = any(
        (row['Start_Date'] <= date_to_check <= row['End_Date']) for _, row in date_medicanes.iterrows()
    )
    return is_in_range

def is_time_for_medicane(d):
    sample_date_str = d.date_time
    date_to_check = convert_data(sample_date_str)
    is_in_range = check_date_in_range(date_medicanes, date_to_check)
    return is_in_range

def extract_dates(filename, pattern):
    """
    Estrae le date di inizio e fine acquisizione dal nome del file.
    
    Esempio di nome file:
    HRSEVIRI_RSS_20200914T014510Z_20200914T014916Z_epct_6b03532f_FC.nc
    """
    pattern = r'HRSEVIRI_RSS_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_epct_[a-f0-9]+_FC\.nc'
    match = re.match(pattern, filename)
    if match:
        start_str, end_str = match.groups()       
       
        # Converte le stringhe in oggetti datetime
        start_dt = datetime.strptime(start_str, '%Y%m%dT%H%M%SZ')
        end_dt = datetime.strptime(end_str, '%Y%m%dT%H%M%SZ')
        return start_dt, end_dt

    else:
        return None, None
    
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
        
    

    
def get_dates_from_filename(root_directory):
    nc_files = list(root_directory.rglob('*.nc'))
    data = []
    for file in nc_files:
        start_dt, end_dt = extract_dates(file.name)
        if start_dt and end_dt:
            data.append({'file_path': file, 'start_time': start_dt, 'end_time': end_dt})
        else:
            print(f"Nome file non conforme al pattern: {file.name}")
            
    return data




def elaborate(data, expected_interval, tolerance):
    df = pd.DataFrame(data)
    print(df['start_time'][df.shape[0]//2].date(), end='\t')
    df = df.sort_values('start_time').reset_index(drop=True)

    # Calcola la differenza tra i tempi di inizio consecutivi
    df['time_diff'] = df['start_time'].diff()
    
    df['gap'] = df['time_diff'] > (expected_interval + tolerance)
    
    # Calcola il numero di file mancanti per ogni gap
    df['missing_files'] = np.where(
        df['gap'],
        (df['time_diff'] // expected_interval) - 1,
        0
    )
    # Assicurati che i valori negativi non siano presenti
    df['missing_files'] = df['missing_files'].clip(lower=0)
    total_missing = df['missing_files'].sum()
    expected_files = 24 * 60 // 5  # 288 file
    missing_percentage = (total_missing / expected_files) * 100
    print(f"Missing files: {int(total_missing)}, {missing_percentage:.2f}%")
    
    missing_time_minutes = total_missing * 5  # 5 minuti per file
    total_minutes_day = 24 * 60  # 1440 minuti
    missing_time_percentage = (missing_time_minutes / total_minutes_day) * 100
    #print(f"Total missing time: {missing_time_minutes} minutes")
    #print(f"Missing time over the day %: {missing_time_percentage:.2f}%")

    gaps = df[df['gap']]
    #if not gaps.empty:
    #    print(f"\n{len(gaps)} missing intervals:")
    #    for idx, row in gaps.iterrows():
    #        print(f"From {row['start_time'] - row['time_diff']} to {row['start_time']} (Interval: {row['time_diff']})")
    #else:
    #    print("\n No missing data.")
        
    return df, total_missing
    
    
    
def plot_all(df):

    # 10. Visualizza la distribuzione degli intervalli
    """
    plt.figure(figsize=(12, 6))
    plt.hist(df['time_diff'].dropna().dt.total_seconds() / 60, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(expected_interval.total_seconds() / 60, color='red', linestyle='dashed', linewidth=2, label=f'Intervallo atteso: {expected_interval}')
    plt.xlabel('Intervallo tra acquisizioni (minuti)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione degli intervalli tra acquisizioni')
    plt.yscale('log')
    plt.legend()
    plt.show()
    """
    
    # 11. Opzionale: Visualizza una serie temporale degli intervalli
    plt.figure(figsize=(20, 5))
    ax = plt.gca()
    plt.plot(df['start_time'], df['time_diff'].dt.total_seconds() / 60, marker='o', linestyle='-')
    time_formatter = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(time_formatter)
    #plt.axhline(expected_interval.total_seconds() / 60, color='red', linestyle='dashed', linewidth=2, label=f'Intervallo atteso: {expected_interval}')
    plt.xlabel('Time')
    plt.ylabel('Intervals (minutes)')
    plt.title(f"Collection time intervals - {df['start_time'][df.shape[0]//2].date()}")
    plt.ylim(0,500)
    #plt.yscale('log')
    #plt.legend()
    plt.show()
