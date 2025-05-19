import pandas as pd
from build_dataset import get_gruppi_date, group_df_by_offsets

def aggiorna_label_distanza_temporale(df, label_col='label', soglia=pd.Timedelta(minutes=30)):
    """
    Modifica la colonna `label` di un DataFrame: pone a 0 le righe in cui `delta_time`
    è maggiore di una soglia specificata.
    """
    #label_updated_df = df.copy()
    # definisco la nuova colonna
    r = format_td_short(soglia)
    new_label_col = f"{label_col}_{r}"
    df[new_label_col] = df.apply(
        lambda row: 0 if pd.notna(row['delta_time']) and row['delta_time'] < soglia else row[label_col],
        axis=1
    )
    print(f"Colonna {new_label_col} aggiunta")
    #return label_updated_df

def format_timedelta_for_name(td):
    if pd.isna(td):
        return "NaT"
    
    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours:02d}h")
    if minutes:
        parts.append(f"{minutes:02d}m")
    if seconds:
        parts.append(f"{seconds:02d}s")

    return ''.join(parts)

def format_td_short(td):
    h = int(td.total_seconds() // 3600)
    m = int((td.total_seconds() % 3600) // 60)
    return f"{h:02d}_{m:02d}"



# definiamo una funzione uguale a create_tile_videos from build_dataset
# però che gestisce anche tutte le colonne che iniziano con label

def create_tile_videos_relabeling(grouped, num_frames = 16):
    """  Crea il VIDEO DATAFRAME con le informazioni per ogni video, 
        seguendo tutte le label aggiuntive date dal relabeling son shift temporale
    """    
    
    results = []
    video_id = 0
    
    
    for (offset_x, offset_y), group_df in grouped:
        label_cols = [col for col in group_df.columns if col.startswith('label')] 

        # group_df è un sotto-DataFrame con tutte le righe di quella tile
        # Ordinate già per datetime.
        group_df = group_df.reset_index(drop=True)
        row_count = len(group_df)
        num_blocks = row_count // num_frames  # quante volte possiamo formare un blocco di 16
        
        # se row_count non è multiplo di 16, rimarranno righe extra che ignoriamo (oppure gestisci diversamente)
        for i in range(num_blocks):
            start_i = i * num_frames
            end_i   = start_i + num_frames 
            block_df = group_df.iloc[start_i:end_i]
            start_time = group_df.datetime.iloc[start_i]
            end_time = group_df.datetime.iloc[end_i-1] # -1 perché nell'intervallo in block l'estremo sup non è compreso
            
            date_str = end_time.strftime("%d-%m-%Y_%H%M")
            path_name = f"{date_str}_{offset_x}_{offset_y}"
            
            diz = {}
            for label in label_cols:
                num_pos_labels = (block_df[label] == 1).sum()   # non più any()
                if num_pos_labels > num_frames/3:
                    diz[label] = 1
                else:
                    diz[label] = 0

            # A questo punto block_df ha 16 righe consecutive
            results.append({
                "video_id": video_id,
                "tile_offset_x": offset_x,
                "tile_offset_y": offset_y,
                "path": str(path_name),
                "label": label,
                "start_time": start_time,
                "end_time": end_time,                
                "orig_paths": block_df["path"].tolist(),
            } | diz)
            video_id += 1

    df_video = pd.DataFrame(results)
    print(f"Righe del df_video: {df_video.shape[0]}")
    return df_video

def create_df_video_from_master_df_relabeling(df_data, idxs=None):
    gruppi_date_list = get_gruppi_date(df_data)
    if idxs is None:
        idxs = range(len(gruppi_date_list))

    df_videos = []
    for i, df in enumerate(gruppi_date_list):
        if i in idxs:
            df_offsets_groups = group_df_by_offsets(df)
            df_for_period = create_tile_videos_relabeling(df_offsets_groups)

            #start_time = df_for_period.datetime.iloc[0]
            #print(f"{len(df_for_period)} video per il periodo da {start_time} \n")
            df_videos.append(df_for_period)

    df_videos = pd.concat(df_videos)
    return df_videos
