# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import csv
import re
from collections import defaultdict
from datetime import datetime
from time import time

import cv2
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from PIL import Image
#from mpl_toolkits.basemap import Basemap

from medicane_utils.load_files import load_cyclones_track_noheader, get_files_from_folder, extract_dates_pattern_airmass_rgb_20200101_0000
from medicane_utils.geo_const import latcorners, loncorners, x_center, y_center, default_basem_obj
from medicane_utils.geo_const import get_lon_lat_grid_2_pixel, trova_indici_vicini


from medicane_utils.load_files import load_all_images, get_all_cyclones
from medicane_utils.load_files import load_cyclones_track_noheader

from arguments import prepare_finetuning_args

#from view_test_tiles import plot_image, draw_tiles_and_center, create_gif_pil


# region old csv write
# def create_csv(output_dir):
#     # File CSV di output
#     train_csv = os.path.join(output_dir, "train.csv")
#     test_csv = os.path.join(output_dir, "test.csv")
#     val_csv = os.path.join(output_dir, "val.csv")

#     subfolders = sorted([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

#     total = len(subfolders)
#     train_split = int(total * 0.7)
#     test_split = int(total * 0.99)

#     train_dirs = subfolders[:train_split]
#     test_dirs = subfolders[train_split:test_split]
#     val_dirs = subfolders[test_split:]

#     # Scrive nei file CSV con il formato richiesto
#     def write_to_csv(dirs, csv_file):
#         with open(csv_file, 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["path", "start", "end"])  # Intestazione
#             for dir_path in dirs:
#                 writer.writerow([dir_path, 1, 16])  # Riga nel formato richiesto

#     write_to_csv(train_dirs, train_csv)
#     write_to_csv(test_dirs, test_csv)
#     write_to_csv(val_dirs, val_csv)

#     print(f"File CSV generati:\nTrain: {train_csv}\nTest: {test_csv}\nValidation: {val_csv}")
# endregion



##########################################################
###############  CREAZIONE TILES     #####################
##########################################################

# region

def calc_tile_offsets(image_width=1290, image_height=420, tile_size=224, stride_x=213, stride_y=196):
    """
    Ritorna una lista di (x_off, y_off) 
    """    
    offsets = []
    for y_off in range(0, image_height - tile_size + 1, stride_y):
        for x_off in range(0, image_width - tile_size + 1, stride_x):
            offsets.append((x_off, y_off))
    return offsets

# def get_dataset_offsets(frames_list, tile_size=224, stride=112):
#     """Prende tutti gli offsets di ogni immagine"""
#     all_offsets = []
#     for frame in frames_list:  # Itera su ogni frame (immagine PIL)
#         w, h = frame.size
#         offsets_this_frame = calc_tile_offsets(w, h, tile_size, stride)
#         all_offsets.append(offsets_this_frame)

#     return all_offsets

def create_tiles(frame, offsets_list, tile_size=224):
    w, h = frame.size    
    # costruisco le tiles
    tiles_this_frame = []
    for x,y in offsets_list:  
        tile = frame.crop((x, y, x + tile_size, y + tile_size))  # Usa crop() di PIL
        tiles_this_frame.append(tile)

    return tiles_this_frame
    
def save_single_tile(img_path, new_path, offset_x, offset_y, tile_size):
    frame_img = Image.open(img_path)
    crop = (offset_x, offset_y, offset_x + tile_size, offset_y + tile_size)
    tile = frame_img.crop(crop)
    tile.save(new_path)

# endregion

    
##########################################################
###################         #Logica per determinare se (lat, lon) cade dentro un tile 224×224
##########################################################

#region 

def compute_pixel_scale(big_image_w=1290, big_image_h=420):
    """
    Proietta i 4 corner in coordinate geostazionarie,
    trova Xmin,Xmax, Ymin,Ymax => calcola px_scale_x,y.
    Ritorna (x_min, y_min, px_scale_x, px_scale_y).
    """
    lat_min, lat_max = latcorners
    lon_min, lon_max = loncorners
    # Calling a Basemap class instance with the arguments lon, lat
    # will convert lon/lat (in degrees) to x/y map projection coordinates *(in meters)*:
    Xmin, Ymin = default_basem_obj(lon_min, lat_min)
    Xmax, Ymax = default_basem_obj(lon_max, lat_max)

    # quante "pixel su un metro" in orizzontale e verticale
    px_scale_x = big_image_w / (Xmax - Xmin)
    px_scale_y = big_image_h / (Ymax - Ymin)

    return Xmin, Ymin, px_scale_x, px_scale_y

def coord2px(lat, lon, px_per_m_x, px_per_m_y, Xmin, Ymin):
    # 1) Ottieni la proiezione "Xgeo, Ygeo" in metri (circa) 
    x_geo, y_geo = default_basem_obj(lon, lat)
    # 2) Sottrai offset
    Xlocal = x_geo - Xmin
    Ylocal = y_geo - Ymin
    # 3) Converti Xlocal, Ylocal in pixel
    x_pix = Xlocal * px_per_m_x
    y_pix = Ylocal * px_per_m_y

    y_pix = 420 - y_pix  # necessario per rovesciare lungo l'asse y.  420 = altezza immagine

    return x_pix, y_pix
    

# il codice di sopra è vecchio (e sbagliato)
# ->
lon_grid, lat_grid, x, y = get_lon_lat_grid_2_pixel(image_w=1290, image_h=420)

def get_cyclone_center_pixel(lat, lon, image_h=420):
    #Xmin, Ymin, px_scale_x, px_scale_y = compute_pixel_scale()
    #x_pix, y_pix = coord2px(lat, lon, px_scale_x, px_scale_y, Xmin, Ymin) 

    px, py = trova_indici_vicini(lon_grid, lat_grid, lon, lat)
    py = image_h - py

    return px, py

def inside_tile(lat, lon, tile_x, tile_y,
                tile_width=224, tile_height=224):
    """
    Verifica se la lat/lon (gradi) cade dentro i confini di un tile 224×224 
    definito in coordinate "pixel".
    
    1) Converti (lat,lon) in coordinate geostazionarie (m. stuff).
    2) Sottrai offset del sub-satellite point (x_center, y_center).
    3) Confronta con l'intervallo [tile_x, tile_x+tile_width] in coordinate pixel.
       -> serve una scala (px_per_km_x, px_per_km_y) o simile per passare da “metri geostazionari” a pixel.
    """
    #Xmin, Ymin, px_scale_x, px_scale_y = compute_pixel_scale()
    #x_pix, y_pix = coord2px(lat, lon, px_scale_x, px_scale_y, Xmin, Ymin)    

    x_pix, y_pix = get_cyclone_center_pixel(lat, lon)
    
    # Check se (x_pix, y_pix) cade nel tile 
    if (tile_x <= x_pix < tile_x + tile_width) and \
       (tile_y <= y_pix < tile_y + tile_height):
        return True
    else:
        return False

def inside_tile_faster(x_pix, y_pix, tile_x, tile_y,
                tile_width=224, tile_height=224):
    # Check se (x_pix, y_pix) cade nel tile 
    if (tile_x <= x_pix < tile_x + tile_width) and \
       (tile_y <= y_pix < tile_y + tile_height):
        return True
    else:
        return False



#endregion

# region old
def get_tile_labels(lat, lon):

    default_offsets_for_frame = calc_tile_offsets()
    labeled_tiles_offsets = [None] * len(default_offsets_for_frame)  # segue lo stesso ordine degli offsets_for_frame
    for i, (tile_offset_x, tile_offset_y) in enumerate(default_offsets_for_frame):
        #print(tile_offset_x, tile_offset_y)
        if inside_tile(lat, lon, tile_offset_x, tile_offset_y):
            #print("presente")
            labeled_tiles_offsets[i] = 1
        else:
            #print("assente")
            labeled_tiles_offsets[i] = 0

    return labeled_tiles_offsets
# endregion


def create_df_unlabeled_tiles_from_metadatafiles(sorted_metadata_files, offsets_for_frame):
    
    updated_metadata = []    

    for img_path, frame_dt in sorted_metadata_files:           
        
        for tile_offset_x, tile_offset_y in offsets_for_frame:
            
            updated_metadata.append({
                        "path": img_path,
                        "datetime": frame_dt,
                        "tile_offset_x": tile_offset_x,
                        "tile_offset_y": tile_offset_y,
                    })
            
    res =  pd.DataFrame(updated_metadata)
    
    res = res.astype({
        "path": 'string',
        "datetime": 'datetime64[ns]',
        "tile_offset_x": 'int16',
        "tile_offset_y": 'int16',
    })
    return res





# region CLOUD INDEX copertura nuvolosa delle tile

def threshold_image(img, thresh_val=150):
    green_channel = img[:,:,1]    
    cloud_mask = green_channel > thresh_val
    return cloud_mask

def cloud_idx(cloud_mask):
    v = cloud_mask.sum()/cloud_mask.size
    return round(v, 3)

def get_cloud_idx_from_image_path(image_path):
    #img = Image.open(image_path)
    try:
        img = cv2.imread(str(image_path))
    except:
        print(f"ERRORE DI LETTURA FILE {image_path}")
    try:
        if img is not None:
            mask = threshold_image(img)
            idx = cloud_idx(mask)
        else:
            idx = 0.0
    except:
        print("Errore numero di canali o immagine nulla")
        idx = 0.0
    
    return idx

def get_cloud_idx_from_img(img):
    mask = threshold_image(img)
    idx = cloud_idx(mask)
    return idx


# endregion




# region CLOUD INDEX on master_df (per-tile)

def get_cloud_idx_from_image_tile(image_path, offset_x, offset_y, tile_size=224):
    """
    Legge l'immagine `image_path`, ritaglia la tile definita da (offset_x, offset_y)
    di dimensione `tile_size` e calcola l'indice di nuvolosità su quella tile.
    Ritorna un float in [0,1]. In caso di errore, ritorna 0.0.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        tile = img[offset_y:offset_y + tile_size, offset_x:offset_x + tile_size]
        if tile.size == 0 or tile.shape[0] == 0 or tile.shape[1] == 0:
            return 0.0
        return get_cloud_idx_from_img(tile)
    except Exception:
        return 0.0


def add_cloud_idx_to_master_df(df_master, col_name='cloud_idx', tile_size=224, efficient=True):
    """
    Aggiunge al master DataFrame (righe per singola tile) una colonna con
    l'indice di nuvolosità calcolato sulla tile corrispondente.

    - df_master: DataFrame con colonne 'path', 'tile_offset_x', 'tile_offset_y'.
    - col_name: nome della nuova colonna (se già presente, non ricalcola nulla).
    - efficient: se True, legge ogni immagine una sola volta e calcola per le sue tile.
    """
    if col_name in df_master.columns:
        return df_master

    if not efficient:
        df_master[col_name] = df_master.apply(
            lambda r: get_cloud_idx_from_image_tile(r['path'], int(r['tile_offset_x']), int(r['tile_offset_y']), tile_size),
            axis=1
        )
        return df_master

    # Versione efficiente: raggruppa per path e legge l'immagine una volta sola
    values = []
    for path_val, sub in df_master.groupby('path', sort=False):
        img = cv2.imread(str(path_val))
        if img is None:
            vals = [0.0] * len(sub)
        else:
            # Calcola per tutte le tile di questa immagine
            offs = sub[['tile_offset_x', 'tile_offset_y']].astype('int32').to_numpy()
            vals = [
                get_cloud_idx_from_img(
                    img[y:y + tile_size, x:x + tile_size]
                ) if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] else 0.0
                for x, y in offs
            ]
        values.append(pd.Series(vals, index=sub.index))

    df_master[col_name] = pd.concat(values).sort_index()
    return df_master


def add_cloud_idx_to_master_df_file(master_df_csv_path, out_csv_path=None, col_name='cloud_idx', tile_size=224):
    """
    Carica un master_df da CSV, aggiunge la colonna dell'indice nuvoloso per tile se
    mancante e salva il risultato (stesso file o nuovo percorso se specificato).
    """
    df = pd.read_csv(master_df_csv_path, dtype={
        "path": 'string',
        "tile_offset_x": 'int16',
        "tile_offset_y": 'int16',
        "label": 'int8',
        "x_pix": 'object',
        "y_pix": 'object',
        "source": 'string',
        'id_cyc_unico': 'int32'
    }, parse_dates=['datetime', 'start_time', 'end_time'])
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)

    df = add_cloud_idx_to_master_df(df, col_name=col_name, tile_size=tile_size, efficient=True)

    out_path = out_csv_path if out_csv_path is not None else master_df_csv_path
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M")
    return out_path

# endregion


# crea il master dataframe
def labeled_tiles_from_metadatafiles(sorted_metadata_files, df_tracks, offsets_for_frame):   #, save_to_file=False):

    updated_metadata = []

    jj = 0
    for img_path, frame_dt in sorted_metadata_files:       
        # recupero la riga corrispondente all'ora intera dell'immagine
        # devo perci arrotondare (in eccesso o difetto?) l'istante dell'img
        # -> sto arrotondando per difetto
        dt_floor = frame_dt.replace(minute=0, second=0, microsecond=0) # ora con :00 di un'immagine che può avere minuti non :00
        mask = df_tracks["time"] == dt_floor
        df_candidates = df_tracks[mask]

        
        for tile_offset_x, tile_offset_y in offsets_for_frame:
            found_any = False
            source = []
            xp, yp = [], []
            id_cyc_unico = 0

            for row in df_candidates.itertuples(index=False):  
                # devo considerare il caso in cui ho più cicloni *** anche nella stessa tile! ***
                x_pix, y_pix = row.x_pix, row.y_pix
                s_ = row.source                

                if inside_tile_faster(x_pix, y_pix, tile_offset_x, tile_offset_y):
                    found_any = True
                    xp.append(x_pix)
                    yp.append(y_pix)
                    source.append(s_)
                    id_cyc_unico = row.id_cyc_unico  # al momento assumo che sia lo stesso ciclone


            label = 1 if found_any else 0
            # append UNA sola volta la tile           
            # quindi aggiungo un'immagine con minuti anche non :00 e la etichetto
            # in base al track che ha minuti :00
            updated_metadata.append({
                        "path": img_path,
                        "datetime": frame_dt,
                        "tile_offset_x": tile_offset_x,
                        "tile_offset_y": tile_offset_y,
                        "label": label,
                        "x_pix":xp,
                        "y_pix":yp,
                        "source": source,
                        "id_cyc_unico": id_cyc_unico
                    })
        jj += 1
        if jj % 1000 == 0:
            print(f"{jj} su {len(sorted_metadata_files)}")
            
    res =  pd.DataFrame(updated_metadata)  # il "master DataFrame"
    res = res.astype({
        "path": 'string',
        "datetime": 'datetime64[ns]',
        "tile_offset_x": 'int16',
        "tile_offset_y": 'int16',
        "label": 'category',
        "x_pix": 'object', # non più Int16
        "y_pix": 'object', # non più Int16
        "name": 'string',
        "source": 'string',
        "id_cyc_unico":  'int32'
    })
    return res


# crea il MASTER DATAFRAME (VELOCE)
def labeled_tiles_from_metadatafiles_maxfast(sorted_metadata_files, df_tracks, offsets_for_frame, tile_width=224, tile_height=224):
    """
    Versione massimamente vettorizzata di labeled_tiles_from_metadatafiles.
    - Prodotto cartesiano immagini × offset
    - Merge su "dt_floor" == "time"
    - Boolean indexing puro per inside
    - GroupBy per aggregare liste e label
    """
    # Assicura tipo datetime su df_tracks['time']
    df_tracks = df_tracks.copy()
    df_tracks['time'] = pd.to_datetime(df_tracks['time'])
    
    # 1) Metadata → DataFrame
    meta = pd.DataFrame(sorted_metadata_files, columns=['path', 'datetime'])
    meta['datetime'] = pd.to_datetime(meta['datetime'])
    meta['dt_floor'] = meta['datetime'].dt.round('h') # prima era floor, introduceva un errore inutile

    # 2) Offsets → DataFrame
    offs = pd.DataFrame(offsets_for_frame, columns=['tile_offset_x', 'tile_offset_y'])
    offs['key'] = 1
    meta['key'] = 1

    # 3) Prodotto cartesiano immagini × offsets
    meta_off = meta.merge(offs, on='key').drop('key', axis=1)

    # sample = meta_off.sample(5)
    # display(HTML(sample.to_html()))

    # display(HTML(df_tracks[df_tracks['time'].isin(sample['dt_floor'])].to_html()))


    # 4) Merge con df_tracks su dt_floor → time
    merged = meta_off.merge(
        df_tracks,
        left_on='dt_floor',
        right_on='time',
        how='left',
        #suffixes=('', '_track')
    )
    #display(HTML( merged[ merged['path'].isin(sample['path']) ].head().to_html() ))

    # 5) Calcola vettorialmente la maschera "inside"
    cond_x = (merged['x_pix'] >= merged['tile_offset_x']) & \
             (merged['x_pix'] <  merged['tile_offset_x'] + tile_width)
    cond_y = (merged['y_pix'] >= merged['tile_offset_y']) & \
             (merged['y_pix'] <  merged['tile_offset_y'] + tile_height)
    merged['inside'] = cond_x & cond_y

    # 6) Filtra solo i punti dentro la tile
    hits = merged[merged['inside']].copy()

    # 7) Raggruppa per costruire liste di xp, yp, source e prende il primo id
    grp = hits.groupby(
        ['path','datetime','tile_offset_x','tile_offset_y'],
        as_index=False
    ).agg({
        'x_pix': list,
        'y_pix': list,
        'source': list,
        'id_cyc_unico': 'first',
        'start_time': 'first',
        'end_time': 'first'
    })
    grp['label'] = 1

    # 8) Unisce con tutte le combinazioni per avere anche label=0
    all_tiles = meta_off[['path','datetime','tile_offset_x','tile_offset_y']].drop_duplicates()
    res = all_tiles.merge(
        grp,
        on=['path','datetime','tile_offset_x','tile_offset_y'],
        how='left'
    )

    # 9) Fillna e creazione colonne x_pix, y_pix, source per i tile vuoti
    res['label'] = res['label'].fillna(0)#.astype('category')
    for col in ['x_pix', 'y_pix', 'source']:
        res[col] = res[col].apply(lambda x: x if isinstance(x, list) else [])
    res['id_cyc_unico'] = res['id_cyc_unico'].fillna(0).astype('int32')
    res['start_time'] = res['start_time'].fillna(pd.NaT)
    res['end_time']   = res['end_time'].fillna(pd.NaT)


    ####### crea finalmente il master df
    # 10) Tipizza tutte le colonne per corrispondenza all'originale
    return res.astype({
        'path': 'string',
        'datetime': 'datetime64[ns]',
        'start_time': 'datetime64[ns]',
        'end_time': 'datetime64[ns]',
        'tile_offset_x': 'int16',
        'tile_offset_y': 'int16',
        'label': 'int8',
        'x_pix': 'object',
        'y_pix': 'object',
        'source': 'string',
        'id_cyc_unico': 'int32'
    })



#### temp
        # # però il nome non dovrebbe essere sempre quello per tutta l'immagine,
        # # perché esistono altri cicloni contemporanei al medicane
        # if 'Medicane' in df_candidates.columns:
        #     med_name = df_candidates['Medicane'].unique()
        #     if len(med_name) > 0:
        #         #print(med_name, flush=True)
        #         medicane_name = med_name[0]
        #     else:
        #         medicane_name = med_name
        # else:
        #     medicane_name = None
####



##########################################################
###############  CREA il VIDEO DATAFRAME   #####################
##########################################################

def group_df_by_offsets(df):
    # 1) Ordiniamo il DataFrame per (tile_offset_x, tile_offset_y, datetime)
    df_sorted = df.sort_values(["tile_offset_x", "tile_offset_y", "datetime"])

    # 2) Raggruppiamo per tile_offset
    grouped = df_sorted.groupby(["tile_offset_x", "tile_offset_y"], group_keys=False)

    return grouped




def create_tile_videos(grouped, output_dir=None, tile_size=224, supervised=True, num_frames = 16):
    """  Crea il VIDEO DATAFRAME con le informazioni per ogni video, 
    subito prima del csv per videoMAE

    non salva i video su cartella se non è specificata output_dir

    grouped è il df raggruppato per gli offsets:
      tile_offset_x, tile_offset_y, datetime, path, ...
    Ritorna una lista (videos_list) di DataFrame,
    ognuno con 16 righe consecutive. 
    """    
    
    results = []
    video_id = 0
    
    for (offset_x, offset_y), group_df in grouped:
        #assert 'label' in group_df.columns, "Manca la colonna label"
        # group_df è un sotto-DataFrame con tutte le righe di quella tile
        # Ordinate già per datetime.
        group_df = group_df.reset_index(drop=True)
        row_count = len(group_df)
        num_blocks = row_count // num_frames  # quante volte possiamo formare un blocco di 16
        
        # se row_count non è multiplo di 16, rimarranno righe extra che ignoriamo (oppure gestisci diversamente)
        #         
        for i in range(num_blocks):
            start_i = i * num_frames
            end_i   = start_i + num_frames 
            block_df = group_df.iloc[start_i:end_i]
            start_time = group_df.datetime.iloc[start_i]
            end_time = group_df.datetime.iloc[end_i-1] # -1 perché nell'intervallo in block l'estremo sup non è compreso
            
            date_str = end_time.strftime("%d-%m-%Y_%H%M")
            path_name = f"{date_str}_{offset_x}_{offset_y}"
            
            # se output dir è passato: salva le tile video in nuove cartelle
            if output_dir is not None:  
                subfolder = Path(output_dir) / path_name                              
                subfolder.mkdir(parents=True, exist_ok=True)

                for k, row_ in enumerate(block_df.itertuples(index=False)):
                    orig_path = row_.path  # colonna col path dell'immagine
                    new_name = subfolder / f"img_{k+1:05d}.png"

                    #default_offsets_list = calc_tile_offsets() # valori di default   w, h, tile_size, stride)
                    save_single_tile(orig_path, new_name, offset_x, offset_y, tile_size)
                    
    
            if supervised:
                #print(block_df['label'].dtype) l'ho messo a 'int'
                num_pos_labels = (block_df['label'] == 1).sum()   # non più any()
                if num_pos_labels > num_frames/3:
                    label = 1
                else:
                    label = 0
            else:
                label = None

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
            })
            video_id += 1

    return pd.DataFrame(results)


def create_tile_videos_last_frame_integer_hour(grouped, tile_size=224, supervised=True, num_frames=16):
    """  Crea il VIDEO DATAFRAME con le informazioni per ogni video, 
    OGni video finisce in corrispondenza di un'ora intera (HH:00) così da combaciare con i dati di Manos
    subito prima del csv per videoMAE

    grouped è il df raggruppato per gli offsets:
      tile_offset_x, tile_offset_y, datetime, path, ...
    Ritorna un DataFrame.
    """    
    frame_interval = pd.Timedelta(minutes=5)
    block_duration = num_frames * frame_interval  # 1h20'
    overlap = pd.Timedelta(minutes=20)  # equivale a 4 righe
    step = num_frames - int(overlap / frame_interval)  # 12 righe = 1h

    results = []
    video_id = 0
    
    for (offset_x, offset_y), group_df in grouped:
        blocks = []
        #print(f"offset_x, offset_y {offset_x, offset_y}")
        # group_df è un sotto-DataFrame con tutte le righe di quella singola tile
        # Ordinate già per datetime.
        group_df = group_df.reset_index(drop=True)        

        # Trova l'ultima riga con datetime esattamente su un'ora intera
        mask_on_the_hour = group_df['datetime'] == group_df['datetime'].dt.floor('h')
        # indici per le ore intere
        on_the_hour_indices = group_df[mask_on_the_hour].index

        saltati_inizio = 0
        saltati_fine = 0
        for end_idx in on_the_hour_indices:
            start_idx = end_idx - (num_frames - 1)
            if start_idx < 0:
                saltati_inizio += 1
                continue  # non abbastanza righe prima per formare il blocco

            block_df = group_df.iloc[start_idx:end_idx + 1].copy()
            #print(block_df.label.sum())

            # Verifica finale, opzionale: il datetime finale è proprio un'ora intera
            if block_df['datetime'].iloc[-1].minute != 0:
                saltati_fine += 1
                continue  # salta se non è perfettamente su ora intera

            blocks.append(block_df)
        #print(f"Blocchi video saltati: inziali {saltati_inizio}, finali: {saltati_fine}, aggiunti: {len(blocks)}")

    
        for i, block_df in enumerate(blocks):
            start_time = block_df['datetime'].iloc[0]
            end_time = block_df['datetime'].iloc[-1]
            #print(f"start_time {start_time} - end_time {end_time}")
            #print(f"offsets: {block_df[['tile_offset_x', 'tile_offset_y']].value_counts().index} - label: {block_df['label'].value_counts()}")
            #break
            
            date_str = end_time.strftime("%d-%m-%Y_%H%M")
            path_name = f"{date_str}_{offset_x}_{offset_y}"
    
            if supervised:
                #print(block_df['label'].dtype) #l'ho messo a 'int'
                num_pos_labels = (block_df['label']).sum()   # non più any()
                #print(num_pos_labels)
                if num_pos_labels > num_frames/3:
                    label = 1
                else:
                    label = 0
            else:
                label = None
    
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
            })
            video_id += 1

    return pd.DataFrame(results)





def create_and_save_tile_from_complete_df(df, output_dir, overwrite=False):
    num_video = df.shape[0]
    if num_video > 0:
        print(f"\nCreazione delle folder per i {num_video} video...", end='\t')

        salvati_ora = 0
        gia_salvati = 0
        totali = 0

        for idx, row in df.iterrows():
            # crea la cartella di destinazione
            path_name = row.path
            subfolder = Path(output_dir) / path_name
            #print(subfolder)
            subfolder.mkdir(parents=True, exist_ok=True)

            offset_x, offset_y = row.tile_offset_x, row.tile_offset_y
            for k, orig_p in enumerate(row.orig_paths):
                new_name = subfolder / f"img_{k+1:05d}.png"
                totali += 1
                #print(f"new_name {new_name}", end='\t')
                #print(f"è un file? {os.path.isfile(new_name)}")
                if not os.path.isfile(new_name) or overwrite:
                    #print(f"{new_name} lo sto risalvando?!")
                    save_single_tile(orig_p, new_name, offset_x, offset_y, tile_size=224)
                    salvati_ora += 1
                else:
                    #print(f"non c'è stato bisogno di risalvarlo")
                    gia_salvati += 1

        print(f"Salvati {salvati_ora} file - Erano già presenti {gia_salvati} file - File totali {totali}")






# region build dataset VideoMAE

def get_gruppi_date(df_data):
    # separo gruppi temporali contigui
    df_data = df_data.sort_values('datetime') 
    # Calcola la differenza temporale rispetto alla riga precedente
    df_data['delta'] = df_data['datetime'].diff()

    # Definisci i punti di rottura: True se la differenza è maggiore della frequenza attesa
    df_data['new_group'] = (df_data['delta'] > pd.Timedelta(minutes=60))  # 1h di intervallo massimo

    # Crea gli ID di gruppo cumulando i True
    df_data['gruppo'] = df_data['new_group'].cumsum()    

    gruppi_date = [g for _, g in df_data.groupby('gruppo')]

    return gruppi_date


def balance_time_group(df_videos, seed=1):    

    mask_cicloni = df_videos.label == 1
    mask_non_cicloni = df_videos.label == 0
    df_cicloni = df_videos[mask_cicloni]
    df_non_cicloni = df_videos[mask_non_cicloni]

    print(f"Num video CON cicloni: {len(df_cicloni)}, Num video SENZA cicloni: {len(df_non_cicloni)}", end="\t")
    print(f"Totale video tiles: {len(df_videos)}") #, end="\t\t")

    print(f"Bilanciamento video...", end="\t\t")
    df_0_balanced = df_non_cicloni.sample(len(df_cicloni), random_state=seed)
    print(f" video senza cicloni tenuti: {len(df_0_balanced)}")    

    return pd.concat([df_cicloni, df_0_balanced])


def create_df_video_from_master_df(df_data, idxs=None, output_dir=None, is_to_balance=False):
    gruppi_date_list = get_gruppi_date(df_data)
    if idxs is None:
        idxs = range(1, len(gruppi_date_list)+1)

    df_videos = []
    i = 1
    for df in gruppi_date_list:
        if i in idxs:
            print(f"{i})  ->")
            df_offsets_groups = group_df_by_offsets(df)            
            #df_for_period = create_tile_videos(df_offsets_groups)
            df_for_period = create_tile_videos_last_frame_integer_hour(df_offsets_groups)
            if df_for_period.shape[0] == 0:
                continue
            #assert 'label' in df_for_period.columns, f"Manca la colonna label - shape: {df_for_period.shape[0]}"
            
            if is_to_balance:
                df_for_period = balance_time_group(df_for_period)
            else: # comunque tengo traccia dei positivi e negativi
                print(f"Con cicloni: {(df_for_period.label == 1).sum()}")
                print(f"Senza cicloni: {(df_for_period.label == 0).sum()}")                

            if output_dir is not None:        
                create_and_save_tile_from_complete_df(df_for_period, output_dir)

            if not df_for_period.empty:
                start_time = df_for_period.start_time.min()
                end_time = df_for_period.end_time.max()
                print(f"{len(df_for_period)} video per il periodo (effettivo) da {start_time} a {end_time}\n")
                #print(f"start: {df.datetime.iloc[0]} \t end: {df.datetime.iloc[-1]}\n\n")
                df_videos.append(df_for_period)
            else:
                print(f"No video present for period: {df.datetime.iloc[0]} to {df.datetime.iloc[-1]}")
        i += 1

    if len(df_videos) > 0:
        df_videos = pd.concat(df_videos)
    else:
        df_videos = pd.DataFrame([])
    return df_videos


def create_final_df_csv(df_in, output_dir):
    df_dataset_csv = df_in[['path', 'label']].copy()
    df_dataset_csv['path'] = output_dir + df_dataset_csv['path']
    df_dataset_csv['start'] = 1
    df_dataset_csv['end'] = 16
    df_dataset_csv = df_dataset_csv[['path', 'start', 'end', 'label']]
    return df_dataset_csv




#region old code



def split_into_tiles_subfolders_and_track_dates(
    sorted_filenames,  # lista di (filepath, dt) ORDINATI cronologicamente
    output_dir,
    tile_width=224,
    tile_height=224,
    stride_x=112, # deveessere di 98:  [0, 98, 196] -> [(0,224), (98, 322), (196, 420)]
    stride_y=112,
    num_frames=16,
    big_image_w=1290, 
    big_image_h=420,
):
    """
    Crea sub‐cartelle "part1_0_0", "part2_0_0", ecc., ognuna con 16 frame
    crop 224x224. L'offset si sposta di stride_x, stride_y.
    Ritorna subfolder_info = lista di dict:
       {
         'folder': path_della_subcartella,
         'datetimes': [ dt1, dt2, ..., dt16 ],
         'tile_x': X_off,
         'tile_y': Y_off
       }
    """

    subfolder_info = []
    num_total_files = len(sorted_filenames)
    # quanti blocchi di 16 consecutivi
    num_subfolders = num_total_files // num_frames
    print(f"num_subfolders video: {num_subfolders}")

    iteration=0
    # Per ogni tile offset in (0.. big_image_w-224, step stride_x)
    for tile_y in range(0, big_image_h - tile_height + 1, stride_y):
        for tile_x in range(0, big_image_w - tile_width + 1, stride_x):
            print(iteration, end=" ", flush=True)
            print(f"tilex {tile_x} \t tiley {tile_y}", end=" ", flush=True)
            iteration += 1

            # Ora scandiamo i blocchi di 16 frame
            part_index = 1
            for i in range(num_subfolders):
                subfolder_name = f"part{part_index}_{tile_x}_{tile_y}"
                part_index += 1
                subfolder_path = os.path.join(output_dir, subfolder_name)
                #os.makedirs(subfolder_path, exist_ok=True)  TODO: rimettere

                start_idx = i * num_frames
                end_idx = start_idx + num_frames
                # Catturo i 16 file
                block_files = sorted_filenames[start_idx:end_idx]

                # Copia/crop i 16 frame nella sottocartella
                dt_list = []
                for idx, (img_path, frame_dt) in enumerate(block_files):
                    new_name = os.path.join(subfolder_path, f"img_{idx+1:05d}.png")

                # TODO: rimettere poi!!!    # Faccio open + crop
                #    with Image.open(img_path) as im:
                #        crop_region = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)
                #        cropped_im = im.crop(crop_region)
                #        cropped_im.save(new_name)

                    dt_list.append(frame_dt)

                # Salvo in subfolder_info
                subfolder_info.append({
                    "folder": subfolder_path,
                    "datetimes": dt_list,  # 16 dt
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                })
                
            print(f"part_index {part_index-1}")


    return subfolder_info


def label_subfolders_with_cyclones_df(
    subfolder_info,
    df_tracks,
    basemap_obj,
    x_center, y_center,
    px_scale_x, px_scale_y,
):
    """
    Per ciascuna voce di subfolder_info, calcola una label (0/1) se almeno
    un frame in quell'intervallo ha un centro ciclone dentro la tile.

    Ritorna un DataFrame con colonne [folder, start, end, label].
    un CSV: [folder, start, end, label].
    """
    # prepara groupby
    df_tracks['year'] = df_tracks['time'].dt.year
    df_tracks['month']= df_tracks['time'].dt.month
    df_tracks['day']  = df_tracks['time'].dt.day
    df_tracks['hour'] = df_tracks['time'].dt.hour

    cyc_groups = df_tracks.groupby(['year','month','day','hour'])

    results = []  # accumuliamo i risultati in una lista di dict


    for info in subfolder_info:
        folder_path = info['folder']
        tile_x = info['tile_x']
        tile_y = info['tile_y']
        dt_list = info['datetimes']

        found = False
        for dt_ in dt_list:
            if dt_ is None:
                continue
            k = (dt_.year, dt_.month, dt_.day, dt_.hour)

            if k in cyc_groups.indices:
                idxes = cyc_groups.indices[k]
                group_df = df_tracks.iloc[idxes]
                for row_ in group_df.itertuples(index=False):
                    if inside_tile(row_.lat, row_.lon, tile_x, tile_y,
                                    224, 224,
                                    basemap_obj,
                                    x_center, y_center,
                                    px_per_km_x=px_scale_x, 
                                    px_per_km_y=px_scale_y):
                        found = True
                        break
            if found:
                break

        label = 1 if found else 0
        results.append({
            "folder": folder_path,
            "start": 1,      # fissi 1 e 16 come "start" e "end"
            "end":   16,
            "label": label,
            "start_time": dt_list[0],
            "end_time": dt_list[-1]
    
        })

    
    df_out = pd.DataFrame(results, columns=["path","start","end","label","start_time","end_time"])
    return df_out
    
#endregion




#region ######## CODE FOR TERMINAL EXECUTION 
 



def make_unsup_dataset(input_dir, output_dir):

    #input_dir = "../fromgcloud"
    #output_dir = "../airmassRGB/supervised/" 
    #unsup_output_dir = "../airmassRGB/unsupervised/" 

    #expanded_path = os.path.expandvars(path)

    #creo il file all_data_unsup.csv
    sorted_metadata_files = load_all_images(input_dir)
    offsets_for_frame = calc_tile_offsets(stride_x=213, stride_y=196) # provo con 213 per coprire anche l'area più a destra
    df_data_unsup = create_df_unlabeled_tiles_from_metadatafiles(sorted_metadata_files, offsets_for_frame)
    df_data_unsup.to_csv("all_data_unsup.csv")
     

    nome_file = "all_data_unsup.csv"
    df_data = pd.read_csv(nome_file, dtype={
            "path": 'string',
            "tile_offset_x": 'int16',
            "tile_offset_y": 'int16',
        }, parse_dates=['datetime'])
    df_data.drop(columns="Unnamed: 0", inplace=True)

    #df_data = df_data[:1000]

    gruppi_date = get_gruppi_date(df_data)
    all_videos = []
    for df in gruppi_date:
        df_offsets_groups = group_df_by_offsets(df)
        df_videos = create_tile_videos(df_offsets_groups, supervised=False)
        all_videos.append(df_videos)
        create_and_save_tile_from_complete_df(df_videos, output_dir)
    
    all_df_videos = pd.concat(all_videos)
    df_dataset_csv_unsup = create_final_df_csv(all_df_videos, output_dir)
    df_dataset_csv_unsup.drop(columns='label').to_csv("./train_UNsupervised.csv", index=False)


def make_sup_dataset(input_dir, output_dir):
    #input_dir = "$FAST/Medicanes_Data/fromgcloud"
    #output_dir = "$FAST/airmass/"  # uso la stessa cartella, poi cambierà il csv
    #unsup_output_dir = "../airmassRGB/unsupervised/" 
    output_dir = solve_paths(output_dir)

    from .data_manager import BuildDataset
    from dataset.dataset_labeling_study import aggiorna_label_distanza_temporale

    #### TRAIN
    sup_data_train = BuildDataset(type='SUPERVISED', master_df_path="all_data_CL7_tracks_complete_fast.csv")
    sup_data_train.load_master_df()
    sup_data_train.get_sequential_periods()
    sup_data_train.print_sequential_periods()

    sup_data_train.make_df_video(output_dir, idxs=[1,2,3,4,5,6,7,8], is_to_balance=True)
    cicloni = sup_data_train.df_video.label.sum()
    totali = sup_data_train.df_video.shape[0]
    print(cicloni, totali)
    sup_data_train.create_final_df_csv(output_dir, f"train_dataset_{totali}.csv")

    #### TEST
    sup_data_test = BuildDataset(type='SUPERVISED', master_df_path="all_data_CL7_tracks_complete_fast.csv")
    sup_data_test.load_master_df()
    sup_data_test.make_df_video(output_dir, idxs=[9], is_to_balance=True)
    cicloni = sup_data_train.df_video.label.sum()
    totali = sup_data_train.df_video.shape[0]
    print(cicloni, totali)
    sup_data_test.create_final_df_csv(output_dir, f"test_dataset_{totali}.csv")


def calc_avg_cld_idx(video_subfolder):
    """
    Calcola l'indice di nuvolosità medio per il video nella cartella video_subfolder.
    Assume che ci siano 16 frame, img_00001.png, ..., img_00016.png e calcola l'indice per ciascuna immagine.
    0 = cielo sereno, 1 = completamente nuvoloso.
    Ritorna float tra 0 e 1.
    """
    frames_cld_idx = []
    for k in range(16):
        frame_path = Path(video_subfolder) / f"img_{k+1:05d}.png"        
        if not os.path.exists(frame_path):
            print(f"file non esistente? {frame_path}")
        #display(Image.open(frame_path))
        cidx = get_cloud_idx_from_image_path(frame_path)
        #print(cidx)
        frames_cld_idx.append(cidx)
    frames_cld_idx = np.array(frames_cld_idx)

    avg = frames_cld_idx.mean()
    return avg


def train_test_split(df_video, train_p=0.7):
    """
    Divide il DataFrame df_video in due parti: train e test.
    train_p è la percentuale di dati da usare per il training.
    """
    len_p = int(train_p * df_video.shape[0])
    df_video_train = df_video.sort_values('start_time').iloc[:len_p]
    df_video_test = df_video.sort_values('start_time').iloc[len_p:]
    print(f"Train e test lengths: {df_video_train.shape[0]}, {df_video_test.shape[0]}")
    return df_video_train, df_video_test

def train_test_cyclones_num_split(tracks_df, train_p, id_col):
    """
    Divide il DataFrame tracks_df in due parti di train e test.
    basato sul numero di cicloni unici (id_cyc_unico).
    """
    cicloni_unici = tracks_df[id_col].unique()    
    len_p = int(train_p*cicloni_unici.shape[0])
    cicloni_unici_train = cicloni_unici[:len_p]
    cicloni_unici_test = cicloni_unici[len_p:]
    print(f"Cicloni nel train: {cicloni_unici_train.shape[0]}, cicloni nel test: {cicloni_unici_test.shape[0]}")
    return cicloni_unici_train,cicloni_unici_test

def make_relabeled_master_df(data_manager, hours_shift=12):
    from dataset.dataset_labeling_study import aggiorna_label_distanza_temporale
    data_manager.calc_delta_time()    
    new_label = aggiorna_label_distanza_temporale(data_manager.master_df, soglia=pd.Timedelta(hours=hours_shift), sub_lab=-1)
    m = data_manager.master_df[new_label] == -1
    df_mod = data_manager.master_df[~m].copy().drop(columns='label').rename(columns={new_label:'label'})
    assert 'label' in df_mod.columns, "Manca la colonna label"
    return df_mod

def make_relabeled_dataset(input_dir, output_dir, cloudy=False, 
                           master_df_path="all_data_CL7_tracks_complete_fast.csv", hours_shift=12):
    output_dir = solve_paths(output_dir)
    from .data_manager import BuildDataset
    
    sup_data = BuildDataset(type='SUPERVISED', master_df_path=master_df_path)
    sup_data.load_master_df()

    df_mod = make_relabeled_master_df(sup_data, hours_shift=hours_shift)
    sup_data.make_df_video(new_master_df=df_mod, output_dir=output_dir,  is_to_balance=True) #, idxs=[1,2,3,4,5,6,7,8])
    df_v = sup_data.df_video.copy()

    suffix = f"_{hours_shift}h"
    if cloudy:
        df_v = filter_out_clear_sky(output_dir, sup_data)
        suffix += "_cloudy"


    #train e test
    df_video_train, df_video_test = train_test_split(df_v, train_p=0.7)


    # salva i csv
    df_dataset_csv = create_final_df_csv(df_video_train, output_dir)
    df_dataset_csv.to_csv(f"train_dataset{suffix}_{df_video_train.shape[0]}.csv", index=False)
    df_dataset_csv = create_final_df_csv(df_video_test, output_dir)
    df_dataset_csv.to_csv(f"test_dataset{suffix}_{df_video_test.shape[0]}.csv", index=False)

def filter_out_clear_sky(output_dir, sup_data, idx_thresh=0.1):
    sup_data.df_video.path = output_dir + sup_data.df_video.path
    new_col_name = "avg_cloud_idx"

    print(f"Calcolando l'indice di nuvolosità...")
    # calcolo l'indice di nuvolosità medio per ogni video
    sup_data.df_video[new_col_name] = sup_data.df_video.path.apply(calc_avg_cld_idx)
    # filtro i video con indice di nuvolosità medio > 0.1
    mask_cloud = sup_data.df_video.avg_cloud_idx > idx_thresh
    df_video_cloudy = sup_data.df_video[mask_cloud]
    # rispristino i path relativi
    df_v = df_video_cloudy.copy()
    df_v.path = df_v.path.str.split('/').str[-1]
    return df_v

def mark_neighboring_tiles(df_video, stride_x=213, stride_y=196, include_diagonals=True, only_negatives=True, time_key='end_time'):
    """
    Aggiunge una colonna boolean "neighboring" che vale True per le tile
    adiacenti (4- o 8-neighborhood) ad almeno una tile positiva (label == 1),
    calcolate all'interno dello stesso istante temporale del video (per default, end_time).

    - df_video: DataFrame prodotto da create_df_video_from_master_df / create_tile_videos*
    - stride_x, stride_y: passi della griglia degli offset (in pixel)
    - include_diagonals: se True usa l'8-neighborhood, altrimenti solo 4-neighborhood
    - only_negatives: se True non marca come neighboring le tile positive stesse
    - time_key: chiave temporale su cui ragionare (default 'end_time')

    Ritorna una copia di df_video con la colonna "neighboring" (dtype bool).
    """
    if df_video is None or df_video.empty:
        if df_video is None:
            return df_video
        out = df_video.copy()
        out['neighboring'] = False
        return out

    required_cols = {time_key, 'tile_offset_x', 'tile_offset_y', 'label'}
    missing = required_cols - set(df_video.columns)
    if missing:
        raise KeyError(f"Mancano colonne richieste in df_video: {missing}")

    df = df_video.copy()

    # Prepara i delta degli adiacenti rispetto al centro
    steps = [-1, 0, 1]
    shifts = [(dx, dy) for dx in steps for dy in steps if not (dx == 0 and dy == 0)]
    if not include_diagonals:
        shifts = [(dx, dy) for dx, dy in shifts if abs(dx) + abs(dy) == 1]

    # Seleziona le tile positive
    pos = df[df['label'] == 1][[time_key, 'tile_offset_x', 'tile_offset_y']].copy()
    if pos.empty:
        df['neighboring'] = False
        return df

    # Genera tutte le posizioni adiacenti alle positive (stesso time_key)
    neighbor_frames = []
    for dx, dy in shifts:
        t = pos.copy()
        t['tile_offset_x'] = t['tile_offset_x'] + dx * stride_x
        t['tile_offset_y'] = t['tile_offset_y'] + dy * stride_y
        neighbor_frames.append(t)

    neighbors_df = pd.concat(neighbor_frames, ignore_index=True).drop_duplicates()

    # Merge per marcare le righe di df che coincidono con una posizione adiacente ad una positiva
    key_cols = [time_key, 'tile_offset_x', 'tile_offset_y']
    neighbors_df['neighboring'] = True
    out = df.merge(neighbors_df[key_cols + ['neighboring']], on=key_cols, how='left')
    out['neighboring'] = out['neighboring'].fillna(False).astype(bool)

    # Opzionale: non marcare come neighboring le tile positive stesse
    if only_negatives:
        out.loc[out['label'] == 1, 'neighboring'] = False

    return out


def get_train_test_df(tracks_df, percentage=0.7, id_col='id_cyc_unico', verbose=True):
    cicloni_unici_train, cicloni_unici_test = train_test_cyclones_num_split(tracks_df, train_p=percentage, id_col=id_col)
    tracks_df_train = tracks_df[tracks_df[id_col].isin(cicloni_unici_train)]
    tracks_df_test = tracks_df[tracks_df[id_col].isin(cicloni_unici_test)]
    print(f"Train e test lengths: {tracks_df_train.shape[0]}, {tracks_df_test.shape[0]}")

    if verbose:
        u_train = tracks_df_train.groupby(tracks_df_train[id_col]).apply('first')  
        print((u_train.end_time - u_train.start_time).sum())
        u_test = tracks_df_test.groupby(tracks_df_test[id_col]).apply('first')  
        print((u_test.end_time - u_test.start_time).sum())

    return tracks_df_train, tracks_df_test

def get_train_test_validation_df(tracks_df, percentage=0.7, validation_percentage=0.15, id_col='id_final', verbose=True):
    """
    Divide il DataFrame tracks_df in tre parti: train, test e validation.
    percentage è la percentuale di dati da usare per il training.
    validation_percentage è la percentuale di dati da usare per la validazione.
    """

    cicloni_unici = tracks_df[id_col].unique()    
    len_p = int(percentage*cicloni_unici.shape[0])
    len_t = int((percentage + validation_percentage) * cicloni_unici.shape[0])
    cicloni_unici_train = cicloni_unici[:len_p]
    cicloni_unici_test = cicloni_unici[len_p:len_t]
    cicloni_unici_validation = cicloni_unici[len_t:]
    print(f"Cicloni nel train: {cicloni_unici_train.shape[0]}, cicloni nel test: {cicloni_unici_test.shape[0]}, cicloni nella validation: {cicloni_unici_validation.shape[0]}")

    tracks_df_train = tracks_df[tracks_df[id_col].isin(cicloni_unici_train)]    
    tracks_df_test = tracks_df[tracks_df[id_col].isin(cicloni_unici_test)]
    tracks_df_validation = tracks_df[tracks_df[id_col].isin(cicloni_unici_validation)]

    if verbose:
        print(f"Train rows: {tracks_df_train.shape[0]}, Test rows: {tracks_df_test.shape[0]}, Validation rows: {tracks_df_validation.shape[0]}")
        u_train = tracks_df_train.groupby(tracks_df_train[id_col]).apply('first')  
        print((u_train.end_time - u_train.start_time).sum())
        u_test = tracks_df_test.groupby(tracks_df_test[id_col]).apply('first')  
        print((u_test.end_time - u_test.start_time).sum())
        u_val = tracks_df_validation.groupby(tracks_df_validation[id_col]).apply('first')  
        print((u_val.end_time - u_val.start_time).sum())

    return tracks_df_train, tracks_df_test, tracks_df_validation

def make_dataset_from_manos_tracks(manos_track_file, input_dir, output_dir):
    # vecchio file di manos "manos_CL10_pixel.csv"    
    from dataset.data_manager import BuildDataset
    args = prepare_finetuning_args()

    output_dir = solve_paths(output_dir)
    input_dir = solve_paths(input_dir)

    tracks_df = pd.read_csv(manos_track_file, parse_dates=['time', 'start_time', 'end_time'])

    # divido le track di Manos in train e test
    #tracks_df_train, tracks_df_test = get_train_test_df(tracks_df, percentage=0.7)
    tracks_df_train, tracks_df_test, tracks_df_val = get_train_test_validation_df(tracks_df, 0.7, 0.15)

    print("Building training set...")
    train_b = BuildDataset(type='SUPERVISED', args=args)
    train_b.get_data_ready(tracks_df_train, input_dir, output_dir, csv_file="train_manos_unbalanced", is_to_balance=False)
    print("Building test set...")
    test_b = BuildDataset(type='SUPERVISED', args=args)
    test_b.get_data_ready(tracks_df_test, input_dir, output_dir, csv_file="test_manos_w")
    print("Building validation set...")
    val_b = BuildDataset(type='SUPERVISED', args=args)
    val_b.get_data_ready(tracks_df_val, input_dir, output_dir, csv_file="val_manos_w", is_to_balance=False)

    return train_b, tracks_df_train





def make_master_df(input_dir, output_dir):
    input_dir = solve_paths(input_dir)
    tracks_df_MED_CL7 = pd.read_csv("./manos_CL7_pixel.csv", parse_dates=['time', 'start_time', 'end_time'])
    sorted_metadata_files = load_all_images(input_dir)
    offsets_for_frame = calc_tile_offsets(stride_x=213, stride_y=196)
    df_data_CL7 = labeled_tiles_from_metadatafiles_maxfast(sorted_metadata_files, tracks_df_MED_CL7, offsets_for_frame)
    # Aggiunge la colonna cloud_idx per tile se non presente
    df_data_CL7 = add_cloud_idx_to_master_df(df_data_CL7, col_name='cloud_idx', tile_size=224, efficient=True)
    df_data_CL7.to_csv("./all_data_CL7_tracks_complete_fast.csv", date_format="%Y-%m-%d %H:%M")


def solve_paths(path):
    exp_path = os.path.expandvars(path)
    if '$' in exp_path:
        raise EnvironmentError(f"Errore con una variabile d'ambiente in {exp_path}")
    return exp_path




# endregion
