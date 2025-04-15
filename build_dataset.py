# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import csv
import re
from collections import defaultdict
from datetime import datetime
from time import time

import pandas as pd
from PIL import Image
#from mpl_toolkits.basemap import Basemap

from medicane_utils.load_files import load_cyclones_track_noheader, get_files_from_folder, extract_dates_pattern_airmass_rgb_20200101_0000
from medicane_utils.geo_const import latcorners, loncorners, x_center, y_center, basemap_obj


from medicane_utils.load_files import load_all_images, get_all_cyclones
from medicane_utils.load_files import load_cyclones_track_noheader

from view_test_tiles import plot_image, draw_tiles_and_center, create_gif_pil



def create_csv(output_dir):
    # File CSV di output
    train_csv = os.path.join(output_dir, "train.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    val_csv = os.path.join(output_dir, "val.csv")

    subfolders = sorted([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

    total = len(subfolders)
    train_split = int(total * 0.7)
    test_split = int(total * 0.99)

    train_dirs = subfolders[:train_split]
    test_dirs = subfolders[train_split:test_split]
    val_dirs = subfolders[test_split:]

    # Scrive nei file CSV con il formato richiesto
    def write_to_csv(dirs, csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["path", "start", "end"])  # Intestazione
            for dir_path in dirs:
                writer.writerow([dir_path, 1, 16])  # Riga nel formato richiesto

    write_to_csv(train_dirs, train_csv)
    write_to_csv(test_dirs, test_csv)
    write_to_csv(val_dirs, val_csv)

    print(f"File CSV generati:\nTrain: {train_csv}\nTest: {test_csv}\nValidation: {val_csv}")
    



##########################################################
###############  CREAZIONE TILES     #####################
##########################################################



def calc_tile_offsets(image_width=1290, image_height=420, tile_size=224, stride_x=224, stride_y=196):
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
    Xmin, Ymin = basemap_obj(lon_min, lat_min)
    Xmax, Ymax = basemap_obj(lon_max, lat_max)

    # quante "pixel su un metro" in orizzontale e verticale
    px_scale_x = big_image_w / (Xmax - Xmin)
    px_scale_y = big_image_h / (Ymax - Ymin)

    return Xmin, Ymin, px_scale_x, px_scale_y

def coord2px(lat, lon, px_per_m_x, px_per_m_y, Xmin, Ymin):
    # 1) Ottieni la proiezione "Xgeo, Ygeo" in metri (circa) 
    x_geo, y_geo = basemap_obj(lon, lat)
    # 2) Sottrai offset
    Xlocal = x_geo - Xmin
    Ylocal = y_geo - Ymin
    # 3) Converti Xlocal, Ylocal in pixel
    x_pix = Xlocal * px_per_m_x
    y_pix = Ylocal * px_per_m_y

    y_pix = 420 - y_pix  # necessario per rovesciare lungo l'asse y.  420 = altezza immagine

    return x_pix, y_pix
    

def get_cyclone_center_pixel(lat, lon):
    Xmin, Ymin, px_scale_x, px_scale_y = compute_pixel_scale()
    x_pix, y_pix = coord2px(lat, lon, px_scale_x, px_scale_y, Xmin, Ymin) 
    return int(x_pix), int(y_pix)

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

#endregion


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


# crea i video e salva in cartelle partX e traccia la data dei video
def labeled_tiles_from_metadatafiles(sorted_metadata_files, df_tracks):   #, save_to_file=False):

    updated_metadata = []
    
    # Constants
    default_offsets_for_frame = calc_tile_offsets()

    for img_path, frame_dt in sorted_metadata_files:       
        # recupero la riga corrispondente all'ora intera dell'immagine
        # devo perci arrotondare (in eccesso o difetto?) l'istante dell'img
        dt_floor = frame_dt.replace(minute=0, second=0, microsecond=0)
        mask = df_tracks["time"] == dt_floor
        df_candidates = df_tracks[mask]
        # il nome è sempre quello per tutta l'immagine
        med_name = df_candidates['Medicane'].unique()
        if len(med_name) > 0:
            #print(med_name, flush=True)
            medicane_name = med_name[0]
        else:
            medicane_name = med_name
        
        for tile_offset_x, tile_offset_y in default_offsets_for_frame:
            found_any = False
            lat = None
            lon = None
            
            for row in df_candidates.itertuples(index=False):  # devo considerare il caso in cui ho più cicloni        
                lat_, lon_ = row.lat, row.lon           #df_candidates[['lat', 'lon']].values[0]
                #print(lat_, lon_, tile_offset_x, tile_offset_y, frame_dt)
                if inside_tile(lat_, lon_, tile_offset_x, tile_offset_y):
                    found_any = True
                    lat = lat_
                    lon = lon_                    
                    #print("trovato!\n")
                    break  # TODO: verificare che non sia dannoso

            label = 1 if found_any else 0
            # append UNA sola volta la tile
            # associando lat/lon del primo ciclone trovato (se c'è)
            if lat is not None:
                x_pix, y_pix = get_cyclone_center_pixel(lat, lon)
            else:
                x_pix, y_pix = None, None
            
            #print(lat, lon)
            
            updated_metadata.append({
                        "path": img_path,
                        "datetime": frame_dt,
                        "tile_offset_x": tile_offset_x,
                        "tile_offset_y": tile_offset_y,
                        "label": label,
                        "lat": lat,
                        "lon": lon,
                        "x_pix":x_pix,
                        "y_pix":y_pix,
                        "name": medicane_name
                    })
            
    res =  pd.DataFrame(updated_metadata)
    res = res.astype({
        "path": 'string',
        "datetime": 'datetime64[ns]',
        "tile_offset_x": 'int16',
        "tile_offset_y": 'int16',
        "label": 'category',
        "lat": 'float16',
        "lon": 'float16',
        "x_pix": 'Int16',
        "y_pix": 'Int16',
        "name": 'string'
    })
    return res




def create_tile_videos(df, output_dir=None, tile_size=224):
    """
    df contiene:
      tile_offset_x, tile_offset_y, datetime, path, ...
    Ritorna una lista (videos_list) di DataFrame,
    ognuno con 16 righe consecutive. 
    """

    # 1) Ordiniamo il DataFrame per (tile_offset_x, tile_offset_y, datetime)
    df_sorted = df.sort_values(["tile_offset_x", "tile_offset_y", "datetime"])

    # 2) Raggruppiamo per tile_offset
    grouped = df_sorted.groupby(["tile_offset_x", "tile_offset_y"], group_keys=False)
    
    results = []
    video_id = 0
    
    for (offset_x, offset_y), group_df in grouped:
        # group_df è un sotto-DataFrame con tutte le righe di quella tile
        # Ordinate già per datetime.
        group_df = group_df.reset_index(drop=True)
        row_count = len(group_df)
        num_blocks = row_count // 16  # quante volte possiamo formare un blocco di 16
        
        # se row_count non è multiplo di 16, rimarranno righe extra che ignoriamo (oppure gestisci diversamente)
        #         
        for i in range(num_blocks):
            start_i = i * 16
            end_i   = start_i + 16
            block_df = group_df.iloc[start_i:end_i]
            start_time = group_df.datetime.iloc[start_i]
            end_time = group_df.datetime.iloc[end_i]
            
            # se output dir è passato: salva le tile video in nuove cartelle
            date_str = end_time.strftime("%d-%m-%Y_%H%M")
            path_name = f"{date_str}_{offset_x}_{offset_y}"
            subfolder = path_name
            
            if output_dir is not None:  
                subfolder = Path(output_dir) / path_name                              
                subfolder.mkdir(parents=True, exist_ok=True)

                for k, row_ in enumerate(block_df.itertuples(index=False)):
                    orig_path = row_.path  # colonna col path dell'immagine
                    new_name = subfolder / f"img_{k+1:05d}.png"

                    #default_offsets_list = calc_tile_offsets() # valori di default   w, h, tile_size, stride)
                    save_single_tile(orig_path, new_name, offset_x, offset_y, tile_size)
                    
    
            if (block_df['label'] == '1').any():
                label = 1
            else:
                label = 0
            # A questo punto block_df ha 16 righe consecutive
            results.append({
                "video_id": video_id,
                "tile_offset_x": offset_x,
                "tile_offset_y": offset_y,
                "path": str(subfolder),
                "label": label,
                "start_time": start_time,
                "end_time": end_time,                
                "orig_paths": block_df["path"].tolist(),
            })
            video_id += 1

    return pd.DataFrame(results)


def create_and_save_tile_from_complete_df(df, output_dir, overwrite=False):
    for idx, row in df.iterrows():
        # crea la cartella di destinazione
        path_name = row.path
        subfolder = Path(output_dir) / path_name
        subfolder.mkdir(parents=True, exist_ok=True)

        offset_x, offset_y = row.tile_offset_x, row.tile_offset_y
        for k, orig_p in enumerate(row.orig_paths):
            new_name = subfolder / f"img_{k+1:05d}.png"
            #print(f"new_name {new_name}")
            if not os.path.isfile(new_name) or overwrite:
                #print(f"{new_name} lo sto risalvando?!")
                save_single_tile(orig_p, new_name, offset_x, offset_y, tile_size=224)
            else:
                #print(f"non c'è stato bisogno di risalvarlo")
                pass






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



def main():
 
    Xmin, Ymin, px_scale_x, px_scale_y = compute_pixel_scale(m, latcorners, loncorners,
        big_image_w=1290, big_image_h=420)

    ######################################  carica i files e le date
    input_dir = "../fromgcloud"  # Cambia questo percorso
    output_dir = "../airmassRGB/supervised"
    #os.makedirs(output_dir, exist_ok=True)
    
    filenames = get_files_from_folder(folder=input_dir, extension="png")
    #print(f"Trovati {len(filenames)} files")
    file_metadata = []  # contiene i nomi e le date
    for fname in filenames:
        start_dt = extract_dates_pattern_airmass_rgb_20200101_0000(fname.name)
        file_metadata.append((fname, start_dt))

    sorted_metadata = sorted(file_metadata, key=lambda x: x[1])  # Ordina per start_dt
    #random_fnames =  [item[0] for item in file_metadata]
    sorted_filenames = [item[0] for item in sorted_metadata]
    print(f" Ci sono {len(sorted_filenames)} files.")
    
    #sorted_filenames = sorted_filenames[:32]

    # 1) Abbiamo gia' i sorted_filenames caricati, e output_dir definito
    #sf = [(p, dt) for p, dt in zip(sorted_filenames, [f[1] for f in sorted_files])]
    #print(f"sorted filenames, len = {len(sf)}")
    subfolder_info = split_into_tiles_subfolders_and_track_dates(sorted_filenames=sf, output_dir=output_dir)
    print(f"Creati {len(subfolder_info)} tile-video in total.")

    # 2) Carico i ciclonici
    tracks_file = "./TRACKS_CL7.dat"  
    df_tracks = load_cyclones_track_noheader(tracks_file)


    # 4) label subfolders
    out_csv_label = os.path.join(output_dir, "label_tiles.csv")
    df = label_subfolders_with_cyclones_df(
        subfolder_info,
        df_tracks,
        basemap_obj=m,
        x_center=x_center,
        y_center=y_center,
        px_scale_x=px_scale_x,
        px_scale_y=px_scale_y,
        out_csv=out_csv_label
    )

if __name__ == "__main__":
    tracks_df = get_all_cyclones()
    sorted_metadata_files = load_all_images(input_dir = "../fromgcloud")
    start = time()
    df_data = labeled_tiles_from_metadatafiles(sorted_metadata_files, tracks_df)
    end = time()
    print(round((end-start)/60, 2))
    df_data.to_csv("all_data.csv")

    
    # Richiediamo gli offset dai tile (non ci serve la lista di sub-tile veri e propri)
    #default_offsets = calc_tile_offsets()



    
