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
from mpl_toolkits.basemap import Basemap

from medicane_utils.load_files import load_cyclones_track_noheader, get_files_from_folder, extract_dates_pattern_airmass_rgb_20200101_0000
from medicane_utils.geo_const import latcorners, loncorners, x_center, y_center, basemap_obj






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



def calc_tile_offsets(image_width=1290, image_height=420, tile_size=224, stride=112):
    """
    Ritorna una lista di (x_off, y_off) 
    """    
    offsets = []
    for y_off in range(0, image_height - tile_size + 1, stride):
        for x_off in range(0, image_width - tile_size + 1, stride):
            offsets.append((x_off, y_off))
    return offsets

def get_dataset_offsets(frames_list, tile_size=224, stride=112):
    """Prende tutti gli offsets di ogni immagine"""
    all_offsets = []
    for frame in frames_list:  # Itera su ogni frame (immagine PIL)
        w, h = frame.size
        offsets_this_frame = calc_tile_offsets(w, h, tile_size, stride)
        all_offsets.append(offsets_this_frame)

    return all_offsets

def create_tile(frame, tile_size=224, stride=112):
    w, h = frame.size
    offsets_this_frame = calc_tile_offsets(w, h, tile_size, stride)
    # costruisco le tiles
    tiles_this_frame = []
    for x,y in offsets_this_frame:  
        tile = frame.crop((x, y, x + tile_size, y + tile_size))  # Usa crop() di PIL
        tiles_this_frame.append(tile)

    return tiles_this_frame, offsets_this_frame
    
    
    
##########################################################
###################         #Logica per determinare se (lat, lon) cade dentro un tile 224×224
##########################################################

def compute_pixel_scale(
    basemap_obj,
    latcorners, loncorners,
    big_image_w=1290, big_image_h=420
):
    """
    Proietta i 4 corner in coordinate geostazionarie,
    trova Xmin,Xmax, Ymin,Ymax => calcola px_scale_x,y.
    Ritorna (x_min, y_min, px_scale_x, px_scale_y).
    """
    corner_coords = [
        (loncorners[0], latcorners[0]),
        (loncorners[1], latcorners[0]),
        (loncorners[0], latcorners[1]),
        (loncorners[1], latcorners[1])
    ]
    x_vals = []
    y_vals = []
    for (lo, la) in corner_coords:
        xg, yg = basemap_obj(lo, la)
        x_vals.append(xg)
        y_vals.append(yg)
    Xmin, Xmax = min(x_vals), max(x_vals)
    Ymin, Ymax = min(y_vals), max(y_vals)

    # quante "pixel su un metro" in orizzontale e verticale
    px_scale_x = big_image_w / (Xmax - Xmin)
    px_scale_y = big_image_h / (Ymax - Ymin)

    return Xmin, Ymin, px_scale_x, px_scale_y

def coord2px(lat, lon, px_per_km_x, px_per_km_y, Xmin, Ymin):
    # 1) Ottieni la proiezione "Xgeo, Ygeo" in metri (circa) 
    x_geo, y_geo = basemap_obj(lon, lat)
    # 2) Sottrai offset
    Xlocal = x_geo - Xmin
    Ylocal = y_geo - Ymin
    # 3) Converti Xlocal, Ylocal in pixel
    x_pix = Xlocal * px_per_km_x
    y_pix = Ylocal * px_per_km_y

    return x_pix, y_pix
    

def inside_tile(lat, lon, tile_x, tile_y,
                tile_width, tile_height,
                px_per_km_x, px_per_km_y):
    """
    Verifica se la lat/lon (gradi) cade dentro i confini di un tile 224×224 
    definito in coordinate "pixel".
    
    1) Converti (lat,lon) in coordinate geostazionarie (m. stuff).
    2) Sottrai offset del sub-satellite point (x_center, y_center).
    3) Confronta con l'intervallo [tile_x, tile_x+tile_width] in coordinate pixel.
       -> serve una scala (px_per_km_x, px_per_km_y) o simile per passare da “metri geostazionari” a pixel.
    """
    
    x_pix, y_pix = coord2px(lat, lon, px_per_km_x, px_per_km_y)

    # Check se (x_pix, y_pix) cade nel tile 
    if (tile_x <= x_pix < tile_x + tile_width) and \
       (tile_y <= y_pix < tile_y + tile_height):
        return True
    else:
        return False




    

# crea i video e salva in cartelle partX e traccia la data dei video
def labeled_tiles_from_metadatafiles(sorted_metadata_files):   #, save_to_file=False):

    updated_metadata = []
    default_offsets_for_frame = calc_tile_offsets()
    for idx, (img_path, frame_dt) in enumerate(sorted_metadata_files):
        
        for tile_x, tile_y in default_offsets_for_frame:


            updated_metadata.append({                    
                        "datetime": frame_dt,  
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                    })






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
    


def main():
    ###################################### BASEMAP 
    latcorners = [30, 48]
    loncorners = [-7, 46]
    m = Basemap(
        projection='geos',
        rsphere=(6378137.0, 6356752.3142),
        resolution='i',
        area_thresh=10000.,
        lon_0=9.5,
        satellite_height=3.5785831E7
    )
    x_center, y_center = m(9.5, 0)
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
    main()




    
