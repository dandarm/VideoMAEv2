import numpy as np
import os
import re
# import datetime
# import time
# import random
# import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torch.backends.cudnn as cudnn

#from utils import multiple_pretrain_samples_collate
#from functools import partial

#import utils
#from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_finetuning_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
#from model_analysis import get_dataloader, get_dataset_dataloader

# from dataset import build_dataset

# from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test
# from optim_factory import create_optimizer
import models # NECESSARIO ALTRIMENTI NON CARICA IL MODELLO (? è come Args? )
from timm.models import create_model
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from utils import NativeScalerWithGradNormCount as NativeScaler

from medicane_utils.geo_const import latcorners, loncorners, create_basemap_obj
from build_dataset import calc_tile_offsets
from medicane_utils.load_files import extract_dates_pattern_airmass_rgb_20200101_0000


PALETTE = {
    'CL2': (255, 0, 0),      # Rosso
    'CL3': (0, 128, 0),      # Verde scuro
    'CL4': (0, 0, 255),      # Blu
    'CL5': (255, 165, 0),    # Arancione
    'CL6': (128, 0, 128),    # Viola
    'CL7': (0, 255, 255),    # Ciano
    'CL8': (255, 0, 255),    # Magenta
    'CL9': (128, 128, 0),    # Oliva
    'CL10': (0, 0, 0),        # Nero
}


#################################################################################
###########################  FUNZIONI DI VISUALIZZAZIONE
#################################################################################

def plot_image(img, basemap_obj, dpi=96, width=1290, height=420, draw_parallels_meridians=False, fig=None):
    
    lat_min, lat_max = latcorners
    lon_min, lon_max = loncorners

    if fig is None:
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_position([0, 0, 1, 1])

    basemap_obj.imshow(img, origin='upper')
    basemap_obj.drawcoastlines()

    ax.set_xlim(basemap_obj.xmin, basemap_obj.xmax)
    ax.set_ylim(basemap_obj.ymin, basemap_obj.ymax)
        
    # set parallels and meridians
    if draw_parallels_meridians:
        dparal=2.0 #separation in deg between drawn parallels
        parallels = np.arange(lat_min,lat_max,dparal)
        dmerid=2.0 #separation in deg between drawn meridians
        meridians = np.arange(lon_min,lon_max,dmerid)
        basemap_obj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)  #,weight='bold')
        basemap_obj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10, rotation=45)  #,weight='bold')
    
    #plt.show()  #bbox_inches='tight', pad_inches=0)
    # Salva la figura in un buffer in memoria
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
    if fig is None:
        plt.close(fig)  # Chiude la figura per liberare memoria
    buf.seek(0)

    # Converte il buffer in un'immagine PIL
    img_pil = Image.open(buf)
    return img_pil



def draw_timestamp_in_bottom_right(
    pil_img,
    text_str,
    margin=10,
    font_size=30,
    font_path="digital-7 (italic).ttf",
    text_color=(255, 80, 80)
):
    """
    Disegna `text_str` in basso a destra dell'immagine `pil_img`.
    Usa textbbox(...) per calcolare larghezza e altezza del testo.
    Necessita Pillow >= 8.0
    """
    draw = ImageDraw.Draw(pil_img)

    #if font is None:
    #    font = ImageFont.load_default()  # default Pillow font
    font = ImageFont.truetype(font_path, font_size)

    # textbbox restituisce (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text_str, font=font)
    text_width = bbox[2] - bbox[0]
    text_height= bbox[3] - bbox[1]

    # Calcoliamo la posizione in basso a destra
    img_w, img_h = pil_img.size
    x = img_w - text_width - margin
    y = img_h - text_height - margin

    # Disegniamo
    draw.text((x, y), text_str, font=font, fill=text_color)
    return pil_img




def extract_cl_number(cl):
    """ Estrae il numero da 'CLn' come intero. """
    match = re.match(r'CL(\d+)', cl)
    return int(match.group(1)) if match else -1

def deduplicate_xy_source(x_list, y_list, source_list):
    if not x_list or not y_list or not source_list:
        return []  # o None, o np.nan

    grouped = {}
    for x, y, src in zip(x_list, y_list, source_list):
        key = (x, y)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(src)

    # per ogni gruppo scegli il CL con n più grande
    result = []
    for (x, y), srcs in grouped.items():
        max_src = max(srcs, key=extract_cl_number)
        result.append((x, y, max_src))

    return result


def draw_tiles_and_center(
    pil_image: Image.Image,
    default_offsets,
    tile_size=224,
    cyclone_centers=None,
    labeled_tiles_offsets=None,
    point_color=(255, 255, 255),
    point_radius=4
):
    """
    Disegna, sull'immagine `pil_image`, una serie di riquadri (224×224 di default)
    generati con stride specificato, in modo identico alla suddivisione in tile.

    `cyclone_center` è una lista di tuple (cx, cy, source), disegna un punto rosso in ogni posizione.

    Ritorna l'immagine PIL con i disegni sopra.
    """

    # Creiamo una copia su cui disegnare
    out_img = pil_image.copy()
    draw = ImageDraw.Draw(out_img)

    # Disegniamo i rettangoli
    present_color = (0, 255, 0)
    absent_color = (216,216,216)  # grigio 
    for i, (x_off, y_off) in enumerate(default_offsets):
        x1, y1 = x_off, y_off
        x2, y2 = x_off + tile_size, y_off + tile_size
        color = absent_color
        width = 1
        if labeled_tiles_offsets is not None:
            if labeled_tiles_offsets[i] == '1':
                color = present_color
                width = 4            
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=width)

    #print(cyclone_centers)
    for center in cyclone_centers:
        for cx_cy_source in center:
            #print(f"cx_cy_source {cx_cy_source}")
            cx, cy, source = cx_cy_source
            
            # Disegniamo un piccolo cerchio intorno al centro
            color = PALETTE.get(source, point_color)  # fallback a bianco
            draw.ellipse(
                [
                    (cx - point_radius, cy - point_radius),
                    (cx + point_radius, cy + point_radius)
                ],
                fill=color
            )
            #draw.text((cx, cy),source,(255,255,255))#,font=font)

    return out_img




def create_gif_pil(image_paths, output_gif, duration=100, loop=0):
    """
    Crea una GIF animata partendo da una lista di immagini (stessa dimensione).
    - image_paths: lista dei path delle immagini, in ordine temporale.
    - output_gif: nome file di output, es: "anim.gif".
    - duration: tempo (ms) tra un fotogramma e l'altro.
    - loop=0 => la gif si ripete all'infinito. Imposta un intero >0 per numero di loop.
    """
    # Carichiamo tutte le immagini PIL in memoria
    frames = [Image.open(p).convert('RGB') for p in image_paths]

    # Salviamo la prima, e poi "appendiamo" le altre con save_all=True
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # millisecondi tra un frame e il successivo
        loop=loop
    )
    print(f"GIF creata in: {output_gif}")



##### l'ho usata per l'animazione della singola tile con play in jupyter

def display_video_clip(frames_tensors, interval=200):
    """
    frames_tensors: array di shape (T, H, W, 3) in formato RGB normalizzato 0-1
    interval: intervallo in millisecondi tra i frame dell'animazione
    """
    fig = plt.figure()
    ims = []

    for i in range(len(frames_tensors)):
        # Mostra un singolo frame
        # Se frames_tensors[i] è un tensore Torch, converti in numpy
        frame_np = frames_tensors[i].detach().cpu().numpy() if torch.is_tensor(frames_tensors[i]) else frames_tensors[i]
        # Assumi che sia [H, W, 3] con valori in [0,1]
        im = plt.imshow(frame_np, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)
    plt.close(fig)  # Chiudiamo la figura per evitare doppia visualizzazione
    return HTML(ani.to_jshtml())






####### l'ho usata per realizzare l'animazione di tutto il mediterraneo

def create_labeled_images_with_tiles(df_grouped, nome_gif, basemap_obj):
# in ogni group abbiamo una sola immagine (un istante temporale)
# e tutte le tiles con le rispettive label. 
# possiamo avere più cicloni con le rispettive coordinate, da trovare uniche, perché si ripetono in tutte le tiles vicine
    lista_immagini = []    

    for path_img, group_df in df_grouped:
        # Apriamo l'immagine
        img = Image.open(path_img)#.convert("RGB")
        #center_px_list = (x_pix, y_pix)
        center_px_list = group_df[['x_pix','y_pix']].value_counts().index.values
        #labeled_tiles_offsets = get_tile_labels(lat, lon)
        labeled_tiles_offsets = group_df['label'].values

        date_str = group_df['datetime'].unique()[0].strftime(" %H:%M %d-%m-%Y")
        
        # Disegniamo
        default_offsets = calc_tile_offsets()
        out_img = draw_tiles_and_center(img, default_offsets,
            cyclone_centers=center_px_list,
            labeled_tiles_offsets=labeled_tiles_offsets
            )
        stamped_img = draw_timestamp_in_bottom_right(out_img, date_str, margin=15)
        pi_img = plot_image(stamped_img, basemap_obj, draw_parallels_meridians=True)
        lista_immagini.append(pi_img)
        #display(out_img)
    
    lista_immagini[0].save(nome_gif, save_all=True, append_images=lista_immagini[1:], duration=200, loop=0)




def normalize_01(img_array):
    # prende un numpy array e lo normalizza
    _min = img_array.min()
    _max = img_array.max()
    _vis = (img_array - _min) / (_max - _min + 1e-5)  # normalizzato [0,1]
    return _vis


import ast 
# per trasformare liste di tuple di stringhe in liste di tuple di liste di int (!)
def safe_parse(s):
    try:
        print(s, end= '\t')
        lst = ast.literal_eval(s)      
        print(lst)
        return [int(x) for x in lst] if isinstance(lst, list) else []
    except (ValueError, SyntaxError):
        #print('fail')
        return []
    
def safe_literal_eval(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
    return val  # già una lista, None, o qualsiasi altro tipo
    
    
def create_mediterranean_video(list_grouped_df, interval=200, dpi=96, width=1290, height=420):

    #lista_immagini = []
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_position([0, 0, 1, 1])


    ax_map = fig.add_axes(ax.get_position(), frameon=False)
    ax_map.set_axis_off()

    basemap_obj = create_basemap_obj(ax=ax_map)
    basemap_obj.drawcoastlines(linewidth=1.0, color='black', zorder=2)

    lat_min, lat_max = latcorners
    lon_min, lon_max = loncorners
    dparal=2.0 #separation in deg between drawn parallels
    parallels = np.arange(lat_min,lat_max,dparal)
    dmerid=2.0 #separation in deg between drawn meridians
    meridians = np.arange(lon_min,lon_max,dmerid)
    basemap_obj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)  #,weight='bold')
    basemap_obj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10, rotation=45)  #,weight='bold')

    #ax_map.set_xlim(basemap_obj.xmin, basemap_obj.xmax)
    #axax_map.set_ylim(basemap_obj.ymin, basemap_obj.ymax)

    #  for path in path_list:
    first_path, first_df = list_grouped_df[0]
    img = Image.open(first_path)
    norm_array = normalize_01(np.array(img))
    im_obj = ax.imshow(norm_array, origin='upper', zorder=1)

    def update(frame_index):
        path, group_df = list_grouped_df[frame_index]

        # Carica la nuova immagine e aggiorna i dati dell'oggetto immagine
        #path = path_list[frame_index]
        img = Image.open(path) 

        center_px_df = group_df[['x_pix','y_pix', 'source']]
        # -> center_px_df è un dataframe, con tante righe quante sono le tiles, e tanti elementi per ogni tile
        # dove source è la CL di Manos
        center_px_df_parsed = center_px_df.map(safe_literal_eval)
        xy_source_list = center_px_df_parsed.apply(
            lambda row: deduplicate_xy_source(row['x_pix'], row['y_pix'], row['source']),
            axis=1)        
        # print(f"xy_source_list {xy_source_list}", flush=True)
        
        labeled_tiles_offsets = group_df['label'].values
        default_offsets = calc_tile_offsets()
        out_img = draw_tiles_and_center(img, default_offsets,
            cyclone_centers=xy_source_list,
            labeled_tiles_offsets=labeled_tiles_offsets
        )
        time_str = path.split('\\')[-1]        
        time = extract_dates_pattern_airmass_rgb_20200101_0000(time_str)
        stamped_img = draw_timestamp_in_bottom_right(out_img, time.strftime(" %H:%M %d-%m-%Y"), margin=15)

        im_array_new = np.array(stamped_img)
        norm_array = normalize_01(im_array_new)
        im_obj.set_data(norm_array)
        
        # Se le coste devono essere ridisegnate, puoi farlo qui, ma se restano fisse non è necessario
        return [im_obj]


        #im_artist = ax.imshow(_vis, animated=True)
        #lista_immagini.append([im_artist])

    #lista_immagini[0].save(nome_gif, save_all=True, append_images=lista_immagini[1:], duration=200, loop=0)

    #ani = animation.ArtistAnimation(fig, update, frames=len(path_list), interval=interval, blit=True, repeat_delay=1000)
    ani = animation.FuncAnimation(fig, update, frames=len(list_grouped_df), interval=interval, blit=True, repeat_delay=1000)
   

    #plt.show()
    plt.close(fig)  # Chiudiamo la figura per evitare doppia visualizzazione
    #return HTML(ani.to_jshtml())
    return ani

    







import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args = prepare_finetuning_args()
    # dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)

    # data_loader_val = DataLoader(
    # dataset_val,
    # batch_size=args.batch_size,
    # shuffle=True,         # Per estrarre sample casuali
    # num_workers=args.num_workers,
    # pin_memory=args.pin_mem,
    # drop_last=False
    # )

    print(f"Creating model: {args.model} (nb_classes={args.nb_classes})")

    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        **args.__dict__
    )