import numpy as np
import os
# import datetime
# import time
# import random
# import json
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torch.backends.cudnn as cudnn

#from utils import multiple_pretrain_samples_collate
#from functools import partial

#import utils
#from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_finetuning_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
#from model_analysis import get_dataloader, get_dataset_dataloader

from build_dataset import calc_tile_offsets

# from dataset import build_dataset

# from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test
# from optim_factory import create_optimizer
import models # NECESSARIO ALTRIMENTI NON CARICA IL MODELLO (? è come Args? )
from timm.models import create_model
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from utils import NativeScalerWithGradNormCount as NativeScaler

#################################################################################
###########################  FUNZIONI DI VISUALIZZAZIONE
#################################################################################

def plot_image(img, basemap_obj, latcorners, loncorners, dpi=96, width=1290, height=420, draw_parallels_meridians=False):
    
    lat_min, lat_max = latcorners
    lon_min, lon_max = loncorners

    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    basemap_obj.imshow(img, origin='upper')
    basemap_obj.drawcoastlines()
        
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
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Chiude la figura per liberare memoria
    buf.seek(0)

    # Converte il buffer in un'immagine PIL
    img_pil = Image.open(buf)
    return img_pil


def draw_tiles_and_center(
    pil_image: Image.Image,
    tile_size=224,
    cyclone_centers=None,
    labeled_tiles_offsets=None,
    point_color=(255, 0, 0),
    point_radius=4
):
    """
    Disegna, sull'immagine `pil_image`, una serie di riquadri (224×224 di default)
    generati con stride specificato, in modo identico alla suddivisione in tile.

    `cyclone_center` è una lista di tuple (cx, cy), disegna un punto rosso in ogni posizione.

    Ritorna l'immagine PIL con i disegni sopra.
    """

    # Richiediamo gli offset dai tile (non ci serve la lista di sub-tile veri e propri)
    default_offsets = calc_tile_offsets()

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
            if labeled_tiles_offsets[i] == 1:
                color = present_color
                width = 4            
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=width)

    # Se c'è un centro da disegnare
    if cyclone_centers is not None:
        for cx, cy in cyclone_centers:
        # Disegniamo un piccolo cerchio intorno al centro
            draw.ellipse(
                [
                    (cx - point_radius, cy - point_radius),
                    (cx + point_radius, cy + point_radius)
                ],
                fill=point_color
        )

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