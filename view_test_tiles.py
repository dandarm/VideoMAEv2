import os
import multiprocessing
import shutil
from time import time
import re
import math
from pathlib import Path
import matplotlib
# import datetime
# import random
# import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

import ipywidgets as widgets
from typing import Optional, List, Tuple

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
import models # NECESSARIO ALTRIMENTI NON CARICA IL MODELLO 
from timm.models import create_model
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from utils import NativeScalerWithGradNormCount as NativeScaler

from medicane_utils.geo_const import latcorners, loncorners, create_basemap_obj
from dataset.build_dataset import calc_tile_offsets
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

filling_missing_tile = 'filled_gray'

# palette fallback for tracking annotations
PALETTE.setdefault('GT', (0, 255, 0))
PALETTE.setdefault('PRED', (255, 0, 0))


#################################################################################
###########################  FUNZIONI DI VISUALIZZAZIONE
#################################################################################
# region funzioni di arricchimento dell'immagine airmassRGB

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
    if cl == 'PRED':
        return 100  # assicurati che la traccia predetta prevalga in caso di duplicati
    if cl == 'GT':
        return 50
    match = re.match(r'CL(\d+)', cl)
    return int(match.group(1)) if match else -1

def deduplicate_xy_source(x_list, y_list, source_list):
    """
    Gli input possono provenire dal CSV come liste, stringhe o NaN.
    Questa funzione li normalizza in liste e rimuove i duplicati,
    privilegiando la sorgente con CL più alta (o la traccia PRED).
    """
    def _ensure_list(val):
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, (np.ndarray, pd.Series)):
            return list(val)
        if isinstance(val, float) and math.isnan(val):
            return []
        if isinstance(val, str):
            stripped = val.strip()
            if not stripped:
                return []
            if stripped.startswith('[') and stripped.endswith(']'):
                try:
                    parsed = safe_literal_eval(stripped)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                except Exception:
                    return [stripped]
            return [stripped]
        return [val]

    x_vals = _ensure_list(x_list)
    y_vals = _ensure_list(y_list)
    src_vals = _ensure_list(source_list)

    if not x_vals or not y_vals or not src_vals:
        return []

    result = []
    try:
        grouped = {}
        for x, y, src in zip(x_vals, y_vals, src_vals):
            key = (x, y)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(src)

        # per ogni gruppo scegli il CL con n più grande
        for (x, y), srcs in grouped.items():
            max_src = max(srcs, key=extract_cl_number)
            result.append((x, y, max_src))
    except:
        #print()
        pass
        

    return result


def draw_tiles_and_center(
    pil_image: Image.Image,
    offsets,
    tile_size=224,
    cyclone_centers=[],
    labeled_tiles_offsets=None,
    predicted_tiles=None,
    gray_offsets=None,
    neighboring_tile=None,
    point_color=(255, 255, 255),
    point_radius=4,
    show_tile_boxes=True,
):
    """
    Disegna, sull'immagine `pil_image`, una serie di riquadri (224×224 di default)
    generati con stride specificato, in modo identico alla suddivisione in tile.

    `cyclone_center` è una lista di tuple (cx, cy, source), disegna un punto rosso in ogni posizione.

     -->>> AGGIUNGO I RIQUADRI ROSSI : LE PREDIZIONI DEL MODELLO
    
    Ritorna l'immagine PIL con i disegni sopra.
    """
    out_img = pil_image.copy()
    base = out_img.convert("RGBA") 

    # Creiamo una copia su cui disegnare    
    draw = ImageDraw.Draw(base)

    # Crea un overlay trasparente della stessa misura.
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    # Disegniamo i rettangoli
    present_color = (0, 255, 0) # verde
    absent_color = (216,216,216)  # grigio 
    predicted_color = (255, 0, 0) # rosso
    neighboring_color = (255, 165, 0) # arancione
    fill_color = (216,216,216, 120) # grigio semi trasparente
    for i, (x_off, y_off) in enumerate(offsets):
        x1, y1 = x_off, y_off
        x2, y2 = x_off + tile_size, y_off + tile_size
        color = absent_color
        width = 1
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=width)

        # riquadro arancione per tile neighboring
        if neighboring_tile is not None:
            try:
                if neighboring_tile[i]:
                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline=neighboring_color,
                        width=2)
            except Exception:
                pass

        #print(f"labeled_tiles_offsets {labeled_tiles_offsets}")
        if show_tile_boxes:
            if labeled_tiles_offsets is not None:
                if not pd.isna(labeled_tiles_offsets[i]) and labeled_tiles_offsets[i] == 1:
                    color = present_color
                    width = 4         
                    draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=color,
                    width=width)
            # riquadro rosso per la predizione
            #print(f"predicted_tiles {predicted_tiles}")
            if predicted_tiles is not None:
                if not pd.isna(predicted_tiles[i]) and predicted_tiles[i] == 1:
                    color = predicted_color
                    width = 4      
                    x1 += 10
                    y1 += 10
                    x2 -= 10
                    y2 -= 10
                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline=color,
                        width=width)
                
            #print(f"gray_offsets {gray_offsets}")
            if gray_offsets is not None:
                if gray_offsets[i]:
                    # disegna filling grigio
                    draw_ov.rectangle(
                    [(x1, y1), (x2, y2)],
                    fill=fill_color)

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


    combined = Image.alpha_composite(base, overlay)
    final_img = combined.convert("RGB")

    return final_img

# endregion



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

def _resolve_ffmpeg_executable():
    """Trova il binario ffmpeg nel PATH o nella cartella locale del progetto."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    local_ffmpeg = Path(__file__).resolve().parent / "ffmpeg-7.0.2-amd64-static" / "ffmpeg"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)

    return None


def display_video_clip(frames_tensors, interval=200, save_path=None):
    """Crea l'animazione e (opzionalmente) la salva su file.

    frames_tensors: array di shape (T, H, W, 3) in formato RGB normalizzato 0-1
    interval: intervallo in millisecondi tra i frame dell'animazione
    save_path: path di output. Se None, l'animazione non viene salvata
    """
    fig = plt.figure()
    ims = []

    for i in range(len(frames_tensors)):
        # Mostra un singolo frame
        # Se frames_tensors[i] è un tensore Torch, converti in numpy
        frame_np = frames_tensors[i].detach().cpu().numpy() if torch.is_tensor(frames_tensors[i]) else frames_tensors[i]
        # Assumi che sia [H, W, 3] con valori in [0,1]
        im = plt.imshow(frame_np, animated=True)
        plt.xticks([])
        plt.yticks([])
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 1000.0 / interval if interval else 5
        try:
            if save_path.suffix.lower() == ".gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                ffmpeg_exec = _resolve_ffmpeg_executable()
                if ffmpeg_exec is None:
                    raise RuntimeError(
                        "ffmpeg non è stato trovato nel PATH né in 'ffmpeg-7.0.2-amd64-static/'. "
                        "Aggiungi il binario a PATH oppure passare un percorso valido."
                    )
                matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_exec
                writer = animation.FFMpegWriter(fps=fps)
            ani.save(str(save_path), writer=writer)
        except Exception as exc:
            plt.close(fig)
            raise RuntimeError(f"Impossibile salvare l'animazione in '{save_path}': {exc}") from exc

    plt.close(fig)  # Chiudiamo la figura per evitare doppia visualizzazione
    return HTML(ani.to_jshtml())



# region select videoclips

class VideoGallerySelector:
    """Galleria interattiva per selezionare clip di tracking."""

    def __init__(
        self,
        clips,
        identifiers=None,
        columns: int = 3,
        interval: int = 200,
        toggle_icon: str = 'check',
        preview_width: Optional[int] = 160,
        preview_height: Optional[int] = None,
        grid_gap: str = '12px',
        default_selected: bool = True
    ) -> None:
        if identifiers is None:
            identifiers = list(range(len(clips)))
        if len(identifiers) != len(clips):
            raise ValueError('`identifiers` deve avere la stessa lunghezza di `clips`.')

        self.clips = clips
        self.identifiers = list(identifiers)
        self.columns = max(1, int(columns))
        self.interval = max(1, int(interval))
        self.toggle_icon = toggle_icon
        self.preview_width = preview_width if preview_width and preview_width > 0 else None
        self.preview_height = preview_height if preview_height and preview_height > 0 else None
        self.grid_gap = grid_gap
        self.default_selected = bool(default_selected)
        self._resample_filter = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

        self._toggles = []
        self._container = self._build_widget()

    def _resize_frame(self, frame_img, target_size):
        if self.preview_width is None and self.preview_height is None:
            return frame_img, target_size

        if target_size is None:
            width, height = frame_img.size
            if self.preview_width and self.preview_height:
                target_size = (self.preview_width, self.preview_height)
            elif self.preview_width:
                scale = self.preview_width / max(1, width)
                target_size = (
                    self.preview_width,
                    max(1, int(round(height * scale)))
                )
            else:
                scale = self.preview_height / max(1, height)
                target_size = (
                    max(1, int(round(width * scale))),
                    self.preview_height
                )

        return frame_img.resize(target_size, self._resample_filter), target_size

    def _clip_to_gif_bytes(self, clip):
        if not clip:
            raise ValueError('Clip vuota: impossibile creare la preview animata.')

        frames = []
        target_size = None
        for frame in clip:
            arr = np.asarray(frame)
            if arr.ndim == 2:  # grayscale -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
                raise ValueError('Ogni frame deve avere shape [H, W, 3/4].')
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype('uint8')

            img = Image.fromarray(arr)
            img, target_size = self._resize_frame(img, target_size)
            frames.append(img)

        if len(frames) == 1:
            buf = BytesIO()
            frames[0].save(buf, format='PNG')
            return buf.getvalue(), 'png', frames[0].size

        buf = BytesIO()
        frames[0].save(
            buf,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=self.interval,
            loop=0,
            disposal=2
        )
        return buf.getvalue(), 'gif', frames[0].size

    def _build_media_widget(self, clip):
        try:
            data, fmt, size = self._clip_to_gif_bytes(clip)
        except Exception as exc:
            error_html = widgets.HTML(f'<pre style="color:#b00">Errore nel rendering clip: {exc}</pre>')
            return error_html

        layout = widgets.Layout()
        if size:
            layout.width = f'{size[0]}px'
        return widgets.Image(value=data, format=fmt, layout=layout)

    def _build_widget(self):
        cards = []
        grid_columns = min(self.columns, len(self.clips)) or 1
        min_width = (self.preview_width or 160) + 32
        column_template = f'repeat({grid_columns}, minmax({min_width}px, 1fr))'

        for clip, ident in zip(self.clips, self.identifiers):
            media_widget = self._build_media_widget(clip)

            toggle = widgets.ToggleButton(
                value=self.default_selected,
                description=str(ident),
                icon=self.toggle_icon,
                layout=widgets.Layout(width='100%'),
                tooltip='Attivo = clip inclusa; disattiva per escluderla'
            )
            toggle.observe(self._on_toggle_change, names='value')
            self._sync_toggle_style(toggle)
            self._toggles.append(toggle)

            cards.append(
                widgets.VBox(
                    [media_widget, toggle],
                    layout=widgets.Layout(
                        border='1px solid #ccc',
                        padding='6px',
                        align_items='center',
                        width='100%'
                    )
                )
            )

        grid = widgets.GridBox(
            cards,
            layout=widgets.Layout(
                grid_template_columns=column_template,
                grid_gap=self.grid_gap,
                width='100%'
            )
        )

        self._selection_label = widgets.HTML()
        self._update_selection_label()

        return widgets.VBox([
            grid,
            self._selection_label
        ], layout=widgets.Layout(width='100%'))

    def _sync_toggle_style(self, toggle):
        toggle.button_style = '' if toggle.value else 'danger'

    def _on_toggle_change(self, change):
        if change.get('name') == 'value':
            self._sync_toggle_style(change['owner'])
            self._update_selection_label()

    def _update_selection_label(self):
        kept = self.get_selected_identifiers()
        kept_count = len(kept)
        excluded_count = len(self.identifiers) - kept_count
        if excluded_count:
            self._selection_label.value = (
                f"<b>Da tenere:</b> {kept_count} clip | "
                f"<span style='color:#b00'><b>Escluse:</b> {excluded_count}</span>"
            )
        else:
            self._selection_label.value = (
                f"<b>Da tenere:</b> {kept_count} clip | <i>Nessuna esclusa</i>"
            )

    @property
    def widget(self):
        return self._container

    def display(self):
        display(self._container)

    def get_selected_indices(self):
        """Indici delle clip ancora incluse."""
        return [idx for idx, toggle in enumerate(self._toggles) if toggle.value]

    def get_selected_identifiers(self):
        indices = self.get_selected_indices()
        return [self.identifiers[idx] for idx in indices]

    def get_excluded_indices(self):
        """Indici delle clip escluse esplicitamente."""
        return [idx for idx, toggle in enumerate(self._toggles) if not toggle.value]

    def get_excluded_identifiers(self):
        indices = self.get_excluded_indices()
        return [self.identifiers[idx] for idx in indices]

# endregion






# Disegna tutto i lgframe completo con tile, orario, tracks... di tutto il mediterraneo


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

    # Carica il primo frame per creare l'oggetto immagine
    first_path, first_df = list_grouped_df[0]
    img = Image.open(first_path)
    norm_array = normalize_01(np.array(img))
    im_obj = ax.imshow(norm_array, origin='upper', zorder=1)

    def update(frame_index):
        path, group_df = list_grouped_df[frame_index]

        # Carica la nuova immagine e aggiorna i dati dell'oggetto immagine
        img = Image.open(path)

        center_px_df = group_df[['x_pix', 'y_pix', 'source']]
        center_px_df_parsed = center_px_df.map(safe_literal_eval)

        disable_tile_boxes = False
        if 'disable_tile_boxes' in group_df.columns:
            try:
                disable_tile_boxes = bool(group_df['disable_tile_boxes'].iloc[0])
            except Exception:
                disable_tile_boxes = False

        is_tracking_view = 'track_pred_x' in group_df.columns or disable_tile_boxes

        if is_tracking_view:
            def _ensure_list_local(val):
                if val is None:
                    return []
                if isinstance(val, (list, tuple)):
                    return list(val)
                if isinstance(val, (np.ndarray, pd.Series)):
                    return list(val)
                if isinstance(val, float) and math.isnan(val):
                    return []
                if isinstance(val, str):
                    stripped = val.strip()
                    if not stripped:
                        return []
                    if stripped.startswith('[') and stripped.endswith(']'):
                        try:
                            parsed = safe_literal_eval(stripped)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                        except Exception:
                            return [stripped]
                    return [stripped]
                return [val]

            center_px_df_parsed = center_px_df_parsed.copy()
            center_px_df_parsed['source'] = center_px_df_parsed['source'].apply(
                lambda src_list: [
                    'PRED' if isinstance(s, str) and s == 'PRED' else 'GT'
                    for s in _ensure_list_local(src_list)
                    if not pd.isna(s)
                ]
            )

        xy_source_list = center_px_df_parsed.apply(
            lambda row: deduplicate_xy_source(row['x_pix'], row['y_pix'], row['source']),
            axis=1
        )

        if is_tracking_view:
            unique_points = {}
            for centers in xy_source_list:
                for cx_cy_src in centers:
                    if len(cx_cy_src) != 3:
                        continue
                    src = cx_cy_src[2]
                    if src not in unique_points:
                        unique_points[src] = cx_cy_src
            condensed = [[] for _ in range(len(xy_source_list))]
            selected = [unique_points[key] for key in ['GT', 'PRED'] if key in unique_points]
            if condensed:
                condensed[0] = selected
            xy_source_list = condensed

        labeled_tiles_offsets = group_df['label'].values

        if 'predictions' in group_df.columns and not is_tracking_view:
            predicted_tiles_offsets = group_df['predictions'].values
        else:
            predicted_tiles_offsets = None

        if filling_missing_tile in group_df.columns:
            to_be_filled_offsets = group_df[filling_missing_tile].values
        else:
            to_be_filled_offsets = None

        neighboring_tiles = None
        if not is_tracking_view and 'neighboring' in group_df.columns:
            neighboring_tiles = group_df['neighboring'].values
        
        offsets = list(group_df[['tile_offset_x', 'tile_offset_y']].value_counts().index.values)

        out_img = draw_tiles_and_center(
            img,
            offsets,
            cyclone_centers=xy_source_list,
            labeled_tiles_offsets=labeled_tiles_offsets,
            predicted_tiles=predicted_tiles_offsets,
            gray_offsets=to_be_filled_offsets,
            neighboring_tile=neighboring_tiles,
            show_tile_boxes=not is_tracking_view
        )

        time_str = Path(path).name
        tempo = extract_dates_pattern_airmass_rgb_20200101_0000(time_str)
        stamped_img = draw_timestamp_in_bottom_right(out_img, tempo.strftime(" %H:%M %d-%m-%Y"), margin=15)

        im_array_new = np.array(stamped_img)
        norm_array = normalize_01(im_array_new)
        im_obj.set_data(norm_array)
        
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


import subprocess
from multiprocessing import Pool

def compose_image(frame_idx, list_grouped_df, debug=False):
    dpi=96
    width=1290
    height=420

    # region Crea figura
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_position([0, 0, 1, 1])
    ax_map = fig.add_axes(ax.get_position(), frameon=False)
    ax_map.set_axis_off()
    # endregion

    # region Operazioni basemap
    if not debug:
        basemap_obj = create_basemap_obj(ax=ax_map)
        basemap_obj.drawcoastlines(linewidth=1.0, color='black', zorder=2)

        # Draw parallels and meridians
        lat_min, lat_max = latcorners
        lon_min, lon_max = loncorners
        dparal = 2.0
        parallels = np.arange(lat_min, lat_max, dparal)
        dmerid = 2.0
        meridians = np.arange(lon_min, lon_max, dmerid)
        basemap_obj.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)
        basemap_obj.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10, rotation=45)
    #endregion
    
    # Chiamate come nella vecchia update
    path, group_df = list_grouped_df[frame_idx]

    # Carica la nuova immagine e aggiorna i dati dell'oggetto immagine
    img = Image.open(path) 


    # region componenti di labeling delle tiles

    center_px_df = group_df[['x_pix','y_pix', 'source']]        
    # -> center_px_df è un dataframe, con tante righe quante sono le tiles, e tanti elementi per ogni tile
    # dove source è la CL di Manos
    center_px_df_parsed = center_px_df.map(safe_literal_eval)

    disable_tile_boxes = False
    if 'disable_tile_boxes' in group_df.columns:
        try:
            disable_tile_boxes = bool(group_df['disable_tile_boxes'].iloc[0])
        except Exception:
            disable_tile_boxes = False

    is_tracking_view = 'track_pred_x' in group_df.columns or disable_tile_boxes

    if is_tracking_view:
        def _ensure_list_local(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, (np.ndarray, pd.Series)):
                return list(val)
            if isinstance(val, float) and math.isnan(val):
                return []
            if isinstance(val, str):
                stripped = val.strip()
                if not stripped:
                    return []
                if stripped.startswith('[') and stripped.endswith(']'):
                    try:
                        parsed = safe_literal_eval(stripped)
                        if isinstance(parsed, (list, tuple)):
                            return list(parsed)
                    except Exception:
                        return [stripped]
                return [stripped]
            return [val]

        center_px_df_parsed = center_px_df_parsed.copy()
        center_px_df_parsed['source'] = center_px_df_parsed['source'].apply(
            lambda src_list: [
                'PRED' if isinstance(s, str) and s == 'PRED' else 'GT'
                for s in _ensure_list_local(src_list)
                if not pd.isna(s)
            ]
        )

    xy_source_list = center_px_df_parsed.apply(
        lambda row: deduplicate_xy_source(row['x_pix'], row['y_pix'], row['source']),
        axis=1)        

    if is_tracking_view:
        unique_points = {}
        for centers in xy_source_list:
            for cx_cy_src in centers:
                if len(cx_cy_src) != 3:
                    continue
                src = cx_cy_src[2]
                if src not in unique_points:
                    unique_points[src] = cx_cy_src
        condensed = [[] for _ in range(len(xy_source_list))]
        selected = [unique_points[key] for key in ['GT', 'PRED'] if key in unique_points]
        if condensed:
            condensed[0] = selected
        xy_source_list = condensed

    labeled_tiles_offsets = group_df['label'].values # dovrebbe avere tanti valori quante sono le tiles
    # se ne ha di meno è perché stiamo guardando un sottoinsieme, es. il dataset di test
    # quindi quelle che mancano dovremmo riempire con un velo grigio
    if debug:
        print(f"labeled_tiles_offsets: {labeled_tiles_offsets}")

    if 'predictions' in group_df.columns and not is_tracking_view:
        predicted_tiles_offsets = group_df['predictions'].values
    else:
        predicted_tiles_offsets = None

    if filling_missing_tile in group_df.columns:
        to_be_filled_offsets = group_df[filling_missing_tile].values
    else:
        to_be_filled_offsets = None
    # endregion

    offsets = [tuple(riga) for riga in group_df[['tile_offset_x','tile_offset_y']].values]
    
    neighboring_tiles = None
    if not is_tracking_view and 'neighboring' in group_df.columns:
        neighboring_tiles = group_df['neighboring'].values

    out_img = draw_tiles_and_center(img, offsets,
        cyclone_centers=xy_source_list,
        labeled_tiles_offsets=labeled_tiles_offsets,
        predicted_tiles=predicted_tiles_offsets,
        gray_offsets=to_be_filled_offsets,
        neighboring_tile=neighboring_tiles,
        show_tile_boxes=not is_tracking_view
    )

    # region add timestamp 
    if not debug:
        time_str = Path(path).name    
        tempo = extract_dates_pattern_airmass_rgb_20200101_0000(time_str)
        stamped_img = draw_timestamp_in_bottom_right(out_img, tempo.strftime(" %H:%M %d-%m-%Y"), margin=15)
    else:
        stamped_img = out_img
    # endregion

    # normalizza
    im_array_new = np.array(stamped_img)
    norm_array = normalize_01(im_array_new)

    ax.imshow(norm_array, origin='upper', zorder=1,
              extent=[0, norm_array.shape[1], norm_array.shape[0], 0])
    ax.set_xlim(0, norm_array.shape[1])
    ax.set_ylim(norm_array.shape[0], 0)
    if debug:
        print(f"limiti settati a {norm_array.shape}")
    return fig

def render_and_save_frame(args, overwrite=False):
    frame_idx, list_grouped_df, output_folder = args

    # rapido check per non risalvare figure già esistenti:
    src_path, _ = list_grouped_df[frame_idx]
    #output_file = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
    base = os.path.splitext(os.path.basename(src_path))[0] + '.png'
    output_file = os.path.join(output_folder, base)
    
    if os.path.isfile(output_file) and not overwrite:
        return frame_idx, output_file

    fig = compose_image(frame_idx, list_grouped_df)

    # Salvataggio del frame
    dpi = 96
    plt.savefig(output_file, dpi=dpi, transparent=True)
    plt.close(fig)

    #print(f"Salvato frame {frame_idx} → {output_file}")
    return frame_idx, output_file

def save_frames_parallel(df, output_folder, print_every=50):
    grouped = df.groupby("path", dropna=False)
    list_grouped_df = list(grouped)
    total = len(list_grouped_df)
    print(f" abbiamo {total} gruppi", flush=True)

    args = [(i, list_grouped_df, output_folder) for i in range(total)]
    
    num_processes = multiprocessing.cpu_count()
    start = time()
    results = [None] * total
    with Pool(processes=num_processes) as pool:
        done = 0
        for frame_idx, output_path in pool.imap_unordered(render_and_save_frame, args, chunksize=1):
            results[frame_idx] = output_path
            done += 1
            if done % print_every == 0 or done == total:
                elapsed = max(time() - start, 1e-6)
                rate = done / elapsed
                remaining = total - done
                eta_sec = remaining / rate if rate > 0 else float("inf")
                eta_min = eta_sec / 60.0
                print(f"{done}/{total} ETA {eta_min:.1f}m", end="\t", flush=True)
    end = time()

    # gestisco i file in modo numerico sequenziale per ffmpeg, 
    # poiché il nome dei file non contiene l'ordine sequenziale
    ordered_paths = results
    frames_txt = os.path.join(output_folder, 'frames.txt')
    dur = 1.0 / framerate
    with open(frames_txt, 'w') as f:
        for p in ordered_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
            f.write(f"duration {dur}\n")
        f.write(f"file '{os.path.abspath(ordered_paths[-1])}'\n")

    print("Tutti i frame sono stati generati")
    print(f"{round((end-start)/60.0, 2)} minuti")
    
framerate=10
ffmpeg_command = lambda folder,nomefile : [
            'ffmpeg','-y',   # -y sovrascrive l’output senza chiedere
            '-f', 'concat', '-safe', '0',    # concat demuxer + percorsi assoluti
            '-i', os.path.join(folder, 'frames.txt'),
            '-framerate', str(framerate),
            '-vsync', 'vfr', 
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            nomefile
        ]
def make_animation_parallel_ffmpeg(
    df,
    id_cyc=None,
    output_folder="./anim_frames",
    nomefile=None,
    only_video=False,
):
    # Safety check: ensure grayed tiles column is present when rendering full Mediterranean frames
    assert filling_missing_tile in df.columns, (
        f"Manca la colonna '{filling_missing_tile}'. "
        "Assicurati di passare il DataFrame espanso (es. expanded_df) "
        "ottenuto con expand_group per riempire di grigio le tile mancanti."
    )
    if id_cyc is not None:
        nomefile=f"ciclone{id_cyc}.mp4"
        folder = Path(output_folder + '_' + str(id_cyc))
    else:        
        folder = Path(output_folder + "_" + nomefile)
        nomefile = nomefile + ".mp4"
    
    #if not folder.exists():
    os.makedirs(folder, exist_ok=True)
    if not only_video:
        print(f"\n>>> Generazione dei frame PNG in {folder}")
        save_frames_parallel(df, folder)
    else:
        if not folder.exists():
            raise RuntimeError(
                f"Cartella frame non trovata: {folder}. "
                "Rimuovi --only_video o genera prima i frame."
            )
        png_files = list(folder.glob("*.png"))
        if not png_files:
            raise RuntimeError(
                f"Nessun frame PNG trovato in {folder}. "
                "Rimuovi --only_video o genera prima i frame."
            )
        print(f"\n>>> Uso i frame PNG esistenti in {folder}")

    print("\n>>> Creazione del video MP4 con ffmpeg...")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg non trovato nel PATH. "
            "Passa --ffmpeg_path /percorso/ffmpeg oppure installa ffmpeg."
        )
    subprocess.run(ffmpeg_command(folder,nomefile))
    print(f"\nVideo salvato: {nomefile}\n")

    #else:
    print(f"Cartella {folder} già esistente, non ricreo i frame. Controlla se il video è già stato creato.")
    if not (Path(nomefile)).exists():
        print(f"Il video {nomefile} non esiste.")
        print("\n>>> Creazione del video MP4 con ffmpeg...")       
        subprocess.run(ffmpeg_command(folder,nomefile))
        print(f"\nVideo salvato: {nomefile}\n")
    else:
        print(f"Video già esistente: {nomefile}")




# region utils
def sub_select_frequency(df, freq='20min'):
    # selezionare ore intere
    df['dt_floor'] = df['datetime'].dt.floor(freq)
    mask = df['datetime'] == df['dt_floor']
    df_filtered = df[mask]
    #grouped = df_filtered.groupby("path", dropna=False)
    print(f"Tengo solo i frame ogni {freq}: rimangono {len(df_filtered)} elementi")
    return df_filtered

def expand_group(group, df_offsets):
    #df_offsets = group[['tile_offset_x','tile_offset_y']].value_counts()#.index.values)
    merged = df_offsets.merge(group, on=['tile_offset_x', 'tile_offset_y'], how='left', indicator=True)
    path_value = group['path'].iloc[0]
    merged['path'] = path_value

    extra_cols = [col for col in group.columns if col not in ['path', 'tile_offset_x', 'tile_offset_y']]
    # Ricopia i valori costanti del gruppo originale
    for col in ['datetime']:
        val = group[col].iloc[0]   #.mode()[0] if not group[col].isnull().all() else None
        merged[col] = merged[col].fillna(val)

    merged[filling_missing_tile] = merged['_merge'] == 'left_only'  # True se mancava
    return merged[['path', 'tile_offset_x', 'tile_offset_y'] + extra_cols + [filling_missing_tile]]

def filter_on_intervals(intervals, master_df):
    from intervaltree import IntervalTree
    # 1. Costruzione dell'albero degli intervalli
    tree = IntervalTree()
    for start, end in zip(intervals['start_time'], intervals['end_time']):
        # IntervalTree lavora con numeri: usiamo i timestamp in formato intero
        tree[start.value:end.value] = True

    # 2. Funzione per verificare se un timestamp è contenuto in almeno un intervallo
    def is_in_any_interval(ts):
        return bool(tree[ts.value])

    # 3. Applichiamo la funzione alla colonna datetime
    mask = master_df['datetime'].apply(is_in_any_interval)

    # 4. Selezione dei risultati
    filtered_df1 = master_df[mask].reset_index(drop=True)
    return filtered_df1
# endregion

# region non-parallel building animation
def get_writer4animation(ffmpeg_path_colon):
    from matplotlib.animation import FFMpegWriter

    os.environ['PATH'] = ffmpeg_path_colon + os.environ['PATH']

    writer = FFMpegWriter(
        fps=10,
        codec='libx264',
        bitrate=None,             # se usi crf, puoi lasciare bitrate a None
        extra_args=[
            '-crf', '18',         # qualità alta (visualmente quasi lossless)
            '-preset', 'slow',    # migliore compressione
            '-pix_fmt', 'yuv420p' # formato compatibile con la maggior parte dei player
        ])
    #writer = FFMpegWriter(fps=10, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'], bitrate=1800) #, executable=r'C:\ffmpeg\bin\ffmpeg.exe')
    return writer


def make_animation(df, nomefile='predictions_validation3.gif', writer='pillow'):
    # Safety check: ensure grayed tiles column is present when rendering full Mediterranean frames
    assert filling_missing_tile in df.columns, (
        f"Manca la colonna '{filling_missing_tile}'. "
        "Assicurati di passare il DataFrame espanso (es. expanded_df) "
        "ottenuto con expand_group per riempire di grigio le tile mancanti."
    )
    grouped = df.groupby("path", dropna=False)
    print(f" abbiamo {len(list(grouped))} gruppi", flush=True)
    start = time()
    video = create_mediterranean_video(list(grouped))
    video.save(nomefile, writer=writer)
    end = time()
    print(f"{round((end-start)/60.0, 2)} minuti")
    print(f"Video salvato: {nomefile}")

# endregion



# region video specifici
def video_specifico1(input_dir, output_dir):
    from medicane_utils.load_files import decodifica_id_intero
    from dataset.data_manager import BuildDataset
    from view_test_tiles import make_animation_parallel_ffmpeg

    import pandas as pd
    import io

    id_cyc = 1678
    tracks_df_MED7 = pd.read_csv("manos_CL7_pixel.csv", parse_dates=['time', 'start_time', 'end_time'])
    mask_id = tracks_df_MED7.id_cyc_unico.apply(decodifica_id_intero).str[1].astype(int) == id_cyc
    manos_sel = tracks_df_MED7[mask_id]

    data_builder = BuildDataset(type='SUPERVISED')
    data_builder.create_master_df_short(input_dir, manos_sel)

    make_animation_parallel_ffmpeg(data_builder.master_df, nomefile=f"ciclone{id_cyc}.mp4", output_folder = f"./anim_cyc{id_cyc}")



# endregion

if __name__ == "__main__":
    from dataset.build_dataset import make_dataset_from_manos_tracks
    from dataset.data_manager import BuildDataset, DataManager
    #import multiprocessing
    import warnings
    warnings.filterwarnings('ignore')
    import sys

    os.environ['PATH'] = './ffmpeg-7.0.2-amd64-static:' + os.environ['PATH']
    input_dir="../fromgcloud"
    output_dir = "../airmassRGB/supervised/" 

    video_specifico1(input_dir, output_dir)
    sys.exit(0)




    train_m, tracks_df_train = make_dataset_from_manos_tracks(input_dir, output_dir)

    cl10_intervals = tracks_df_train.groupby('id_cyc_unico').agg({'start_time':'first', 'end_time':'first'})

    filtered_df1 = filter_on_intervals(cl10_intervals, train_m.master_df)

    # devo gestire le tile grigie
    offsets = calc_tile_offsets(stride_x=213, stride_y=196)
    df_offsets = pd.DataFrame(offsets, columns=['tile_offset_x', 'tile_offset_y'])
    #df_offsets = train_m.master_df[['tile_offset_x','tile_offset_y']].value_counts().reset_index().drop(columns=['count'])
    assert df_offsets.shape[0] == 12
    #expanded_df = filtered_df1.groupby('path', group_keys=False).apply(lambda x: expand_group(x, df_offsets)).reset_index(drop=True)

    #make_animation_parallel_ffmpeg(expanded_df, nomefile='testset_CL10_148.mp4')
    
    make_animation_parallel_ffmpeg(filtered_df1, nomefile='testset_CL10_148.mp4')
