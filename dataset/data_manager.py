from pathlib import Path
import pandas as pd
from functools import partial

from torch.utils.data import DataLoader, DistributedSampler

from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE)
from .datasets import MedicanesClsDataset  # RawFrameClsDataset, VideoClsDataset,
from .datasets import MedicanesTrackDataset
import sys
sys.path.append("../")
from medicane_utils.load_files import  load_all_images, load_all_images_in_intervals, get_intervals_in_tracks_df
from dataset.build_dataset import calc_tile_offsets, labeled_tiles_from_metadatafiles_maxfast, make_relabeled_master_df, solve_paths, get_train_test_validation_df, calc_avg_cld_idx

from dataset.build_dataset import create_and_save_tile_from_complete_df # importo anche questa contenuta in make_df_video

import utils
from utils import multiple_pretrain_samples_collate

class DataManager():
    def __init__(self, is_train, args, type_t='supervised', patch_size=[-1,-1], world_size=1, rank=0, specify_data_path=None):
        self.args = args
        self.is_train = is_train  # train o test
        self.type = type_t  # se UNsupervised o supervised
        
        if args.csv_folder is not None and args.csv_folder != '':
            specify_data_path = self.solve_csv_paths(args, specify_data_path)

        if self.is_train:
            self.file_path = args.train_path
        else:
            self.file_path = args.test_path   
        # se invece voglio specificare un path diverso, lo passo come parametro          
        if specify_data_path is not None:
            self.file_path = specify_data_path   

        #print(args.train_path, args.test_path, args.val_path, self.file_path)

        self.world_size = world_size 
        self.rank = rank

        #sistema parametri del patching
        if self.type == 'unsupervised':
            print("Patch size = %s" % str(patch_size))
            args.window_size = (args.num_frames // args.tubelet_size,
                                args.input_size // patch_size[0],
                                args.input_size // patch_size[1])
            print(f"Window size : {args.window_size}")
            args.patch_size = patch_size
                

        if args.num_sample > 1:
            collate_func = partial(multiple_pretrain_samples_collate, fold=False)
        else:
            collate_func = None
        self.collate_func = collate_func

        self.dataset = None
        self.data_loader = None

        self.dataset_len = -1

    def solve_csv_paths(self, args, specified_path=None):
        csv_folder = Path(args.csv_folder)
        if args.train_path is not None and args.train_path != '':
            args.train_path = csv_folder / args.train_path
        if args.test_path is not None and args.test_path != '':
            args.test_path = csv_folder / args.test_path
        if args.val_path is not None and args.val_path != '':
            args.val_path = csv_folder / args.val_path
        if specified_path is not None:
            specified_path = csv_folder / specified_path
            return specified_path


    def get_specialization_dataset(self, args):
        print("Getting dataset...") 
        transform = DataAugmentationForVideoMAEv2(args)
        dataset = HybridVideoMAE(
            root=args.data_root,
            file_path=self.file_path,
            train=self.is_train,
            test_mode=not self.is_train,
            name_pattern=args.fname_tmpl,
            video_ext='mp4',
            is_color=True,
            modality='rgb',
            num_segments=1,
            num_crop=1,
            new_length=args.num_frames,
            new_step=args.sampling_rate,
            transform=transform,
            temporal_jitter=False,
            lazy_init=False,
            num_sample=args.num_sample)
        print("Data Aug = %s" % str(transform))
        self.dataset_len = len(dataset)
        print(f"DATASET length: {self.dataset_len}")
        return dataset

    def create_specialization_dataloader(self, args):
        self.dataset = self.get_specialization_dataset(args)
        sampler = self.get_dist_sampler()
        print(f"Batch_size: {args.batch_size}")
        self.data_loader = DataLoader(self.dataset,
            batch_size=args.batch_size,
            shuffle=self.is_train,  # se è training -> shuffle , altrimenti no
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=self.collate_func,
            worker_init_fn=utils.seed_worker,
            persistent_workers=True,
            sampler=sampler,
        )

    def get_dist_sampler(self):
        print(f"Creo il DistributedSampler con world_size {self.world_size} e rank {self.rank}")
        # Shuffle solo in training; in validazione/inferenza meglio disattivare
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.is_train
        )
        print("Sampler_train = %s" % str(sampler))
        return sampler

    
    def get_classif_dataset(self, args):
        print("Getting dataset...") 
        if self.is_train:
            mode = 'train'
        else:
            mode = 'test'
        dataset = MedicanesClsDataset(
            anno_path=self.file_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            transform=None  # o una trasformazione custom
        )
        self.dataset_len = len(dataset)
        print(f"DATASET length: {self.dataset_len}")
        return dataset

    def create_classif_dataloader(self, args):
        self.dataset = self.get_classif_dataset(args)
        sampler = self.get_dist_sampler()
        print(f"Batch_size local: {args.batch_size}")
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            # shuffle è gestito dal DistributedSampler
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=self.is_train,  # non droppare in validazione/inferenza
            collate_fn=self.collate_func,
            # mancava il worker_init UNICA DIFFERENZA CON IL DATALOADER UNSUP
            persistent_workers=True,
            sampler=sampler,
        )

    # region Tracking dataset/dataloader
    def get_tracking_dataset(self, args):
        print("Getting TRACKING dataset (pixel coords)...")
        dataset = MedicanesTrackDataset(
            anno_path=self.file_path,
            data_root=args.data_root,
            clip_len=args.num_frames,
            transform=None,
        )
        self.dataset_len = len(dataset)
        print(f"DATASET length: {self.dataset_len}")
        return dataset

    def create_tracking_dataloader(self, args):
        self.dataset = self.get_tracking_dataset(args)
        sampler = self.get_dist_sampler()
        print(f"Batch_size local: {args.batch_size}")
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=self.is_train,
            collate_fn=self.collate_func,
            persistent_workers=True,
            sampler=sampler,
        )
        return self.data_loader

    # Convenience alias to mirror naming used elsewhere
    def get_tracking_dataloader(self, args):
        return self.create_tracking_dataloader(args)
    # endregion




from .build_dataset import create_df_video_from_master_df, filter_out_clear_sky, get_gruppi_date, create_final_df_csv, mark_neighboring_tiles


diz_types = {
        "path": 'string',
        "tile_offset_x": 'int16',
        "tile_offset_y": 'int16',
        "label": 'int16',
        "lat": 'object',
        "lon": 'object',
        "x_pix": 'object',
        "y_pix": 'object',
        "pressure": 'object',
        "name": 'string',
        "source": 'string',
        'id_cyc_unico': 'int32'
    }

# Converters per colonne ID: gestiscono valori vuoti/float/stringa (es. "7001283.0")
# convertendoli in interi o pd.NA, così si può mantenere diz_types invariato in read_csv.
conv = {
    'id_cyc_unico': lambda s: pd.NA if s is None or str(s).strip() == '' else int(float(s)),
    'idorig': lambda s: pd.NA if s is None or str(s).strip() == '' else int(float(s)),
    'id_final': lambda s: pd.NA if s is None or str(s).strip() == '' else int(float(s)),
}

class BuildDataset():
    def __init__(self, type, args=None, master_df_path=None):
        self.args = args
        self.type = type  # se UNsupervised o supervised

        self.master_df = None
        self.path_master_df = master_df_path
        self.df_video = None
        self.csv_file = None

        self.date_inizio_fine_gruppi = None


        self.string_format_time = '%Y-%m-%d %H:%M'

    def create_master_df(self, manos_file, input_dir_images, tracks_df=None):
        # Crea il master dataframe contenente tutte le immagini disponibili nella database folder, 
        # E associa le label derivanti dal manos_file
        # quindi il master_df contiene anche periodi non presenti nel manos_file

        if tracks_df is None:
            tracks_df = pd.read_csv(manos_file, parse_dates=['time', 'start_time', 'end_time'])

        sorted_metadata_files = load_all_images(input_dir_images)

        offsets_for_frame = calc_tile_offsets(stride_x=213, stride_y=196)
        self.master_df = labeled_tiles_from_metadatafiles_maxfast(sorted_metadata_files, tracks_df, offsets_for_frame)


    def create_master_df_short(self, input_dir_images, tracks_df):
        """
        carica le immagini che appartengono agli intervalli definiti dal tracks_df proveniente da Manos 
        invece di tutta la cartella, per velocizzare anche il video_df 
        """
        intervalli = get_intervals_in_tracks_df(tracks_df)
        sorted_metadata_files = load_all_images_in_intervals(input_dir_images, intervalli)
        print(f"sorted_metadata_files num :  {len(sorted_metadata_files)}")

        offsets_for_frame = calc_tile_offsets(stride_x=213, stride_y=196)
        self.master_df = labeled_tiles_from_metadatafiles_maxfast(sorted_metadata_files, tracks_df, offsets_for_frame)
        
    def create_master_df_year(self, input_dir_images, tracks_df, year: int):
        """Crea il master_df includendo TUTTE le immagini dell'anno richiesto,
        non solo quelle vicine alle tracce dei cicloni.

        - Filtra i file per anno prima di etichettare, così da mantenere anche
          i periodi senza cicloni (label=0).
        """
        # 1) Leggi tutte le immagini disponibili
        print(f"Carico tutte le immagini in {input_dir_images}")
        sorted_metadata_files = load_all_images(input_dir_images)
        # 2) Filtra per anno richiesto (robusto a dt None)
        filtered = [(p, dt) for (p, dt) in sorted_metadata_files if dt is not None and getattr(dt, 'year', None) == int(year)]
        print(f"Immagini nell'anno {year}: {len(filtered)}")
        # 3) Costruisci il master_df completo per quell'anno
        offsets_for_frame = calc_tile_offsets(stride_x=213, stride_y=196)
        self.master_df = labeled_tiles_from_metadatafiles_maxfast(filtered, tracks_df, offsets_for_frame)


    def load_master_df(self, master_df_path=None):
        if master_df_path:
            self.path_master_df = master_df_path

        self.master_df = pd.read_csv(self.path_master_df, dtype={
            "path": 'string',
            "tile_offset_x": 'int16',
            "tile_offset_y": 'int16',
            "label": 'int8',
            "x_pix": 'object',
            "y_pix": 'object',
            "pressure": 'object',
            "source": 'string',
            'id_cyc_unico': 'int32'
        }, parse_dates=['datetime', 'start_time', 'end_time'])
        self.master_df.drop(columns="Unnamed: 0", inplace=True)

    def calc_delta_time(self):
        self.master_df['delta_time'] = self.master_df.apply(calcola_delta_time, axis=1)


    def make_df_video(self, output_dir=None, idxs=None, is_to_balance=False, new_master_df=None):
        if new_master_df is not None:
           self.master_df = new_master_df 
        if self.master_df is None:
            self.load_master_df()
        self.df_video = create_df_video_from_master_df(self.master_df, idxs=idxs, output_dir=output_dir, is_to_balance=is_to_balance)

    def get_sequential_periods(self):
        gruppi_date = get_gruppi_date(self.master_df)

        #lista_date = []
        #lungh_gruppi = []
        self.date_inizio_fine_gruppi = []
        for df in gruppi_date:
            #group_datetime = list(df.groupby("datetime"))
            #for gd in group_datetime:
            #    lista_date.append(len(gd[1]))
            #lungh_gruppi.append(len(df)/len(gd[1])) # divido per le 20 tile di ciascun istante temporale
            start_date = df.datetime.iloc[0]
            end_date = df.datetime.iloc[-1]
            self.date_inizio_fine_gruppi.append((start_date, end_date))

    def print_sequential_periods(self):                
        i = 1
        totale = pd.to_timedelta(0)
        for start, end in self.date_inizio_fine_gruppi:
            s = start.strftime(self.string_format_time)
            e = end.strftime(self.string_format_time)
            delta = end - start
            totale += delta
            print(f"{i}:\t{s} → {e} - Δ {delta}")
            i += 1
        return totale


    def create_final_df_csv(self, output_dir, path_csv):        
        # outputdir serve per completare il path dei nomi file video nel csv
        df_dataset_csv = create_final_df_csv(self.df_video, output_dir)
        df_dataset_csv.to_csv(path_csv, index=False)
        self.csv_file = path_csv


    def get_data_ready(self, df_tracks, input_dir, output_dir, csv_file=None, relabeling=False, is_to_balance=True):
        #self.create_master_df(manos_file=None, input_dir_images=input_dir, tracks_df=df_tracks)
        self.create_master_df_short(input_dir_images=input_dir, tracks_df=df_tracks)

        # cambia le etichette per togliere le fasi iniziali e finali dei cicloni
        if relabeling:
            print(f"Relabeling master_df +- {24}h...")
            df_mod = make_relabeled_master_df(self, hours_shift=24)
            self.make_df_video(new_master_df=df_mod, output_dir=output_dir,  is_to_balance=is_to_balance)
        else:
            self.make_df_video(output_dir=output_dir, is_to_balance=is_to_balance)

        if self.args.cloudy:
            df_v = filter_out_clear_sky(output_dir, self)
            self.df_video = df_v.copy()

        # Marca le tile adiacenti alle positive (8-neighborhood, solo negativi)
        # Usa la stessa definizione di griglia degli offset (stride 213x196)
        if self.df_video is not None and not self.df_video.empty:
            self.df_video = mark_neighboring_tiles(self.df_video, stride_x=213, stride_y=196, include_diagonals=True, only_negatives=True)

        if csv_file is None:
            csv_file = f"data_{self.df_video.shape[0]}.csv"
        else:
            csv_file = csv_file + f"_{self.df_video.shape[0]}.csv"
        self.create_final_df_csv(output_dir, csv_file)

    def get_data_ready_full_year(self, df_tracks, input_dir, output_dir, year: int, csv_file=None):
        """Prepara i dati includendo tutte le immagini di un anno intero.

        - Non limita il caricamento ai soli intervalli attorno ai cicloni.
        - Etichetta comunque usando df_tracks (label=1 se inside, altrimenti 0).
        - Costruisce il df_video e salva il CSV finale se richiesto.
        """
        # 1) Master df di un anno intero
        self.create_master_df_year(input_dir_images=input_dir, tracks_df=df_tracks, year=year)

        # 2) Dataframe video
        self.make_df_video(output_dir=output_dir, is_to_balance=False)

        # 3) Opzionale: calcolo indice nuvolosità medio per video (senza filtrare)
        if self.args is not None and getattr(self.args, 'cloudy', False):
            print("Calcolo avg_cloud_idx per video (senza filtrare)...")
            try:
                full_paths = output_dir + self.df_video['path']
                self.df_video['avg_cloud_idx'] = full_paths.apply(calc_avg_cld_idx)
            except Exception as e:
                print(f"Warning: impossibile calcolare avg_cloud_idx: {e}")

        # 4) Marca neighboring
        if self.df_video is not None and not self.df_video.empty:
            self.df_video = mark_neighboring_tiles(self.df_video, stride_x=213, stride_y=196, include_diagonals=True, only_negatives=True)

        # 5) CSV
        if csv_file is None:
            csv_file = f"data_{self.df_video.shape[0]}.csv"
        else:
            csv_file = csv_file + f"_{self.df_video.shape[0]}.csv"
        self.create_final_df_csv(output_dir, csv_file)

        





# New builder for tracking with pixel coordinates
class BuildTrackingDataset(BuildDataset):
    """
    Build a CSV for tracking that includes pixel coordinates for the last
    frame of each video. It keeps only videos with label == 1 by default
    and drops tile offsets in the final CSV.

    Output CSV columns: path, start, end, x_pix, y_pix
    - path: absolute folder path (output_dir + relative path)
    - start: 1
    - end: number of frames (default: 16)
    - x_pix, y_pix: pixel coordinates at the video's end_time
    """

    @staticmethod
    def _first_val(val):
        """Extract first element if list-like; try to parse from string, else return NaN."""
        import ast
        import math
        if isinstance(val, (list, tuple)):
            return float(val[0]) if len(val) > 0 else math.nan
        if isinstance(val, str):
            s = val.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    return float(parsed[0])
                # fallthrough to try converting directly
                return float(parsed)
            except Exception:
                try:
                    return float(s)
                except Exception:
                    return math.nan
        try:
            return float(val)
        except Exception:
            return float('nan')

    def create_tracking_df(self, output_dir: str, only_label_1: bool = True, num_frames: int = 16) -> pd.DataFrame:
        if self.df_video is None or self.master_df is None:
            raise RuntimeError("df_video/master_df non pronti. Esegui make_df_video/create_master_df prima.")

        dfv = self.df_video.copy()
        if only_label_1 and 'label' in dfv.columns:
            dfv = dfv[dfv['label'] == 1]

        # Join df_video rows with the corresponding master_df row at end_time and the same offsets
        coords_cols = ['datetime', 'tile_offset_x', 'tile_offset_y', 'x_pix', 'y_pix']
        mdf = self.master_df[coords_cols].copy()

        merged = dfv.merge(
            mdf,
            left_on=['end_time', 'tile_offset_x', 'tile_offset_y'],
            right_on=['datetime', 'tile_offset_x', 'tile_offset_y'],
            how='left'
        )

        merged['x_pix_last'] = merged['x_pix'].apply(self._first_val)
        merged['y_pix_last'] = merged['y_pix'].apply(self._first_val)
        # Convert to tile-relative coordinates and drop offsets in final CSV
        merged['x_pix_rel'] = merged['x_pix_last'] - merged['tile_offset_x']
        merged['y_pix_rel'] = merged['y_pix_last'] - merged['tile_offset_y']

        out = merged[['path']].copy()
        out['path'] = output_dir + out['path']
        out['start'] = 1
        out['end'] = int(num_frames)
        out['x_pix'] = merged['x_pix_rel']
        out['y_pix'] = merged['y_pix_rel']

        return out

    def create_tracking_csv(self, output_dir: str, path_csv: str, only_label_1: bool = True, num_frames: int = 16) -> None:
        df = self.create_tracking_df(output_dir=output_dir, only_label_1=only_label_1, num_frames=num_frames)
        df.to_csv(path_csv, index=False)
        self.csv_file = path_csv

    def prepare_data(self, df_tracks, input_dir: str, output_dir: str) -> None:
        """Prepare master_df and df_video without writing a classification CSV.

        Mirrors BuildDataset.get_data_ready but avoids saving the classif CSV.
        """
        self.create_master_df_short(input_dir_images=input_dir, tracks_df=df_tracks)
        # Build df_video WITHOUT saving any tiles yet
        print("Creo video senza salvarli...")
        self.make_df_video(output_dir=None, is_to_balance=False)

        print("Save only positive (label==1) video tiles...")        
        if self.df_video is not None and not self.df_video.empty:
            df_pos = self.df_video
            if 'label' in df_pos.columns:
                df_pos = df_pos[df_pos['label'] == 1]
            if not df_pos.empty:
                create_and_save_tile_from_complete_df(df_pos, output_dir)


# Calcola distanza minima rispetto all'intervallo [start_time, end_time]
def calcola_delta_time(row):
    dt = row['datetime']
    start = row['start_time']
    end = row['end_time']
    
    if pd.isna(dt) or pd.isna(start) or pd.isna(end):
        return pd.NaT  # oppure np.nan o pd.Timedelta.max, a seconda della logica
    return min(abs(dt - start), abs(dt - end))


from arguments import prepare_finetuning_args, Args
def make_validation_data_builder_from_manos_tracks(manos_track_file, input_dir, output_dir):
    args = prepare_finetuning_args()  # TODO: spostare tra gli argomenti obbligatori

    output_dir = solve_paths(output_dir)
    input_dir = solve_paths(input_dir)

    tracks_df = pd.read_csv(manos_track_file, parse_dates=['time', 'start_time', 'end_time'])
    tracks_df_train, tracks_df_test, tracks_df_val = get_train_test_validation_df(tracks_df, 0.7, args.val_split_fraction)
    val_b = BuildDataset(type='SUPERVISED', args=args)
    val_b.get_data_ready(tracks_df_val, input_dir, output_dir, csv_file="val_manos_w", is_to_balance=False)
    #val_b.get_data_ready(tracks_df_test, input_dir, output_dir, csv_file="val_manos_w", is_to_balance=True)
    return val_b

def make_validation_data_builder_from_entire_year(year, input_dir, output_dir):
    args = prepare_finetuning_args()  # TODO: spostare tra gli argomenti obbligatori

    output_dir = solve_paths(output_dir)
    input_dir = solve_paths(input_dir)

    manos_track_file = "medicane_data_input/more_medicanes_time_updated.csv"
    tracks_df = pd.read_csv(manos_track_file, parse_dates=['time', 'start_time', 'end_time'])

    bd_full = BuildDataset(type='SUPERVISED', args=args)
    bd_full.get_data_ready_full_year(tracks_df, input_dir, output_dir, year, csv_file=f"full_year_{year}")

    return bd_full


# dataset/data_manager.py
def make_tracking_data_builder_from_csv(
    manos_track_file: str,
    selected_csv: str,
    input_dir: str,
    output_dir: str,
    split: str = "test",
    args=None,
):
    """Crea un BuildTrackingDataset e mantiene solo i video elencati nel CSV selezionato."""

    output_dir = solve_paths(output_dir)
    input_dir = solve_paths(input_dir)

    tracks_df = pd.read_csv(
        manos_track_file,
        parse_dates=["time", "start_time", "end_time"],
        #dtype=conv,  # riusa i converter già definiti in testa al file
    )

    val_fraction = args.val_split_fraction
    tracks_df_train, tracks_df_test, tracks_df_val = get_train_test_validation_df(
        tracks_df, 0.7, val_fraction, id_col="id_final"
    )
    split_map = {"train": tracks_df_train, "test": tracks_df_test, "val": tracks_df_val}
    if split not in split_map:
        raise ValueError(f"split '{split}' non valido. Usa: {list(split_map)}")

    builder = BuildTrackingDataset(type="SUPERVISED", args=args)
    builder.prepare_data(split_map[split], input_dir, output_dir)

    selected_df = pd.read_csv(selected_csv)
    if "path" not in selected_df.columns:
        raise ValueError(f"Nel file {selected_csv} manca la colonna 'path'")

    selected_names = (
        selected_df["path"]
        .astype(str)
        .apply(lambda p: Path(p).name if p else p)
    )
    keep = builder.df_video["path"].isin(set(selected_names))
    builder.df_video = builder.df_video[keep].reset_index(drop=True)

    if builder.master_df is not None:
        valid_orig = builder.df_video["orig_paths"].explode().unique()
        builder.master_df = builder.master_df[builder.master_df["path"].isin(valid_orig)]

    return builder, split_map[split]
