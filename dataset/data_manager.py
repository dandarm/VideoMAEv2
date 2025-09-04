from pathlib import Path
import pandas as pd
from functools import partial

from torch.utils.data import DataLoader, DistributedSampler

from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE)
from .datasets import MedicanesClsDataset  # RawFrameClsDataset, VideoClsDataset,
from medicane_utils.load_files import  load_all_images, load_all_images_in_intervals, get_intervals_in_tracks_df
from dataset.build_dataset import calc_tile_offsets, labeled_tiles_from_metadatafiles_maxfast, make_relabeled_master_df


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




from .build_dataset import create_df_video_from_master_df, filter_out_clear_sky, get_gruppi_date, create_final_df_csv, mark_neighboring_tiles


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

        





# Calcola distanza minima rispetto all'intervallo [start_time, end_time]

def calcola_delta_time(row):
    dt = row['datetime']
    start = row['start_time']
    end = row['end_time']
    
    if pd.isna(dt) or pd.isna(start) or pd.isna(end):
        return pd.NaT  # oppure np.nan o pd.Timedelta.max, a seconda della logica
    return min(abs(dt - start), abs(dt - end))
