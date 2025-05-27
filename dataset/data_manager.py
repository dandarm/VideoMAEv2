import pandas as pd
from functools import partial

from torch.utils.data import DataLoader, DistributedSampler

from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE)
from .datasets import MedicanesClsDataset  # RawFrameClsDataset, VideoClsDataset,

import utils
from utils import multiple_pretrain_samples_collate

class DataManager():
    def __init__(self, is_train, args, type_t, patch_size=[-1,-1], world_size=1, rank=0):
        self.args = args
        self.is_train = is_train  # train o test
        self.type = type_t  # se UNsupervised o supervised
        if self.is_train:
            self.file_path = args.train_path
        else:
            self.file_path = args.test_path

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


    def get_specialization_dataset(self, args):
        print("Getting dataset...") 
        transform = DataAugmentationForVideoMAEv2(args)
        dataset = HybridVideoMAE(
            root=args.data_root,
            file_path=args.file_path,
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
        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
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
        print(f"Batch_size: {args.batch_size}")
        self.data_loader = DataLoader(self.dataset,
            batch_size=args.batch_size,
            #shuffle=self.is_train,  # se è training -> shuffle , altrimenti no  # MA NON PUÒ ANDARE INSIEME CON SAMPLER
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=self.collate_func,
            # mancava il worker_init UNICA DIFFERENZA CON IL DATALOADER UNSUP
            persistent_workers=True,
            sampler=sampler
        )




from .build_dataset import create_df_video_from_master_df, get_gruppi_date, create_final_df_csv


class BuildDataset():
    def __init__(self, type, args=None, master_df_path=None):
        #self.args = args
        self.type = type  # se UNsupervised o supervised

        self.master_df = None
        self.path_master_df = master_df_path
        self.df_video = None

        self.date_inizio_fine_gruppi = None


        self.string_format_time = '%Y-%m-%d %H:%M'

    def create_master_df(self):
        pass

    def load_master_df(self, master_df_path=None):
        if master_df_path:
            self.path_master_df = master_df_path

        self.master_df = pd.read_csv(self.path_master_df, dtype={
            "path": 'string',
            "tile_offset_x": 'int16',
            "tile_offset_y": 'int16',
            "label": 'int16',
            "lat": 'object',
            "lon": 'object',
            "x_pix": 'object',
            "y_pix": 'object',
            "name": 'string',
            "source": 'string'
        }, parse_dates=['datetime'])
        self.master_df.drop(columns="Unnamed: 0", inplace=True)


    def make_df_video(self, output_dir=None, idxs=None, is_to_balance=False):
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
        for start, end in self.date_inizio_fine_gruppi:
            s = start.strftime(self.string_format_time)
            e = end.strftime(self.string_format_time)
            delta = end - start
            print(f"{i}:\t{s} → {e} - Δ {delta}")
            i += 1


    def create_final_df_csv(self, output_dir, path_csv):        
        # outputdir serve per completare il path dei nomi file video nel csv
        df_dataset_csv = create_final_df_csv(self.df_video, output_dir)
        df_dataset_csv.to_csv(path_csv, index=False)

        
