from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE,
)
from .datasets import MedicanesClsDataset  # RawFrameClsDataset, VideoClsDataset,

class DataManager():
    def __init__(self, mode, type, args):
        self.is_train = mode  # train o test
        self.type = type  # se UNsupervised o supervised
        if self.is_train:
            self.file_path = args.data_path
        else:
            self.file_path = args.test_path

    def get_specialization_dataset(self, args):
        transform = DataAugmentationForVideoMAEv2(args)
        dataset = HybridVideoMAE(
            root=args.data_root,
            setting=args.data_path,
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
        return dataset
    
    def get_classif_dataset(self, args):
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
        
        return dataset






class 