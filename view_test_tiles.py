# import numpy as np
# import os
# import datetime
# import time
# import random
# import json
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


# from dataset import build_dataset

# from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test
# from optim_factory import create_optimizer
import models # NECESSARIO ALTRIMENTI NON CARICA IL MODELLO (? è come Args? )
from timm.models import create_model
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from utils import NativeScalerWithGradNormCount as NativeScaler

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