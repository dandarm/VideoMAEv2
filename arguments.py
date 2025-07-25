import os

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # Convert strings to numerical types only when safe
            if isinstance(value, str):
                if value.isdigit():  # Check for integers
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep the value as a string if it can't be converted
            setattr(self, key, value)

    def __getattr__(self, name):
        """ Restituisce None se l'attributo non esiste, invece di sollevare un AttributeError """
        return None


def prepare_args(machine=None):
    # Default arguments from get_args()
    default_args = {
        'batch_size': 64,
        'epochs': 30,
        'update_freq': 1,
        'save_ckpt_freq': 100,
        'model': 'vit_base_patch16_224',
        'tubelet_size': 2,
        'input_size': 224,
        'with_checkpoint': True,
        'drop': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path': 0.1,
        'head_drop_rate': 0.0,
        'disable_eval_during_finetuning': False,
        'model_ema': False,
        'model_ema_decay': 0.9999,
        'model_ema_force_cpu': False,
        'opt': 'adamw',
        'opt_eps': 1e-8,
        'opt_betas': None,
        'clip_grad': None,
        'momentum': 0.9,
        'weight_decay': 0.05,
        'weight_decay_end': None,
        'lr': 1e-3,
        'layer_decay': 0.75,
        'warmup_lr': 1e-8,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        'warmup_steps': -1,
        'color_jitter': 0.4,
        'num_sample': 2,
        'aa': 'rand-m7-n4-mstd0.5-inc1',
        'smoothing': 0.1,
        'train_interpolation': 'bicubic',
        'crop_pct': None,
        'short_side_size': 224,
        'test_num_segment': 10,
        'test_num_crop': 3,
        'reprob': 0.25,
        'remode': 'pixel',
        'recount': 1,
        'resplit': False,
        'mixup': 0.8,
        'cutmix': 1.0,
        'cutmix_minmax': None,
        'mixup_prob': 1.0,
        'mixup_switch_prob': 0.5,
        'mixup_mode': 'batch',
        'finetune': '',
        'model_key': 'model|module',
        'model_prefix': '',
        'init_scale': 0.001,
        'use_mean_pooling': True,
        'data_path': '/your/data/path/',
        'data_root': '',
        'eval_data_path': None,
        'nb_classes': 400,
        'imagenet_default_mean_and_std': True,
        'num_segments': 1,
        'num_frames': 16,
        'sampling_rate': 4,
        'sparse_sample': False,
        'data_set': 'Kinetics-400',
        'fname_tmpl': 'img_{:05}.png',
        'start_idx': 1,
        'output_dir': '',
        'log_dir': None,
        'device': 'cuda',
        'seed': 0,
        'resume': '',
        'auto_resume': True,
        'save_ckpt': True,
        'start_epoch': 0,
        'eval': False,
        'validation': False,
        'dist_eval': False,
        'num_workers': 10,
        'pin_mem': True,
        'world_size': 1,
        'local_rank': -1,
        'dist_on_itp': False,
        'dist_url': 'env://',
        'dist_backend': 'nccl',
        'enable_deepspeed': False,
        ### solo per pretraining,
        'normlize_target': True
    }
    # user argument values
    user_args_pretrain = {
        'model': 'pretrain_videomae_giant_patch14_224',  # 'pretrain_videomae_base_patch16_224',
        'pretrained': True,  # Abilita il caricamento del checkpoint
        'finetune': './vit_g_hybrid_pt_1200e.pth',
        'init_ckpt': './vit_g_hybrid_pt_1200e.pth',
        'data_path': './UNtrain.csv',  #train_UNsupervised.csv',
        'test_path': './UNtest.csv',
        'log_dir': './output',
        'output_dir': './output',
        'data_set': 'medicanes',
        'mask_type': 'tube',
        'mask_ratio': 0.75,
        'decoder_mask_type': 'run_cell',
        'decoder_mask_ratio': 0.5,
        'batch_size': 6,
        'num_sample': 1,
        'num_frames': 16,
        'sampling_rate': 1,  # voglio tutti i frame temporali
        'num_workers': 1,     # TODO: verificare se deve stare a 1 o può aumentare
        'opt': 'adamw',
        'lr': 1e-3,
        'opt_betas': [0.9, 0.95],
        'warmup_epochs': 10,
        'epochs': 150,
        'save_ckpt_freq': 20,
        'decoder_depth': 4,
        'testing_epochs': 5
    }
    # user_args = {
    #     'model': 'vit_base_patch16_224',
    #     'data_path': './',
    #     'finetune': './pytorch_model.bin',
    #     'log_dir': './output',
    #     'output_dir': './output',
    #     'data_set': 'medicanes',
    #     'batch_size': '1',
    #     'input_size': '224',
    #     'short_side_size': '224',
    #     'save_ckpt_freq': '10',
    #     'num_frames': 16,
    #     'sampling_rate': '1',
    #     'num_workers': '10',
    #     'opt': 'adamw',
    #     'lr': '7e-4',
    #     'opt_betas': [0.9, 0.999],
    #     'weight_decay': '0.05',
    #     'layer_decay': '0.75',
    #     'test_num_segment': 1,
    #     'test_num_crop': '1',
    #     'epochs': '90',
    #     'dist_eval': '',
    #     'nb_classes': '1',
    #     'seed': 42,
    #     'warmup_steps': 1000
    # }
    # Merge dictionaries (user_args overrides default_args)
    # args_dict = {**default_args, **user_args}
    args_dict = {**default_args, **user_args_pretrain}
    # Convert args_dict to an Args object
    args = Args(**args_dict)
    return args



def prepare_finetuning_args(machine=None):
    # machine può essere 'ewc' o 'leonardo'


    # Default arguments from get_args()
    default_args = {
        'batch_size': 64,
        'epochs': 30,
        'update_freq': 1,
        'save_ckpt_freq': 100,
        'model': 'vit_base_patch16_224',
        'tubelet_size': 2,
        'input_size': 224,
        'with_checkpoint': True,
        'drop': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path': 0.1,
        'head_drop_rate': 0.0,
        'disable_eval_during_finetuning': False,
        'model_ema': False,
        'model_ema_decay': 0.9999,
        'model_ema_force_cpu': False,
        'opt': 'adamw',
        'opt_eps': 1e-8,
        'opt_betas': None,
        'clip_grad': None,
        'momentum': 0.9,
        'weight_decay': 0.05,
        'weight_decay_end': None,
        'lr': 1e-3,
        'layer_decay': 0.75,
        'warmup_lr': 1e-8,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        'warmup_steps': -1,
        'color_jitter': 0.4,
        'num_sample': 2,
        'aa': 'rand-m7-n4-mstd0.5-inc1',
        'smoothing': 0.1,
        'train_interpolation': 'bicubic',
        'crop_pct': None,
        'short_side_size': 224,
        'test_num_segment': 10,
        'test_num_crop': 3,
        'reprob': 0.25,
        'remode': 'pixel',
        'recount': 1,
        'resplit': False,
        'mixup': 0.0,
        'cutmix': 0.0,
        'cutmix_minmax': None,
        'mixup_prob': 1.0,
        'mixup_switch_prob': 0.5,
        'mixup_mode': 'batch',
        'finetune': '',
        'model_key': 'model|module',
        'model_prefix': '',
        'init_scale': 0.001,
        'use_mean_pooling': True,
        'data_path': '/your/data/path/',
        'data_root': './',
        'eval_data_path': None,
        'nb_classes': 400,
        'imagenet_default_mean_and_std': True,
        'num_segments': 1,
        'num_frames': 16,
        'sampling_rate': 4,
        'sparse_sample': False,
        'data_set': 'Kinetics-400',
        'fname_tmpl': 'img_{:05}.png',
        'start_idx': 1,
        'output_dir': '',
        'log_dir': None,
        'device': 'cuda',
        'seed': 0,
        'resume': '',
        'auto_resume': True,
        'save_ckpt': True,
        'start_epoch': 0,
        'eval': False,
        'validation': False,
        'dist_eval': False,
        'num_workers': 20,
        'pin_mem': True,
        'world_size': 1,
        'local_rank': -1,
        'dist_on_itp': False,
        'dist_url': 'env://',
        'enable_deepspeed': False,
        ### solo per pretraining,
        'normlize_target': True
    }
    # user argument values

    user_args_finetune = {
        'model': 'vit_giant_patch14_224',             #'vit_base_patch16_224'
        'pretrained': True,  # Abilita il caricamento del checkpoint
        #'finetune': './output/checkpoint_10k.pth', #./vit_g_hybrid_pt_1200e_k710_ft.pth
        'init_ckpt': './output/checkpoint_88k.pth',   # 'data_path': './checkpoint_10k
        'auto_resume': False,
        'load_for_test_mode': False,
        'data_path': './',
        'train_path': 'train_manos_w_1238.csv',   #'train_manos_836.csv',
        'test_path': 'test_manos_w_354.csv',  # 'test_manos_372.csv',
        'val_path': 'val_manos_w_2400.csv', 
        'log_dir': './output',
        'output_dir': './output',
        'data_set': 'medicanes',
        'nb_classes': 2,
        'mask_type': 'tube',
        'mask_ratio': 0.8,
        'decoder_mask_type': 'run_cell',
        'decoder_mask_ratio': 0.5,
        'batch_size': 1,  # 16 GPU *6 = 96  era il training di specializazione
        'num_sample': 1,
        'num_frames': 16,
        'sampling_rate': 1,  # voglio tutti i frame temporali
        'test_num_segment': 1, # 10
        'test_num_crop': 1,  # 3
        'num_workers': 10,
        'opt': 'adamw',
        'opt_betas': [0.9, 0.95],
  
        'save_ckpt_freq': 50,
        'decoder_depth': 4,
        'testing_epochs': 1,
        'cloudy': False,

        'epochs': 500,
        'start_epoch_for_saving_best_ckpt': 50,  # dopo 50 epoche inizia a salvare il best checkpoint
        'momentum': 0.9,
        'weight_decay': 0.05,
        'weight_decay_end': None,
        'lr': 1e-6,    # , 1e-3 , 9e-6
        'layer_decay': 0.75,
        'warmup_lr': 2e-7, #  1e-10
        'min_lr': 1e-6,  # era e-6      # 1e-8
        'warmup_epochs': 500,
        'warmup_steps': -1,
        #dist_eval
    }

    

    # Merge dictionaries (user_args overrides default_args)
    # args_dict = {**default_args, **user_args}
    args_dict = {**default_args, **user_args_finetune}

    if machine:
        machine_args_override = {}
        if machine == 'leonardo':
            machine_args_override['batch_size'] = 2

            ckpath = '$FAST/checkpoint-149.pth'
            exp_path = os.path.expandvars(ckpath)
            if "$HOME" in exp_path:
                raise EnvironmentError("La variabile d'ambiente HOME non è definita.")
            machine_args_override['init_ckpt'] = exp_path
            machine_args_override['pretrained'] = True
        elif machine == 'ewc':
            machine_args_override['device'] = 'cpu'

        args_dict = {**args_dict, **machine_args_override}

    # Convert args_dict to an Args object
    args = Args(**args_dict)
    return args

#region 
# argomenti da approfondire
# argomenti utili?
# 'tubelet_size': 2,
# 'input_size': 224,
# 'short_side_size': 224,
# 'num_segments': 1,
# 'num_frames': 16,
# 'sampling_rate': 4,
# 'mask_type': 'tube',
# 'mask_ratio': 0.9,
# 'decoder_mask_type': 'run_cell',
# 'decoder_mask_ratio': 0.5,
# 'num_sample': 2,
#
#
# 'decoder_depth': 4,
# 'test_num_segment': 10,
# 'test_num_crop': 3,
# 'reprob': 0.25,
# 'recount': 1,
# 'resplit': False,
# 'mixup': 0.8,
# 'cutmix': 1.0,
# 'cutmix_minmax': None,
# 'mixup_prob': 1.0,
# 'mixup_switch_prob': 0.5,
# 'mixup_mode': 'batch',
# 'init_scale': 0.001,

#endregion
