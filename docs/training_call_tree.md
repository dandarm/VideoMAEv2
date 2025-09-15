# Training Scripts Dependency and Call Tree
## specialization.py
### Imports
- PIL.Image
- arguments.Args
- arguments.prepare_args
- datetime
- engine_for_pretraining.test
- engine_for_pretraining.train_one_epoch
- functools.partial
- json
- model_analysis.get_dataloader
- model_analysis.get_dataset_dataloader
- numpy as np
- optim_factory.create_optimizer
- os
- time
- torch
- torch.distributed as dist
- torch.nn.functional as F
- utils
- utils.NativeScalerWithGradNormCount as NativeScaler
- utils.get_model
- utils.multiple_pretrain_samples_collate
- utils.setup_for_distributed

### Function `launch_specialization_training` Calls
- NativeScaler
- create_optimizer
- data_loader_train.sampler.set_epoch
- datetime.timedelta
- dist.init_process_group
- f.write
- format
- get_dataloader
- get_dataset_dataloader
- get_model
- int
- json.dumps
- len
- log_writer.flush
- log_writer.set_step
- max
- min
- open
- os.makedirs
- os.path.exists
- os.path.join
- p.numel
- prepare_args
- pretrained_model.parameters
- pretrained_model.to
- print
- range
- setup_for_distributed
- str
- sum
- test
- test_stats.items
- time.time
- torch.cuda.empty_cache
- torch.cuda.set_device
- torch.device
- torch.nn.parallel.DistributedDataParallel
- train_one_epoch
- train_stats.items
- utils.TensorboardLogger
- utils.auto_load_model
- utils.cosine_scheduler
- utils.get_rank
- utils.get_resources
- utils.get_world_size
- utils.is_main_process
- utils.save_model

```mermaid
graph TD
    specialization_py[specialization.py] --> launch_specialization_training
    launch_specialization_training --> NativeScaler
    launch_specialization_training --> create_optimizer
    launch_specialization_training --> data_loader_train_sampler_set_epoch
    launch_specialization_training --> datetime_timedelta
    launch_specialization_training --> dist_init_process_group
    launch_specialization_training --> f_write
    launch_specialization_training --> format
    launch_specialization_training --> get_dataloader
    launch_specialization_training --> get_dataset_dataloader
    launch_specialization_training --> get_model
    launch_specialization_training --> int
    launch_specialization_training --> json_dumps
    launch_specialization_training --> len
    launch_specialization_training --> log_writer_flush
    launch_specialization_training --> log_writer_set_step
    launch_specialization_training --> max
    launch_specialization_training --> min
    launch_specialization_training --> open
    launch_specialization_training --> os_makedirs
    launch_specialization_training --> os_path_exists
    launch_specialization_training --> os_path_join
    launch_specialization_training --> p_numel
    launch_specialization_training --> prepare_args
    launch_specialization_training --> pretrained_model_parameters
    launch_specialization_training --> pretrained_model_to
    launch_specialization_training --> print
    launch_specialization_training --> range
    launch_specialization_training --> setup_for_distributed
    launch_specialization_training --> str
    launch_specialization_training --> sum
    launch_specialization_training --> test
    launch_specialization_training --> test_stats_items
    launch_specialization_training --> time_time
    launch_specialization_training --> torch_cuda_empty_cache
    launch_specialization_training --> torch_cuda_set_device
    launch_specialization_training --> torch_device
    launch_specialization_training --> torch_nn_parallel_DistributedDataParallel
    launch_specialization_training --> train_one_epoch
    launch_specialization_training --> train_stats_items
    launch_specialization_training --> utils_TensorboardLogger
    launch_specialization_training --> utils_auto_load_model
    launch_specialization_training --> utils_cosine_scheduler
    launch_specialization_training --> utils_get_rank
    launch_specialization_training --> utils_get_resources
    launch_specialization_training --> utils_get_world_size
    launch_specialization_training --> utils_is_main_process
    launch_specialization_training --> utils_save_model
```

## classification.py
### Imports
- argparse
- arguments.Args
- arguments.prepare_finetuning_args
- dataset.build_dataset
- dataset.data_manager.DataManager
- datetime
- engine_for_finetuning.final_test
- engine_for_finetuning.train_one_epoch
- engine_for_finetuning.validation_one_epoch
- engine_for_pretraining.test
- engine_for_pretraining.train_one_epoch
- functools.partial
- json
- models
- numpy as np
- optim_factory.create_optimizer
- os
- random
- time
- timm.data.mixup.Mixup
- timm.loss.LabelSmoothingCrossEntropy
- timm.loss.SoftTargetCrossEntropy
- timm.models.create_model
- torch
- torch.backends.cudnn as cudnn
- torch.distributed as dist
- torch.nn.functional as F
- utils
- utils.NativeScalerWithGradNormCount as NativeScaler
- utils.multiple_pretrain_samples_collate
- utils.setup_for_distributed
- warnings

### Function `launch_finetuning_classification` Calls
- DataManager
- LabelSmoothingCrossEntropy
- Mixup
- NativeScaler
- SoftTargetCrossEntropy
- create_model
- create_optimizer
- datetime.timedelta
- dist.init_process_group
- f.write
- getattr
- hasattr
- int
- isinstance
- json.dumps
- labels.max
- len
- log_stats.items
- loss_scaler.state_dict
- model_without_ddp.state_dict
- np.array
- np.bincount
- np.random.seed
- open
- optimizer.state_dict
- os.makedirs
- os.path.exists
- os.path.join
- p.numel
- prepare_finetuning_args
- pretrained_model.parameters
- pretrained_model.to
- print
- random.seed
- range
- round
- setup_for_distributed
- str
- sum
- test_m.create_classif_dataloader
- time.time
- to_numpy
- torch.cuda.empty_cache
- torch.cuda.set_device
- torch.device
- torch.manual_seed
- torch.nn.parallel.DistributedDataParallel
- torch.save
- torch.tensor
- train_m.create_classif_dataloader
- train_one_epoch
- train_stats.items
- utils.auto_load_model
- utils.cosine_scheduler
- utils.get_resources
- val2_stats.items
- val_m.create_classif_dataloader
- val_stats.items
- validation_one_epoch
- vars

```mermaid
graph TD
    classification_py[classification.py] --> launch_finetuning_classification
    launch_finetuning_classification --> DataManager
    launch_finetuning_classification --> LabelSmoothingCrossEntropy
    launch_finetuning_classification --> Mixup
    launch_finetuning_classification --> NativeScaler
    launch_finetuning_classification --> SoftTargetCrossEntropy
    launch_finetuning_classification --> create_model
    launch_finetuning_classification --> create_optimizer
    launch_finetuning_classification --> datetime_timedelta
    launch_finetuning_classification --> dist_init_process_group
    launch_finetuning_classification --> f_write
    launch_finetuning_classification --> getattr
    launch_finetuning_classification --> hasattr
    launch_finetuning_classification --> int
    launch_finetuning_classification --> isinstance
    launch_finetuning_classification --> json_dumps
    launch_finetuning_classification --> labels_max
    launch_finetuning_classification --> len
    launch_finetuning_classification --> log_stats_items
    launch_finetuning_classification --> loss_scaler_state_dict
    launch_finetuning_classification --> model_without_ddp_state_dict
    launch_finetuning_classification --> np_array
    launch_finetuning_classification --> np_bincount
    launch_finetuning_classification --> np_random_seed
    launch_finetuning_classification --> open
    launch_finetuning_classification --> optimizer_state_dict
    launch_finetuning_classification --> os_makedirs
    launch_finetuning_classification --> os_path_exists
    launch_finetuning_classification --> os_path_join
    launch_finetuning_classification --> p_numel
    launch_finetuning_classification --> prepare_finetuning_args
    launch_finetuning_classification --> pretrained_model_parameters
    launch_finetuning_classification --> pretrained_model_to
    launch_finetuning_classification --> print
    launch_finetuning_classification --> random_seed
    launch_finetuning_classification --> range
    launch_finetuning_classification --> round
    launch_finetuning_classification --> setup_for_distributed
    launch_finetuning_classification --> str
    launch_finetuning_classification --> sum
    launch_finetuning_classification --> test_m_create_classif_dataloader
    launch_finetuning_classification --> time_time
    launch_finetuning_classification --> to_numpy
    launch_finetuning_classification --> torch_cuda_empty_cache
    launch_finetuning_classification --> torch_cuda_set_device
    launch_finetuning_classification --> torch_device
    launch_finetuning_classification --> torch_manual_seed
    launch_finetuning_classification --> torch_nn_parallel_DistributedDataParallel
    launch_finetuning_classification --> torch_save
    launch_finetuning_classification --> torch_tensor
    launch_finetuning_classification --> train_m_create_classif_dataloader
    launch_finetuning_classification --> train_one_epoch
    launch_finetuning_classification --> train_stats_items
    launch_finetuning_classification --> utils_auto_load_model
    launch_finetuning_classification --> utils_cosine_scheduler
    launch_finetuning_classification --> utils_get_resources
    launch_finetuning_classification --> val2_stats_items
    launch_finetuning_classification --> val_m_create_classif_dataloader
    launch_finetuning_classification --> val_stats_items
    launch_finetuning_classification --> validation_one_epoch
    launch_finetuning_classification --> vars
```

## tracking.py
### Imports
- argparse
- arguments.prepare_tracking_args
- dataset.data_manager.DataManager
- dataset.tracking_dataset.MedicanesTrackDataset
- engine_for_tracking.evaluate
- engine_for_tracking.train_one_epoch
- json
- models.tracking_model.RegressionHead
- models.tracking_model.create_tracking_model
- numpy as np
- optim_factory.create_optimizer
- os
- random
- time
- torch
- torch.backends.cudnn as cudnn
- torch.distributed as dist
- torch.nn
- torch.utils.data.DataLoader
- torch.utils.data.DistributedSampler
- utils
- utils.setup_for_distributed

### Function `launch_tracking` Calls
- DataManager
- create_optimizer
- create_tracking_model
- dist.init_process_group
- evaluate
- f.write
- float
- getattr
- isinstance
- json.dumps
- log_stats.items
- model.to
- model_without_ddp.state_dict
- nn.MSELoss
- np.random.seed
- open
- optimizer.state_dict
- os.makedirs
- os.path.exists
- os.path.join
- prepare_tracking_args
- print
- random.seed
- range
- round
- setup_for_distributed
- str
- test_m.get_tracking_dataloader
- time.gmtime
- time.strftime
- time.time
- torch.cuda.is_available
- torch.cuda.manual_seed
- torch.cuda.set_device
- torch.device
- torch.manual_seed
- torch.nn.parallel.DistributedDataParallel
- torch.save
- train_m.get_tracking_dataloader
- train_one_epoch
- train_stats.items
- type
- utils.get_resources
- val_m.get_tracking_dataloader
- val_stats.get
- val_stats.items

```mermaid
graph TD
    tracking_py[tracking.py] --> launch_tracking
    launch_tracking --> DataManager
    launch_tracking --> create_optimizer
    launch_tracking --> create_tracking_model
    launch_tracking --> dist_init_process_group
    launch_tracking --> evaluate
    launch_tracking --> f_write
    launch_tracking --> float
    launch_tracking --> getattr
    launch_tracking --> isinstance
    launch_tracking --> json_dumps
    launch_tracking --> log_stats_items
    launch_tracking --> model_to
    launch_tracking --> model_without_ddp_state_dict
    launch_tracking --> nn_MSELoss
    launch_tracking --> np_random_seed
    launch_tracking --> open
    launch_tracking --> optimizer_state_dict
    launch_tracking --> os_makedirs
    launch_tracking --> os_path_exists
    launch_tracking --> os_path_join
    launch_tracking --> prepare_tracking_args
    launch_tracking --> print
    launch_tracking --> random_seed
    launch_tracking --> range
    launch_tracking --> round
    launch_tracking --> setup_for_distributed
    launch_tracking --> str
    launch_tracking --> test_m_get_tracking_dataloader
    launch_tracking --> time_gmtime
    launch_tracking --> time_strftime
    launch_tracking --> time_time
    launch_tracking --> torch_cuda_is_available
    launch_tracking --> torch_cuda_manual_seed
    launch_tracking --> torch_cuda_set_device
    launch_tracking --> torch_device
    launch_tracking --> torch_manual_seed
    launch_tracking --> torch_nn_parallel_DistributedDataParallel
    launch_tracking --> torch_save
    launch_tracking --> train_m_get_tracking_dataloader
    launch_tracking --> train_one_epoch
    launch_tracking --> train_stats_items
    launch_tracking --> type
    launch_tracking --> utils_get_resources
    launch_tracking --> val_m_get_tracking_dataloader
    launch_tracking --> val_stats_get
    launch_tracking --> val_stats_items
```

