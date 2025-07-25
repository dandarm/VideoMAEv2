# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import os
os.environ["INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

import sys
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale

def debug_tensor(t, name="tensor", n_vals=8):
    """
    Stampa forma, tipo, device, intervallo di valori e
    i primi n_vals elementi (piatti) di un tensore PyTorch.
    """
    t_det = t.detach()            # stacca dal grafo
    flat  = t_det.flatten()       # appiattisci: facile da stampare
    print(f"\n{name}:")
    print(f"  shape   → {tuple(t_det.shape)}")
    print(f"  dtype   → {t_det.dtype}, device → {t_det.device}")
    if torch.is_floating_point(t_det):
        print(f"  range   → [{t_det.min():.4f}, {t_det.max():.4f}]")
        print(f"  mean    → {t_det.mean():.4f}, std → {t_det.std():.4f}")
    print(f"  first {n_vals} vals → {flat[:n_vals].cpu().tolist()}")



def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    
    # importante per lo shuffle
    data_loader.sampler.set_epoch(epoch) 

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    #print(f"param group lr scale {param_group['lr_scale']}")
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets_ondevice = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # mixup handle 3th & 4th dimension
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets_ondevice = mixup_fn(samples, targets_ondevice)
            samples = samples.view(B, C, T, H, W)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets_ondevice, criterion)
        else:
            if device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss, output = train_class_batch(model, samples, targets_ondevice, criterion)
            else:
                loss, output = train_class_batch(model, samples, targets_ondevice, criterion)

        # debugging the output
        #debug_tensor(output, name="output", n_vals=16)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()

            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if device.type == 'cuda':
            torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets_ondevice).float().mean()
        else:
            class_acc = None


        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        targets_ondevice = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets_ondevice)

        acc1,  = accuracy(output, targets_ondevice, topk=(1, ))  # Modifcato per classificazione binaria

        # calculate false positive etc...
        
        #tn, fp, fn, tp = confusion_matrix(target, output.cpu().numpy().ravel()).ravel()
        y_true = targets_ondevice  # shape: (N,), dtype: torch.int64
        #y_pred = (output >= 0.5).long().squeeze()  # applica soglia su probabilità  NO è SBAGLIATO! SONO DUE LOGITS
        y_pred = output.argmax(dim=1)          # shape (N,)

        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['fpr'].update(fpr, n=batch_size)
        metric_logger.meters['fnr'].update(fnr, n=batch_size)

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())),
                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1,  = accuracy(output, target, topk=(1, ))  # modificato: classificazione binaria

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        #f.write("{}, {}\n".format(acc1, acc5))
        f.write("{}\n".format(acc1))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
