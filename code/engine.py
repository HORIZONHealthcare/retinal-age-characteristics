import os
import csv
import re
import torch
import numpy as np
import pandas as pd
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from util.datasets import denormalize_biomarker


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (_, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(samples).squeeze(1)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr, max_lr = float('inf'), 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, log_writer):
    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model.eval()
    preds, trues, id_list = [], [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        subject_id, images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        preds.extend(outputs.detach().cpu().numpy())
        trues.extend(targets.detach().cpu().numpy())
        id_list.extend(subject_id)

    preds = denormalize_biomarker(np.array(preds), args.biomarker_mean, args.biomarker_std)
    trues = denormalize_biomarker(np.array(trues), args.biomarker_mean, args.biomarker_std)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    me = np.mean(preds - trues)

    if log_writer:
        log_writer.add_scalar(f'perf/mae', mae, epoch)
        log_writer.add_scalar(f'perf/rmse', rmse, epoch)

    print(f'ME: {me:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}')

    metric_logger.synchronize_between_processes()

    suffix = getattr(args, 'eval_output_suffix', '') or ''
    suffix = re.sub(r'[^A-Za-z0-9._+-]+', '_', str(suffix).strip().strip('_'))
    file_stem = mode if not suffix else f'{mode}_{suffix}'

    results_path = os.path.join(exp_dir, f'metrics_{file_stem}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as f:
        wf = csv.writer(f)
        if not file_exists:
            wf.writerow(['epoch', 'ME', 'MAE', 'RMSE'])
        wf.writerow([epoch, me, mae, rmse])

    df_output = pd.DataFrame({
        'subject_id': id_list,
        'true_value': trues,
        'predicted_value': preds
    })
    output_csv_path = os.path.join(exp_dir, f'predictions_{file_stem}.csv')
    df_output.to_csv(output_csv_path, index=False)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mae
