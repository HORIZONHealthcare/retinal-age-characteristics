import argparse
import datetime
import json
import yaml
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import BiomarkerModel
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, BiomarkerDataset, build_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, evaluate

import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def flatten_config(config, prefix=''):
    """Flatten nested config dict: backbone.version -> backbone_version."""
    flat = {}
    for k, v in config.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, key))
        else:
            flat[key] = v
    return flat


def get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', default='config/default.yaml', type=str)
    pre_args, _ = pre_parser.parse_known_args()

    config = flatten_config(load_config(pre_args.config))

    parser = argparse.ArgumentParser('Biomarker regression training')
    parser.add_argument('--config', default='config/default.yaml', type=str, help='Path to config file')

    # Backbone
    parser.add_argument('--backbone_version', type=str)
    parser.add_argument('--backbone_model', type=str)
    parser.add_argument('--backbone_pretrained', type=lambda x: x.lower() == 'true')
    parser.add_argument('--backbone_img_size', type=int)

    # Prediction head
    parser.add_argument('--head_hidden_dim', type=int)
    parser.add_argument('--head_dropout', type=float)

    # Data
    parser.add_argument('--splits_dir', type=str)
    parser.add_argument('--train_csv_name', type=str)
    parser.add_argument('--val_csv_name', type=str)
    parser.add_argument('--test_csv_name', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--test_name', type=str)
    parser.add_argument('--eval_output_suffix', type=str)
    parser.add_argument('--biomarker_mean', type=float)
    parser.add_argument('--biomarker_std', type=float)

    # Training
    parser.add_argument('--train_mode', type=str)
    parser.add_argument('--finetune_k', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--accum_iter', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--blr', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--layer_decay', type=float)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--clip_grad', type=float)

    # Augmentation
    parser.add_argument('--color_jitter', type=float)
    parser.add_argument('--aa', type=str)
    parser.add_argument('--reprob', type=float)
    parser.add_argument('--remode', type=str)
    parser.add_argument('--recount', type=int)

    # System
    parser.add_argument('--device', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--pin_mem', action='store_true')

    # Distributed
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', type=str)

    # Experiment
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_model', type=lambda x: x.lower() == 'true')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true')

    # Sampling
    parser.add_argument('--balanced_sampler', type=lambda x: x.lower() == 'true')
    parser.add_argument('--balanced_bin_width', type=int)
    parser.add_argument('--balanced_min_count', type=int)

    parser.set_defaults(**config)
    args = parser.parse_args()
    return args


def main(args):
    if args.resume and not args.eval:
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        args = checkpoint['args']
        args.resume = resume

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = BiomarkerModel(
        version=args.backbone_version,
        model=args.backbone_model,
        pretrained=args.backbone_pretrained,
        img_size=args.backbone_img_size,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    model.set_training_mode(args.train_mode, k=args.finetune_k)

    # Build datasets
    if not args.eval:
        dataset_train = build_dataset(split='train', args=args)
        dataset_val = build_dataset(split='val', args=args)

    # Test set: use --test_csv if provided, otherwise splits_dir/test.csv
    if args.test_csv:
        test_transform = build_transform(is_train=False, args=args)
        dataset_test = BiomarkerDataset(args.test_csv, test_transform, args.biomarker_mean, args.biomarker_std)
    else:
        dataset_test = build_dataset(split='test', args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if not args.eval:
        if getattr(args, 'balanced_sampler', False):
            assert num_tasks == 1, 'balanced_sampler only supports single-process (num_tasks==1)'
            ages = dataset_train.df['biomarker_value'].astype(float).values
            bw = int(getattr(args, 'balanced_bin_width', 5))
            min_count = int(getattr(args, 'balanced_min_count', 50))
            bin_idx = (ages // bw).astype(int)
            bin_idx = bin_idx - bin_idx.min()
            bin_counts = np.bincount(bin_idx)
            bin_counts_clipped = np.maximum(bin_counts, min_count)
            weights = 1.0 / bin_counts_clipped[bin_idx]
            weights = weights / weights.mean()
            sampler_train = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(dataset_train),
                replacement=True,
            )
            print('Using WeightedRandomSampler: bin_width=%d min_count=%d n_bins=%d raw_min=%d raw_max=%d' %
                  (bw, min_count, len(bin_counts), int(bin_counts.min()), int(bin_counts.max())))
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print('Sampler_train = %s' % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    if global_rank == 0 and not args.eval:
        os.makedirs(exp_dir, exist_ok=True)
        # Save effective config (YAML defaults + CLI overrides)
        with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
        log_writer = SummaryWriter(log_dir=exp_dir)
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    criterion = torch.nn.MSELoss()

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    no_weight_decay = model_without_ddp.no_weight_decay()
    param_groups = lrd.param_groups_lrd(
        model_without_ddp, weight_decay=args.weight_decay,
        no_weight_decay_list=no_weight_decay,
        layer_decay=args.layer_decay
    )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Determine test mode label (used for output filenames)
    if args.test_csv:
        if args.test_name:
            test_mode = f'test_{args.test_name}'
        else:
            # Auto-derive from test_csv path: splits/ukb/test.csv -> test_ukb
            test_name = os.path.basename(os.path.dirname(args.test_csv))
            test_mode = f'test_{test_name}' if test_name else 'test'
    else:
        test_mode = 'test'

    if args.eval:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        evaluate(data_loader_test, model, device, args, epoch=0, mode=test_mode, log_writer=log_writer)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = float('inf')
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val', log_writer=log_writer)
        if max_score > val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.save_model:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, mode='best')
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(exp_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Test with the best model after training
    best_ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model.to(device)
        print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
        evaluate(data_loader_test, model, device, args, checkpoint['epoch'], mode=test_mode, log_writer=None)


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(os.path.join(args.output_dir, args.exp_name)).mkdir(parents=True, exist_ok=True)
    main(args)
