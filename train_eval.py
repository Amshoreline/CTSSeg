import os
import time
import math
import random
from tqdm import tqdm, trange
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

import datasets
from metrics import Metric


def seed_worker(worker_id):
    # There is a bug (maybe) here
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def warmup_model_data(num_epochs, model, data_loader):
    for epoch in trange(num_epochs):
        data_loader.sampler.set_epoch(epoch)
        for b_ind, (images, gt_masks, indexs) in enumerate(data_loader):
            images = images.cuda()
            gt_masks = gt_masks.cuda(non_blocking=True)
            with torch.no_grad():
                model(images, gt_masks)


def run_epoch(
        phase, epoch, configs, data_loader, model, optimizer,
        lr_schedule, metric, logger, logger_all, recorder, scaler,
    ):
    end = time.time()
    total_scores = {}
    logger.info(f'Epoch {epoch} with {len(data_loader)} iterations')
    for b_ind, data_dict in enumerate(data_loader):
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(configs.device)
        if phase == 'train':
            # update learning rate
            iteration = epoch * len(data_loader) + b_ind
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
            # forward
            optimizer.zero_grad()
            if configs.use_fp16:
                with autocast():
                    loss_names, loss_items, total_loss = model(data_dict)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_names, loss_items, total_loss = model(data_dict)
                total_loss.backward()
                optimizer.step()
            if configs.world_size > 1:
                model.module.update()
            else:
                model.update()
            # record losses
            scores = {}
            for loss_name, loss_item in zip(loss_names, loss_items):
                scores[loss_name] = loss_item
        else:
            # forward
            with torch.no_grad():
                if configs.world_size > 1:
                    pred_coefs, pred_masks = model.module.infer(data_dict)
                else:
                    pred_coefs, pred_masks = model.infer(data_dict)
            # evaluate, we expect the batch size to be 1
            assert pred_coefs.shape[0] == 1
            pred_coef = pred_coefs.cpu().numpy()[0]
            pred_mask = pred_masks.cpu().numpy()[0].astype(np.uint8)
            gt_mask = data_dict['labels'].cpu().numpy()[0].astype(np.uint8)
            scores, map_info = metric(pred_mask, gt_mask)
            index = data_dict['indexs'].item()
            name = data_dict['names'][0]
            # save image, gt, pred
            # sitk.WriteImage(sitk.GetImageFromArray(images[0, 0].cpu().numpy()), f'{configs.infer_dir}/{index}_image.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(pred_mask), f'{configs.infer_dir}/{name}_pred.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(gt_mask), f'{configs.infer_dir}/{index}_gt.nii.gz')
        for key, value in scores.items():
            if not key in total_scores:
                total_scores[key] = []
            total_scores[key].append(value)
        if configs.device.type == 'cuda':
            memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        else:
            memory = 0.
        time_cost = time.time() - end
        logger_all.info(
            f'{configs.tag} {phase} Rank {configs.rank}/{configs.world_size} '
            f'Epoch {epoch}/{configs.max_epoch} '
            f'Batch {b_ind}/{len(data_loader)} '
            f'Time {time_cost: .3f} Mem {memory}MB '
            f'LR {optimizer.param_groups[0]["lr"]:.3e} '
            f'{scores}'
        )
    if phase == 'train':
        reduce_matrix = torch.zeros(configs.world_size, len(total_scores), len(data_loader)).cuda()
        keys = list(total_scores.keys())
        keys.sort()
        for key_ind, key in enumerate(keys):
            reduce_matrix[configs.rank, key_ind] = torch.tensor(total_scores[key])
        if configs.world_size > 1:
            dist.all_reduce(reduce_matrix)
        reduce_matrix = reduce_matrix.cpu().numpy()
        reduce_matrix = np.mean(reduce_matrix, axis=0)  # (#keys, #iters)
        for key, scores in zip(keys, reduce_matrix):
            recorder.add_scalar(f'{phase}/{key}(Epoch)', np.mean(scores), epoch)
    else:
        with open(f'{configs.metric_dir}/eval_score_rank_{configs.rank}', 'w') as writer:
            writer.write(str(total_scores))
        if configs.world_size > 1:
            dist.barrier()
        total_scores = {}
        # total_scores[-1] = []
        for rank in range(configs.world_size):
            with open(f'{configs.metric_dir}/eval_score_rank_{rank}', 'r') as reader:
                rank_scores = eval(reader.read())
            for key, value in rank_scores.items():
                if not key in total_scores:
                    total_scores[key] = value
                else:
                    total_scores[key].extend(value)
                # total_scores[-1].extend(value)
    total_scores = (
        sorted(
            [
                (key, np.mean(value))
                for key, value in total_scores.items()
            ],
            key=lambda x : x[0]
        )
    )
    total_scores.insert(0, (-1, np.mean([item[1] for item in total_scores])))
    total_scores = dict([(key, round(value, 4)) for key, value in total_scores])
    return total_scores


def main(configs, is_test, model, optimizer, logger, logger_all, recorder, scaler):
    # Get metric function
    metric = Metric(**configs.metric)
    # Build dataloader
    train_dataset = datasets.__dict__[configs.dataset.kind](
        phase='train', txt_dir=configs.dataset.txt_dir,
        txt_files=configs.dataset.train_txts,
        **configs.dataset.kwargs
    )
    eval_dataset = datasets.__dict__[configs.dataset.kind](
        phase='eval', txt_dir=configs.dataset.txt_dir,
        txt_files=configs.dataset.eval_txts,
        **configs.dataset.kwargs
    )
    if configs.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False
        )
    else:
        train_sampler = None
        eval_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=configs.dataset.train_batch_size,
        num_workers=configs.dataset.num_workers,
        pin_memory=False, drop_last=True,
        worker_init_fn=seed_worker
    )
    logger_all.info(f'Rank {configs.rank}/{configs.world_size} {len(train_loader)} train iters')
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=configs.dataset.eval_batch_size,
        num_workers=configs.dataset.num_workers,
        pin_memory=False, drop_last=False
    )
    logger_all.info(f'Rank {configs.rank}/{configs.world_size} {len(eval_loader)} eval iters')
    #
    if is_test:
        configs.max_epoch = configs.start_epoch + 1
    else:
        lr_configs = configs.trainer.lr_schedule
        warmup_lr_schedule = np.linspace(
            lr_configs.start_warmup, lr_configs.base_lr,
            len(train_loader) * lr_configs.warmup_epochs
        )
        iters = np.arange(len(train_loader) * lr_configs.cosine_epochs)
        cosine_lr_schedule = np.array(
            [
                lr_configs.final_lr + (
                    0.5 * (lr_configs.base_lr - lr_configs.final_lr)
                    * (1 + math.cos(math.pi * t / (len(train_loader) * lr_configs.cosine_epochs)))
                )
                for t in iters
            ]
        )
        lr_schedule = np.concatenate([warmup_lr_schedule] + [cosine_lr_schedule] * lr_configs.cosine_times)
        configs.max_epoch = lr_configs.warmup_epochs + lr_configs.cosine_epochs * lr_configs.cosine_times
    # if configs.rank == 0 and not is_test:
    #     plt.plot(range(len(lr_schedule)), lr_schedule)
    #     plt.xlabel('iterations')
    #     plt.ylabel('learning rate')
    #     plt.savefig(configs.metric_dir + '/lr.jpg')
    # if not is_test:
    #     for iteration, lr in enumerate(lr_schedule):
    #         recorder.add_scalar('LR', lr, iteration)
    if configs.world_size > 1:
        dist.barrier()
    if configs.start_epoch != 0 and not is_test:
        logger.info('Warmup model and data')
        # There is no BatchNorm here, just Dropout and InstanceNorm
        model.train()
        # warmup_model_data(configs.start_epoch, model, train_loader)
    for epoch in range(configs.start_epoch, configs.max_epoch):
        if configs.world_size > 1:
            dist.barrier()
        if not is_test:
            if configs.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            model.train()
            train_score = run_epoch(
                'train', epoch=epoch, configs=configs, data_loader=train_loader,
                model=model, optimizer=optimizer,
                lr_schedule=lr_schedule, metric=metric,
                logger=logger, logger_all=logger_all, recorder=recorder, scaler=scaler
            )
            logger_all.info(
                f'Train Rank {configs.rank}/{configs.world_size} Epoch {epoch} {train_score}'
            )
            if (
                (configs.rank == 0)
                and (
                    ((epoch + 1) % configs.trainer.save_freq == 0)
                    or (epoch == configs.max_epoch - 1)
                )
            ):
                if configs.world_size > 1:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model': configs.model.kind,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                ckpt_path = (
                    f'{configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth'
                )
                torch.save(checkpoint, ckpt_path)
                os.system(f'cp {configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth {configs.ckpt_dir}/latest.pth')
                logger.info(f'Save checkpoint in epoch {epoch}')
        if (
            ((epoch + 1) % configs.trainer.test_freq == 0)
            or (epoch == configs.max_epoch - 1)
        ):
            model.eval()
            eval_score = run_epoch(
                'eval', epoch=epoch, configs=configs, data_loader=eval_loader,
                model=model, optimizer=optimizer,
                lr_schedule=None, metric=metric,
                logger=logger, logger_all=logger_all, recorder=recorder, scaler=None
            )
            logger_all.info(
                f'Eval Rank {configs.rank}/{configs.world_size} Epoch {epoch} {eval_score}\n'
                f'| {" | ".join(np.array(list(eval_score.values()), dtype=str))} |'
            )
            for key, value in eval_score.items():
                recorder.add_scalar(f'Metric/{key}', value, epoch)
        # To avoid deadlock
        time.sleep(2.33)
