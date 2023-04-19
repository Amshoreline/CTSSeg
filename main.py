import os
import argparse
import random
import time
import yaml
import json
from addict import Dict
import numpy as np
import torch
from torch.backends import cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import train_eval
import models


class Logger:

    def __init__(self, ):
        self.level = 'Info'

    def set_level(self, level):
        self.level = level

    def info(self, msg):
        if self.level == 'None':
            return
        elif self.level == 'Info':
            print('[' + time.asctime(time.localtime(time.time())) + ']', msg)


class FakeRecorder:

    def add_scalar(*args, **kwargs):
        pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dist_init():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    host_ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    init_method = 'tcp://{}:{}'.format(host_ip, port)
    print('dist.init_process_group', init_method, world_size, rank)
    dist.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    print('rank is {}, local_rank is {}, world_size is {}, host ip is {}'.format(rank, local_rank, world_size, host_ip))
    return local_rank


def merge_configs(configs, base_configs):
    for key in base_configs:
        if not key in configs:
            configs[key] = base_configs[key]
        elif type(configs[key]) is dict:
            merge_configs(configs[key], base_configs[key])


def build_configs(config_file, loaded_config_files):
    loaded_config_files.append(config_file)
    with open(config_file, 'r') as reader:
        configs = yaml.load(reader, Loader=yaml.Loader)
    for base_config_file in configs['base']:
        base_config_file = os.getcwd() + '/configs/' + base_config_file
        if base_config_file in loaded_config_files:
            continue
        base_configs = build_configs(base_config_file, loaded_config_files)
        merge_configs(configs, base_configs)
    return configs


def clear_configs(configs):
    keys = list(configs.keys())
    for key in keys:
        if type(configs[key]) is dict:
            configs[key] = clear_configs(configs[key])
        elif configs[key] == 'None':
            print('Clear config', key)
            configs.pop(key)
    return configs


def main():
    # Get parser
    parser = argparse.ArgumentParser(description='zcb 3d segmentation')
    parser.add_argument('--config_file', default='configs/base.yaml', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--local_rank', type=int)
    # Get configs
    args = parser.parse_args()
    out_dir = args.config_file.replace('configs', 'output')
    out_dir = out_dir.replace('.yaml', '')
    tag = args.config_file.split('/')[-1].split('.')[0]
    configs = Dict(clear_configs(build_configs(args.config_file, [])))
    # Basic configuration
    if args.local_rank is not None:
        configs.local_rank = dist_init()
        configs.rank = dist.get_rank()
        configs.world_size = dist.get_world_size()
    else:
        configs.local_rank = 0
        configs.rank = 0
        configs.world_size = 1
    print(f'Seed {configs.seed}')
    seed_all(configs.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # Get logger and logger_all
    logger = Logger()
    logger_all = Logger()
    # Set loggers
    if configs.rank == 0:
        logger.set_level('Info')
        record_dir = 'summaries'
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        record_path = record_dir + '/' + tag
        if args.test:
            recorder = FakeRecorder()
        else:
            recorder = SummaryWriter(log_dir=record_path)
    else:
        logger.set_level('None')
        recorder = FakeRecorder()
    logger_all.set_level('Info')
    logger.info(f'Recorder is {recorder}')
    # Make saving directories
    configs.out_dir = out_dir
    configs.ckpt_dir = configs.out_dir + '/checkpoints'
    configs.infer_dir = configs.out_dir + '/infer_results'
    configs.metric_dir = configs.out_dir + '/metric'
    configs.tag = tag
    if configs.rank == 0:
        os.makedirs(configs.ckpt_dir, exist_ok=True)
        os.makedirs(configs.infer_dir, exist_ok=True)
        os.makedirs(configs.metric_dir, exist_ok=True)
    logger.info(f'configs\n{json.dumps(configs, indent=2, ensure_ascii=False)}')
    assert torch.cuda.is_available()
    configs.device = torch.device('cuda')
    logger.info(f'Device is {configs.device}')
    # Get model
    logger.info(f'creating model: {configs.model.kind}')
    model = models.__dict__[configs.model.kind](**configs.model.kwargs)
    # Synchronize batch norm layers
    assert type(configs.model.sync_bn) is bool
    if configs.world_size > 1 and configs.model.sync_bn:
        logger.info('Use SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    logger.info(f'Use model:\n{model}')
    # Get optimizer
    optimizer = torch.optim.__dict__[configs.trainer.optimizer.kind](
        model.parameters(), **configs.trainer.optimizer.kwargs
    )
    logger.info(f'optimizer:\n{optimizer}')
    # Distribute model
    if configs.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[configs.local_rank],
            find_unused_parameters=True
        )
    # Epoch setting
    configs.start_epoch = 0
    # Resume model, optimizer
    if configs.model.resume and configs.rank == 0:
        logger.info('Resuming model')
        # logger.info(f'Before {model.state_dict()}')
        resume_path = configs.model.resume
        if not os.path.isfile(resume_path):
            raise Exception(f'Not found resume model: {resume_path}')
        checkpoint = torch.load(resume_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # state_dict = dict([(key.replace('module.', ''), value) for key, value in state_dict.items()])
        if configs.world_size > 1:
            state_dict = dict([('module.' + key, value) for key, value in state_dict.items()])
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info('WARNING: start_epoch is not resumed')
        # if 'epoch' in checkpoint:
        #     configs.start_epoch = checkpoint['epoch'] + 1
        #     logger.info('Resume epoch')
        logger.info('WARNING: Optimizer is not resumed')
        # if 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     logger.info('Resume optimizer')
        logger.info(
            f'Finish resuming from {configs.model.resume}\n\n'
            f'Missing keys {missing_keys}\n\nUnexpected keys {unexpected_keys}'
        )
        # logger.info(f'After {model.state_dict()}')
        del checkpoint
    if configs.world_size > 1:
        dist.barrier()
    # Mixed precision
    if configs.use_fp16:
        scaler = GradScaler()
        logger.info('Use mixed precision')
    else:
        scaler = None
    # Jump to specific task
    train_eval.main(
        configs=configs, is_test=args.test, model=model, optimizer=optimizer,
        logger=logger, logger_all=logger_all, recorder=recorder, scaler=scaler
    )


if __name__ == '__main__':
    main()
