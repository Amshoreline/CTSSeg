import os
import random


def main():
    # User defined arguments
    # cfg = 'teeth_unet_fold_1_crop.yaml'
    # cfg = 'teeth_unet_fold_1_crop.yaml'
    # cfg = 'thigh_fl_contra_teacher_full_ssup.yaml'
    # cfg = 'thigh_fl_exclude_unlabeled_slices_test.yaml'
    # cfg = 'myothigh_range04_full_ssup.yaml'
    cfg = 'myothigh_range04_contra_teacher_smooth.yaml'
    gpu_id = '2'
    num_gpus = len(gpu_id.split(','))
    # Prepare command
    tag = cfg.replace('.yaml', '')
    if cfg.endswith('test.yaml'):
        test = '--test'
    else:
        test = ''
    if not os.path.exists('log'):
        os.mkdir('log')
    if num_gpus == 1:
        module = ' -u '
    else:
        port = random.randint(20000, 30000)
        module = (
            '-m torch.distributed.launch '
            f'--nproc_per_node={num_gpus} --master_port={port} '
        )
    command = (
        'now=$(date +"%Y%m%d_%H%M%S")\n'
        f'CUDA_VISIBLE_DEVICES={gpu_id} python '
        f' {module} '
        f'main.py --config_file configs/{cfg} {test} '
        f'2>&1 | tee log/{cfg[: -5]}.log-$now'
    )
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
