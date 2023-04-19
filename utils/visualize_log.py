import os
import numpy as np
import matplotlib.pyplot as plt


def main(log_file):
    print(log_file)
    filename = log_file.split('/')[-1]
    with open(log_file, 'r') as reader:
        lines = reader.readlines()
    key2epoch2loss = {}
    for line in lines:
        if 'Epoch' in line and 'LR' in line:
            epoch = eval(line.split('Epoch')[1].split('/')[0].strip())
            loss_dict = eval('{' + line.split('{')[1].strip())
            for key in loss_dict:
                if not key in key2epoch2loss:
                    key2epoch2loss[key] = {}
                if not epoch in key2epoch2loss[key]:
                    key2epoch2loss[key][epoch] = []
                key2epoch2loss[key][epoch].append(loss_dict[key])
    plt.figure(figsize=(10, 5))
    for i, key in enumerate(key2epoch2loss):
        plt.subplot(f'1{len(key2epoch2loss)}{i}')
        plt.title(filename + '_' + key)
        epoch2loss = key2epoch2loss[key]
        losses = [np.mean(np.array(epoch2loss[epoch]).astype(np.float)) for epoch in epoch2loss]
        # plt.xlim(0, 500)
        plt.ylim(0, 1)
        plt.plot(range(len(losses)), losses)
    plt.savefig(f'../viz_res/{filename}.jpg')
    plt.close('all')


if __name__ == '__main__':
    base_dir = '../log'
    filenames = os.listdir(base_dir)
    for filename in filenames:
        if not '.log' in filename or '_test' in filename:
            continue
        main(f'{base_dir}/{filename}')
