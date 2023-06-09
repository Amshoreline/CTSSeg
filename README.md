# CTSSeg

## How to use
1. Prepare text file by:

```
{image_path_0},{label_path_0},{mask_path_0}\n
{image_path_1},{label_path_1},{mask_path_1}\n
...
```

2. Specify the directory and filename of the text file in the config file by:

```
dataset:
  txt_dir: {the directory}
  train_txts: [
    [{resample_times_0}, {train_text_filename_0}],
    [{resample_times_1}, {train_text_filename_1}],
    ...
  ]
  eval_txts: [
    [1, {test_text_filename_0}],
    [1, {test_text_filename_1}],
    ...
  ]
```
You can refer to configs/base.yaml and configs/thigh_full_label.yaml when writing your own config.

3. python main.py --config_file {config_file_path}

## For reproduction on the Atrial Segmentation Challenge dataset

The train/test split is origin from [SASSNet](https://github.com/kleinzcy/SASSnet)

1. Train with labeled data only for 25,000 iterations.

2. Train with labeled data and unlabeled data simultaneously from the pretrained model in step 1.
