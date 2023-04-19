import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
try:
    from multiprocessing import shared_memory
except:
    print('Shared memory is unavailable in this python version')

try:
    from .augmentations import default_2D_augmentation_params, default_3D_augmentation_params, get_DA_func
except:
    from augmentations import default_2D_augmentation_params, default_3D_augmentation_params, get_DA_func


class MemoryReader:

    def __init__(self, meta_file):
        meta_info = np.load(meta_file, allow_pickle=True).tolist()
        self.images_shm = shared_memory.SharedMemory(name=meta_info['images_shm_name'])
        self.images = np.ndarray(meta_info['images_nelmts'], dtype=meta_info['images_dtype'], buffer=self.images_shm.buf)
        self.labels_shm = shared_memory.SharedMemory(name=meta_info['labels_shm_name'])
        self.labels = np.ndarray(meta_info['labels_nelmts'], dtype=meta_info['labels_dtype'], buffer=self.labels_shm.buf)
        #
        image_paths = meta_info['image_paths']
        self.path2index = dict([(image_path, index) for index, image_path in enumerate(image_paths)])
        self.mem_indexs = meta_info['indexs']  # indexs.shape = (#images + 1, )
        self.shapes = meta_info['shapes']  # shapes.shape = (#images, 3)
        del meta_info

    def __getitem__(self, index):
        left_mem_index = self.mem_indexs[index]
        right_mem_index = self.mem_indexs[index + 1]
        shape = self.shapes[index]
        image = self.images[left_mem_index : right_mem_index].copy().reshape(shape)
        label = self.labels[left_mem_index : right_mem_index].copy().reshape(shape)
        return image, label

    def get_data_by_path(self, path):
        return self.__getitem__(self.path2index[path])

    def close(self, ):
        self.images_shm.close()
        self.labels_shm.close()


def normalize(image, data_type):
    if data_type == 'CT':
        image = (image + 1000.) / 4000.
    elif data_type == 'MRI':
        min_value = np.percentile(image, 1)
        max_value = np.percentile(image, 99)
        image = (np.clip(image, min_value, max_value) - min_value) / (max_value - min_value)
    else:
        image = None
    assert np.min(image) >= 0. and np.max(image) <= 1.
    return image


def pad_if_need(image, min_size):
    '''
    image.shape = (c, d, h, w)
    min_size = (d', h', w') or (h', w')
    '''
    *_, h, w = image.shape
    *_, min_h, min_w = min_size
    left_pad_h = max((min_h - h + 1) // 2, 0)
    right_pad_h = max(min_h - h - left_pad_h, 0)
    left_pad_w = max((min_w - w + 1) // 2, 0)
    right_pad_w = max(min_w - w - left_pad_w, 0)
    if len(min_size) == 3:  # 3D
        d = image.shape[1]
        min_d = min_size[0]
        left_pad_d = max((min_d - d + 1) // 2, 0)
        right_pad_d = max(min_d - d - left_pad_d, 0)
    else:
        left_pad_d = 0
        right_pad_d = 0
    pad_params = [
        (0, 0),
        (left_pad_d, right_pad_d),
        (left_pad_h, right_pad_h),
        (left_pad_w, right_pad_w)
    ]
    #
    image = np.pad(
        image, pad_params,
        mode='constant', constant_values=np.min(image),
    )
    return image


def get_mask_from_label(label, mask_type):
    '''
    Params:
        label.shape = (d, h, w)
    Return:
        mask.shape = (d, h, w)
    '''
    if mask_type == 'all':
        mask = np.ones_like(label, dtype=label.dtype)
    elif mask_type == 'valid_slice':
        slice_mask = (np.sum(label, axis=(1, 2)) > 0)
        mask = np.zeros_like(label, dtype=label.dtype)
        mask[slice_mask] = 1
    return mask


class VanillaDataset(torch.utils.data.Dataset):

    def __init__(
            self, txt_dir, txt_files, data_type='MRI', phase='train',
            dataset_size=-1, data_in_memory=True, shm_meta_file=None,
            out_shape=(64, 288, 288), dummy_2D_aug=True,
            mask_type='all', force_fg=False, depth_2d=1,
        ):
        '''
        Parmas:
            txt_dir: directory of the text files
            txt_files: [[weight(int), filename], ...]
            data_type: 'MRI' or 'CT'
            phase: 'train' or 'test'
            dataset_size: -1, 1, 2, 3, ...
            data_in_memory: True or False, whether to keep data in the memory
            shm_meta_file: None or str, path of meta file of shared memory
            out_shape: (d, h, w) or (h, w)
            dummy_2D_aug: whether to use dummy 2D augmentation,
                valid for 3d out_shape only
            mask_type: 'all' or 'valid_slice', 'valid_slice' will indicate which slice is labeled
            force_fg: True or False, whether to force to use slices with foreground only
                valid for 2d out_shape only
            depth_2d: how many 2d slices are prefetched before collate_fn
                valid for 2d out_shape only
        '''
        # Get data paths
        data_paths = []
        for times, filename in txt_files:
            with open(txt_dir + '/' + filename) as reader:
                data_paths.extend(times * reader.read().strip().split('\n'))
        if dataset_size < 0:
            dataset_size = len(data_paths)
        self.dataset_size = dataset_size
        self.data_paths = [item.split(',') for item in data_paths]
        #
        assert data_type in ['MRI', 'CT']
        self.data_type = data_type
        # Attributes configuration
        self.phase = phase
        #
        if shm_meta_file is None:
            self.use_shm = False
            assert data_in_memory in [True, False]
            self.data_in_memory = data_in_memory
            if data_in_memory:
                print('Load data into memory at runtime')
                self.path2data = {}
            else:
                print('No extra operation between dataset and memory')
        else:
            print('Use shared memory')
            self.use_shm = True
            self.shm_reader = MemoryReader(shm_meta_file)
        #
        self.out_shape = out_shape
        self.threeD = (len(out_shape) == 3)
        self.is_train = (phase == 'train')
        #
        if self.threeD:
            assert dummy_2D_aug in [True, False]
            self.dummy_2D_aug = dummy_2D_aug
            if dummy_2D_aug:
                self.data_aug_params = default_2D_augmentation_params
            else:
                self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['dummy_2D'] = True
            transforms = get_DA_func(self.out_shape, self.data_aug_params, self.is_train)
        else:
            self.data_aug_params = default_2D_augmentation_params
            self.data_aug_params['dummy_2D'] = True
            #
            assert force_fg in [True, False]
            self.force_fg = force_fg
            self.depth_2d = depth_2d
            transforms = get_DA_func((depth_2d, *self.out_shape), self.data_aug_params, self.is_train)
        self.transforms = transforms
        print('Transforms are', self.transforms)
        #
        assert mask_type in ['all', 'valid_slice']
        self.mask_type = mask_type

    def _read_image_label_once(self, index):
        '''
        Desc:
            Read image and label
        Return:
            image: (c, d, h, w)
            label: (2, d, h, w), 2->class & mask
            name: str
        '''
        image_path, label_path, *_ = self.data_paths[index]
        name = label_path.split('/')[-1].replace('.nii.gz', '')
        if self.use_shm:
            image, label = self.shm_reader.get_data_by_path(image_path)
        elif self.data_in_memory and image_path in self.path2data:
            image, label = self.path2data[image_path]
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            if self.data_in_memory:
                self.path2data[image_path] = (image, label)
        image = normalize(image, self.data_type)
        image = image[None]
        image = image.astype(np.float64)
        label = label.astype(np.uint8)
        if self.mask_type == 'valid_slice' and len(self.data_paths[index]) == 3:
            # TODO: this op is tooooooo ugly, I will improve it in the future
            mask = sitk.GetArrayFromImage(sitk.ReadImage(self.data_paths[index][2]))
            mask = mask.astype(np.uint8)
        else:
            mask = get_mask_from_label(label, self.mask_type)
        label = np.concatenate([label[None], mask[None]], axis=0)
        # Random crop in z-axis
        if self.is_train and self.threeD and self.dummy_2D_aug:
            max_crop_offset = image.shape[1] - self.out_shape[0]
            offset_z = np.random.randint(0, max_crop_offset + 1)
            image = image[:, offset_z : offset_z + self.out_shape[0]]
            label = label[:, offset_z : offset_z + self.out_shape[0]]
        elif self.is_train and not self.threeD:
            if self.force_fg:
                valid_offsets = np.where(np.sum(label[0], axis=(1, 2)) > 0)[0]
            else:
                valid_offsets = np.arange(image.shape[1])
            offset_z = np.random.choice(valid_offsets)
            image = image[:, offset_z : offset_z + 1]
            label = label[:, offset_z : offset_z + 1]
        return image, label, name

    def _read_image_label(self, index):
        if not self.threeD and self.is_train:
            images = []
            labels = []
            name = None
            for offset in range(self.depth_2d):
                real_index = (index * self.depth_2d + offset) % self.dataset_size
                image, label, name = self._read_image_label_once(real_index)
                images.append(image)
                labels.append(label)
            num_dims = len(self.out_shape)
            shapes = np.array([image.shape[-num_dims :] for image in images] + [self.out_shape])
            max_shape = np.max(shapes, axis=0)
            images = [pad_if_need(image, max_shape) for image in images]
            labels = [pad_if_need(label, max_shape) for label in labels]
            images = np.concatenate(images, axis=1)
            labels = np.concatenate(labels, axis=1)
            return images, labels, name
        else:
            image, label, name = self._read_image_label_once(index)
            if self.is_train:
                image = pad_if_need(image, self.out_shape)
                label = pad_if_need(label, self.out_shape)
            return image, label, name

    def __len__(self):
        if not self.threeD and self.is_train:
            return (self.dataset_size + self.depth_2d) // self.depth_2d
        else:
            return self.dataset_size

    def __getitem__(self, index):
        image, label, name = self._read_image_label(index)
        sample = {
            'data': image[None],    # (1, c, d, h, w)
            'seg': label[None],     # (1, 2, d, h, w)
        }
        aug_sample = self.transforms(**sample)
        image, label = aug_sample['data'], aug_sample['seg']
        assert image.shape[0] == 1 and label.shape[0] == 1
        return {
            'images': image[0],
            'labels': label[0, 0],
            'masks': label[0, 1][None],
            'indexs': index,
            'names': name,
        }

    def __del__(self):
        if self.use_shm:
            self.shm_reader.close()
