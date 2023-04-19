import numpy as np

try:
    from .vanilla_dataset import VanillaDataset
    from .augmentations import get_DA_func
except:
    from vanilla_dataset import VanillaDataset
    from augmentations import get_DA_func


class SSupDataset(VanillaDataset):

    def __init__(self, use_strong_aug=False, **kwargs):
        super().__init__(**kwargs)
        if self.is_train:
            assert use_strong_aug in [True, False]
            self.use_strong_aug = use_strong_aug
            if self.threeD:
                patch_size = self.out_shape
            else:
                patch_size = (self.depth_2d, *self.out_shape)
            spatial_trans, voxel_trans, strong_voxel_trans = get_DA_func(
                patch_size, self.data_aug_params, self.is_train, indiv_spatial_trans=True
            )
            self.spatial_trans = spatial_trans
            self.voxel_trans = voxel_trans
            self.strong_voxel_trans = strong_voxel_trans
            print('Transforms are', self.spatial_trans, '\n', self.voxel_trans)
    
    def __getitem__(self, index):
        image, label, name = self._read_image_label(index)
        sample = {
            'data': image[None],
            'seg': label[None],
        }
        res = {
            'indexs': index,
            'names': name,
        }
        if self.is_train:
            sample_1 = self.spatial_trans(**sample)
            sample_2 = {'data': sample_1['data'].copy(), 'seg': sample_1['seg'].copy()}
            #
            if self.use_strong_aug:
                aug_sample_1 = self.strong_voxel_trans(**sample_1)
            else:
                aug_sample_1 = self.voxel_trans(**sample_1)
            image_1, label_1 = aug_sample_1['data'], aug_sample_1['seg']
            aug_sample_2 = self.voxel_trans(**sample_2)
            image_2, label_2 = aug_sample_2['data'], aug_sample_2['seg']
            assert image_1.shape[0] == 1 and label_1.shape[0] == 1
            res['images_1'] = image_1[0]
            res['images_2'] = image_2[0]
            res['labels'] = label_1[0, 0]
            res['masks'] = label_1[0, 1][None]
        else:
            aug_sample = self.transforms(**sample)
            image, label = aug_sample['data'], aug_sample['seg']
            assert image.shape[0] == 1 and label.shape[0] == 1
            res['images'] = image[0]
            res['labels'] = label[0, 0]
            res['masks'] = label[0, 1][None]
        return res
