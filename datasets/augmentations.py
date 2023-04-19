import numpy as np
from copy import deepcopy

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class AbstractTransform(object):
    
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


default_3D_augmentation_params = {
    'do_elastic': False,
    'elastic_deform_alpha': (0., 900.),
    'elastic_deform_sigma': (9., 13.),
    'p_eldef': 0.2,

    'do_scaling': True,
    'scale_range': (0.7, 1.4),
    'independent_scale_factor_for_each_axis': False,
    'p_independent_scale_per_axis': 1,
    'p_scale': 0.2,

    'do_rotation': True,
    'rotation_x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_p_per_axis': 1,
    'p_rot': 0.2,

    'random_crop': False,

    'do_gamma': True,
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,

    'do_mirror': False,
    'mirror_axes': (0, 1, 2),

    'dummy_2D': False,
    'border_mode_data': 'constant',

    'do_additive_brightness': False,
    'additive_brightness_p_per_sample': 0.15,
    'additive_brightness_p_per_channel': 0.5,
    'additive_brightness_mu': 0.0,
    'additive_brightness_sigma': 0.1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)
default_2D_augmentation_params['elastic_deform_alpha'] = (0., 200.)
default_2D_augmentation_params['elastic_deform_sigma'] = (9., 13.)
default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
default_2D_augmentation_params['dummy_2D'] = False


def get_voxel_trans(ignore_axes, params):
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0., 0.1), p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    if params.get('do_additive_brightness'):
        tr_transforms.append(
            BrightnessTransform(
                params.get('additive_brightness_mu'),
                params.get('additive_brightness_sigma'),
                True,
                p_per_sample=params.get('additive_brightness_p_per_sample'),
                p_per_channel=params.get('additive_brightness_p_per_channel')
            )
        )
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
            order_downsample=0, order_upsample=3, p_per_sample=0.25,
            ignore_axes=ignore_axes
        )
    )
    tr_transforms.append(
        GammaTransform(
            params.get('gamma_range'), True, True, retain_stats=params.get('gamma_retain_stats'),
            p_per_sample=0.1
        )
    )  # inverted gamma
    if params.get('do_gamma'):
        tr_transforms.append(
            GammaTransform(
                params.get('gamma_range'), False, True, retain_stats=params.get('gamma_retain_stats'),
                p_per_sample=params['p_gamma']
            )
        )

    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    return tr_transforms


def get_voxel_trans_wo_noise(*args, **kwargs):
    trans = get_voxel_trans(*args, **kwargs)
    return trans[1 :]


def get_strong_voxel_trans(ignore_axes):
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0., 0.2), p_per_sample=0.5))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.5, p_per_channel=1.))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2.0), p_per_sample=0.5))
    tr_transforms.append(BrightnessTransform(mu=0.0, sigma=0.1, per_channel=True, p_per_sample=0.5, p_per_channel=1.))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.5))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1), per_channel=True, p_per_channel=1.,
            order_downsample=0, order_upsample=3, p_per_sample=0.5,
            ignore_axes=ignore_axes
        )
    )
    tr_transforms.append(
        GammaTransform(
            (0.7, 1.5), True, True, retain_stats=True,
            p_per_sample=0.1
        )
    )  # inverted gamma
    tr_transforms.append(
        GammaTransform(
            (0.7, 1.5), False, True, retain_stats=True,
            p_per_sample=0.3
        )
    )
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    return tr_transforms


def get_DA_func(patch_size, params, is_train, indiv_spatial_trans=False, use_gaussian_noise=True):
    '''
    Params:
        patch_size: (d, h, w) or (h, w)
        params: dict
        is_train: True or False
        indiv_spatial_trans: True or False, whether to return spatial_trans and voxel_trans separately
    '''
    if is_train:
        tr_transforms = []
        if params.get('dummy_2D'):
            print('Use dummy_2D augmentation')
            ignore_axes = (0, )
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1 :]
        else:
            ignore_axes = None
            patch_size_spatial = patch_size
        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=None,
                p_el_per_sample=params.get('p_eldef'),
                do_elastic_deform=params.get('do_elastic'), alpha=params.get('elastic_deform_alpha'), sigma=params.get('elastic_deform_sigma'),
                p_rot_per_sample=params.get('p_rot'),
                do_rotation=params.get('do_rotation'), angle_x=params.get('rotation_x'), angle_y=params.get('rotation_y'), angle_z=params.get('rotation_z'), p_rot_per_axis=params.get('rotation_p_per_axis'),
                p_scale_per_sample=params.get('p_scale'),
                do_scale=params.get('do_scaling'), scale=params.get('scale_range'), independent_scale_for_each_axis=params.get('independent_scale_factor_for_each_axis'),
                border_mode_data=params.get('border_mode_data'), border_cval_data=0, order_data=3,
                border_mode_seg='constant', border_cval_seg=-1, order_seg=1,
                random_crop=params.get('random_crop'),      
            )
        )
        if params.get('dummy_2D'):
            tr_transforms.append(Convert2DTo3DTransform())
        if params.get('do_mirror'):
            tr_transforms.append(MirrorTransform(params.get('mirror_axes')))
        if indiv_spatial_trans:
            spatial_trans = Compose(tr_transforms)
            if use_gaussian_noise:
                voxel_trans = Compose(get_voxel_trans(ignore_axes, params))
            else:
                voxel_trans = Compose(get_voxel_trans_wo_noise(ignore_axes, params))
            strong_voxel_trans = Compose(get_strong_voxel_trans(ignore_axes))
            return spatial_trans, voxel_trans, strong_voxel_trans
        else:
            if use_gaussian_noise:
                tr_transforms.extend(get_voxel_trans(ignore_axes, params))
            else:
                tr_transforms.extend(get_voxel_trans_wo_noise(ignore_axes, params))
            tr_transforms = Compose(tr_transforms)
            return tr_transforms
    else:
        val_transforms = []
        val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms



