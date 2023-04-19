import os
import random
import itertools
import numpy as np
import SimpleITK as sitk
from skimage import measure


def resample_image(
        itk_image, out_spacing=None, size=None, is_label=False
    ):
    image = sitk.GetArrayFromImage(itk_image)
    min_value = float(image.min())
    itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    if size is None:
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
        ]
    else:
        assert out_spacing is None
        out_size = size
        out_spacing = [
            original_size[0] * original_spacing[0] / out_size[0],
            original_size[1] * original_spacing[1] / out_size[1],
            original_size[2] * original_spacing[2] / out_size[2],
        ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    assert itk_image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    # resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetDefaultPixelValue(min_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(itk_image)  # Not an inplace operation


def crop_pad(image, center, tsize):
    '''
    Param:
        image: np.array
        center = (cz, cy, cx)
            cx == -1: margin crop in left
            cx == -2: margin crop in right
            cx == -3: center
            cx == -4: no crop
    '''
    td, th, tw = tsize
    sd, sh, sw = image.shape
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
    if cz == -3:
        cz = sd // 2
    if cy == -3:
        cy = sh // 2
    if cx == -3:
        cx = sw // 2
    # Calculate parameters
    if cz >= 0:
        pad_lz = max(0, (td // 2) - cz)
        pad_rz = max(0, td - (td // 2) - (sd - cz))
        crop_lz = cz + pad_lz - (td // 2)
        crop_rz = td + crop_lz
    elif cz == -1:
        pad_lz = max(0, td - sd)
        pad_rz = 0
        crop_lz = sd + pad_lz - td
        crop_rz = sd + pad_lz
    elif cz == -2:
        pad_lz = 0
        pad_rz = max(0, td - sd)
        crop_lz = 0
        crop_rz = td
    elif cz == -4:
        pad_lz, pad_rz = 0, 0
        crop_lz, crop_rz = 0, sd
    if cy >= 0:
        pad_ly = max(0, (th // 2) - cy)
        pad_ry = max(0, th - (th // 2) - (sh - cy))
        crop_ly = cy + pad_ly - (th // 2)
        crop_ry = th + crop_ly
    elif cy == -1:
        pad_ly = max(0, th - sh)
        pad_ry = 0
        crop_ly = sh + pad_ly - th
        crop_ry = sh + pad_ly
    elif cy == -2:
        pad_ly = 0
        pad_ry = max(0, th - sh)
        crop_ly = 0
        crop_ry = th
    elif cy == -4:
        pad_ly, pad_ry = 0, 0
        crop_ly, crop_ry = 0, sh
    if cx >= 0:
        pad_lx = max(0, (tw // 2) - cx)
        pad_rx = max(0, tw - (tw // 2) - (sw - cx))
        crop_lx = cx + pad_lx - (tw // 2)
        crop_rx = tw + crop_lx
    elif cx == -1:
        pad_lx = max(0, tw - sw)
        pad_rx = 0
        crop_lx = sw + pad_lx - tw
        crop_rx = sw + pad_lx
    elif cx == -2:
        pad_lx = 0
        pad_rx = max(0, tw - sw)
        crop_lx = 0
        crop_rx = tw
    elif cx == -4:
        pad_lx, pad_rx = 0, 0
        crop_lx, crop_rx = 0, sw
    # Pad and crop
    # print(pad_lz, pad_rz, crop_lz, crop_rz)
    # print(pad_ly, pad_ry, crop_ly, crop_ry)
    # print(pad_lx, pad_rx, crop_lx, crop_rx)
    image = np.pad(
        image, ((pad_lz, pad_rz), (pad_ly, pad_ry), (pad_lx, pad_rx)),
        mode='constant', constant_values=np.min(image)
    )[crop_lz : crop_rz, crop_ly : crop_ry, crop_lx : crop_rx]
    return image