import numpy as np
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_minus_dice(outputs, targets, weight):
    '''
    Params:
        outputs: FloatTensor(bs, num_classes, d, h, w) or (bs, num_classes, h, w)
        targets: LongTensor (bs, d, h, w) or (bs, h, w)
        weight:  FloatTensor(num_classes, ) 
    Return:
        loss
    '''
    device = outputs.device
    bs, num_classes, *_ = outputs.size()
    outputs = outputs.softmax(dim=1).view(bs, num_classes, -1)
    targets = (
        targets.view(bs, 1, -1)
        == torch.arange(num_classes).view(1, -1, 1).float().to(device)
    )  # convert to one-hot encoding
    dice_tensor = (
        2 * torch.sum(outputs * targets, dim=(0, 2))
        / (torch.sum(outputs, dim=(0, 2)) + torch.sum(targets, dim=(0, 2)) + 1e-12)
    )
    return torch.sum((1 - dice_tensor) * weight)


def smooth_slice_loss(outputs, targets):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
    Return:
        loss
    From: http://guanbinli.com/papers/Semi-supervised%20Spatial%20Temporal%20Attention%20Network%20for%20Video%20Polyp%20Segmentation.pdf
    '''
    outputs = outputs.softmax(dim=1)
    loss_left = F.l1_loss(outputs[:, :, : -1], targets[:, :, 1 :])
    loss_center = F.l1_loss(outputs, targets)
    loss_right = F.l1_loss(outputs[:, :, 1 :], targets[:, :, : -1])
    return (loss_left + loss_center + loss_right) / 3.0


def neighbor_voxel_loss(outputs, targets):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
    Return:
        loss
    '''
    log_softmax_outputs = F.log_softmax(outputs, dim=1)  # (bs, num_classes, d, h, w)
    with torch.no_grad():
        max_targets = F.max_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
        avg_targets = F.avg_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
    max_ce_loss = torch.mean(torch.sum(-(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1] * max_targets), dim=1))
    avg_ce_loss = torch.mean(torch.sum(-(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1] * avg_targets), dim=1))
    return (max_ce_loss + avg_ce_loss) / 2


def get_gaussian(patch_size, sigma_scale=1. / 8):
    '''
    Parmas:
        patch_size: (d, h, w) or (h, w)
    Return:
        gaussian_importance_map.shape = (d, h, w) or (h, w)
    '''
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )
    return gaussian_importance_map


def slide_infer_3d(images, gaussian_map, model, num_classes):
    '''
    Parameters:
        images: FloatTensor(bs, c, d, h, w)
        gaussian_map: FloatTensor(1, 1, d', h', w')
        model: nn.Module
    '''
    *_, infer_d, infer_h, infer_w = gaussian_map.shape
    *_, origin_d, origin_h, origin_w = images.shape
    device = images.device
    # pad if need, TODO: change to padding in two sides
    pad_d = max(0, infer_d - origin_d)
    pad_h = max(0, infer_h - origin_h)
    pad_w = max(0, infer_w - origin_w)
    images = np.pad(images.cpu().numpy(), ((0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
    images = torch.tensor(images).to(device)
    #
    bs, c, d, h, w = images.shape
    #
    stride_d, stride_h, stride_w = infer_d // 2, infer_h // 2, infer_w // 2
    # print(f'infer with patch_size {(infer_d, infer_h, infer_w)} and stride {(stride_d, stride_h, stride_w)}')   
    pred_res = torch.zeros(bs, num_classes, d, h, w)
    for d_off in range(0, d - infer_d + stride_d, stride_d):
        if (d_off + infer_d) > d:
            d_off = d - infer_d
        for h_off in range(0, h - infer_h + stride_h, stride_h):
            if (h_off + infer_h) > h:
                h_off = h - infer_h
            for w_off in range(0, w - infer_w + stride_w, stride_w):
                if (w_off + infer_w) > w:
                    w_off = w - infer_w
                pred = model(images[..., d_off : d_off + infer_d, h_off : h_off + infer_h, w_off : w_off + infer_w])
                pred = pred.softmax(dim=1).cpu()
                pred_res[..., d_off : d_off + infer_d, h_off : h_off + infer_h, w_off : w_off + infer_w] += gaussian_map * pred
    return pred_res[..., : origin_d, : origin_h, : origin_w]


def slide_infer_2d(images, gaussian_map, model, num_classes):
    '''
    Parameters:
        images: FloatTensor(bs, c, d, h, w)
        gaussian_map: FloatTensor(1, 1, h', w')
        model: nn.Module
    '''
    *_, infer_h, infer_w = gaussian_map.shape
    *_, origin_d, origin_h, origin_w = images.shape
    device = images.device
    # pad if need, TODO: change to padding in two sides
    pad_h = max(0, infer_h - origin_h)
    pad_w = max(0, infer_w - origin_w)
    images = np.pad(images.cpu().numpy(), ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    images = torch.tensor(images).to(device)
    #
    bs, c, d, h, w = images.shape
    #
    stride_h, stride_w = infer_h // 2, infer_w // 2
    # print(f'infer with patch_size {(infer_d, infer_h, infer_w)} and stride {(stride_d, stride_h, stride_w)}')   
    pred_res = torch.zeros(bs, num_classes, d, h, w)
    for d_off in range(d):
        for h_off in range(0, h - infer_h + stride_h, stride_h):
            if (h_off + infer_h) > h:
                h_off = h - infer_h
            for w_off in range(0, w - infer_w + stride_w, stride_w):
                if (w_off + infer_w) > w:
                    w_off = w - infer_w
                pred = model(images[..., d_off, h_off : h_off + infer_h, w_off : w_off + infer_w])
                pred = pred.softmax(dim=1).cpu()
                pred_res[..., d_off, h_off : h_off + infer_h, w_off : w_off + infer_w] += gaussian_map * pred
    return pred_res[..., : origin_h, : origin_w]


def slide_infer(images, gaussian_map, model, num_classes):
    if len(gaussian_map.shape) == 5:
        return slide_infer_3d(images, gaussian_map, model, num_classes)
    else:
        return slide_infer_2d(images, gaussian_map, model, num_classes)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, global_step, alpha=0.95):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax)


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)


def none_loss(preds, targets):
    return torch.zeros(1).to(preds.device)


class VanillaModel(nn.Module):

    def __init__(self, num_classes, infer_size, dice_weight=None, ce_weight=None):
        super(VanillaModel, self).__init__()
        self.infer_size = infer_size
        self.threeD = (len(infer_size) == 3)
        self.gaussian_map = torch.tensor(get_gaussian(self.infer_size))[None, None]
        self.num_classes = num_classes
        self.unet_body = None
        #
        if dice_weight is None:
            dice_weight = [0.] + [1.] * (num_classes - 1)
        self.dice_weight = torch.tensor(dice_weight).float() / np.sum(dice_weight)
        if ce_weight is None:
            ce_weight = [1.] * num_classes
        self.ce_weight = torch.tensor(ce_weight).float()

    def forward(self, data_dict):
        batch_imgs = data_dict['images']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        batch_targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        batch_masks = data_dict['masks']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        if self.threeD:
            batch_preds = self.unet_body(batch_imgs) * batch_masks
        else:
            bs, c, depth_2d, h, w = batch_imgs.shape
            batch_imgs = batch_imgs.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
            batch_targets = batch_targets.view(bs * depth_2d, h, w)
            batch_masks = batch_masks.view(bs * depth_2d, 1, h, w)
            batch_preds = self.unet_body(batch_imgs) * batch_masks
        dice_loss = one_minus_dice(batch_preds, batch_targets, self.dice_weight.to(batch_imgs.device))
        ce_loss = F.cross_entropy(batch_preds, batch_targets.long(), self.ce_weight.to(batch_imgs.device))
        total_loss = dice_loss + ce_loss
        return ['DiceLoss', 'CELoss', 'Total'], [round(dice_loss.item(), 3), round(ce_loss.item(), 3), total_loss.item()], total_loss

    def infer(self, data_dict):
        output = slide_infer(data_dict['images'], self.gaussian_map, self.unet_body, self.num_classes)
        return torch.max(output, dim=1)

    def update(self, ):
        pass


class MeanTeacher(VanillaModel):
    
    def __init__(self, alpha, rampup_length, consist_loss_name, ssup_range, **kwargs):
        super().__init__(**kwargs)
        self.teacher_unet_body = None
        self.alpha = alpha
        self.rampup_length = rampup_length
        self.consist_criterion = {'mse': softmax_mse_loss, 'kl': softmax_kl_loss}[consist_loss_name]
        assert (type(ssup_range) is list) and (set(ssup_range).issubset({'label', 'unlabel'}))
        self.ssup_range = ssup_range
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        batch_imgs = data_dict['images_1']  # (bs, 1, d, h, w)
        batch_targets = data_dict['labels'] # (bs, d, h, w)
        batch_masks = data_dict['masks']    # (bs, 1, d, h, w)
        batch_preds = self.unet_body(batch_imgs)
        dice_loss = one_minus_dice(batch_preds * batch_masks, batch_targets, self.dice_weight.to(batch_imgs.device))
        ce_loss = F.cross_entropy(batch_preds * batch_masks, batch_targets.long(), self.ce_weight.to(batch_imgs.device))
        # Consistency loss
        batch_ano_imgs = data_dict['images_2']
        with torch.no_grad():
            batch_ano_preds = self.teacher_unet_body(batch_ano_imgs).detach()
        ssup_masks = torch.zeros_like(batch_masks).to(batch_masks.device)
        if 'label' in self.ssup_range:
            ssup_masks = ssup_masks + batch_masks
        if 'unlabel' in self.ssup_range:
            ssup_masks = ssup_masks + (1 - batch_masks)
        consist_loss = self.consist_criterion(batch_preds * ssup_masks, batch_ano_preds * ssup_masks)
        consist_weight = sigmoid_rampup(self.global_step, self.rampup_length)
        #
        total_loss = dice_loss + ce_loss + consist_weight * consist_loss
        return (
            ['DiceLoss', 'CELoss', 'W.Consist', 'Consist', 'Total'],
            [round(dice_loss.item(), 3), round(ce_loss.item(), 3), round(consist_weight, 3), round(consist_loss.item(), 3), total_loss.item()],
            total_loss
        )

    def update(self, ):
        with torch.no_grad():
            update_ema_variables(self.unet_body, self.teacher_unet_body, self.global_step, self.alpha)
        self.global_step += 1

    def parameters(self, ):
        return self.unet_body.parameters()


def get_ConvPred(in_channels, hidden_size, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_size, kernel_size=1),
        nn.BatchNorm3d(hidden_size),
        nn.ReLU(),
        nn.Conv3d(hidden_size, out_channels, kernel_size=1)
    )


class ContraTeacher(VanillaModel):

    def __init__(self, proj_size, alpha, rampup_length, consist_loss_name, ssup_range, pseudo_thres, hidden_size=-1, **kwargs):
        '''
        Parmas:
            proj_size: #channels in feature map
            alpha: momentum update
            rampup_length: momentum update
            consist_loss_name: 'mse' or 'kl', unavailable temporarily
            ssup_range: list, each element is 'label' or 'unlabel'
            pseudo_thres: threshold of pseudo labeling
        '''
        super().__init__(**kwargs)
        self.teacher_unet_body = None
        self.alpha = alpha
        self.rampup_length = rampup_length
        #
        if hidden_size == -1:
            hidden_size = proj_size
        self.conv_pred = get_ConvPred(proj_size, hidden_size, proj_size)
        # self.consist_criterion = {'mse': softmax_mse_loss, 'kl': softmax_kl_loss}[consist_loss_name]
        self.consist_criterion = {'mse': F.mse_loss, 'none': none_loss, 'l1': F.l1_loss}[consist_loss_name]
        #
        assert (type(ssup_range) is list) and (set(ssup_range).issubset({'label', 'unlabel'}))
        self.ssup_range = ssup_range
        self.pseudo_thres = pseudo_thres
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        batch_imgs = data_dict['images_1']  # (bs, 1, d, h, w)
        batch_targets = data_dict['labels'] # (bs, d, h, w)
        batch_masks = data_dict['masks']    # (bs, 1, d, h, w), mask for valid slices
        batch_feats, batch_preds = self.unet_body(batch_imgs, ret_last_feat=True)
        batch_pred_feats = self.conv_pred(batch_feats)  # (bs, proj_size, d, h, w)
        dice_loss = one_minus_dice(batch_preds * batch_masks, batch_targets, self.dice_weight.to(batch_imgs.device))
        ce_loss = F.cross_entropy(batch_preds * batch_masks, batch_targets.long(), self.ce_weight.to(batch_imgs.device))
        # Consistency loss & SSup loss
        batch_ano_imgs = data_dict['images_2']
        with torch.no_grad():
            batch_ano_feats, batch_ano_preds = self.teacher_unet_body(batch_ano_imgs, ret_last_feat=True)
            batch_ano_feats = batch_ano_feats.detach()
            batch_ano_preds = batch_ano_preds.detach().softmax(dim=1)
            pseudo_coefs, pseudo_targets = torch.max(batch_ano_preds, dim=1)  # (bs, D, H, W)
            ssup_masks = (pseudo_coefs > self.pseudo_thres).float()
            ssup_masks = ssup_masks[:, None]  # (bs, 1, D, H, W)
        ssup_upbound_masks = torch.zeros_like(batch_masks).to(batch_masks.device)
        if 'label' in self.ssup_range:
            ssup_upbound_masks = ssup_upbound_masks + batch_masks
        if 'unlabel' in self.ssup_range:
            ssup_upbound_masks = ssup_upbound_masks + (1 - batch_masks)
        ssup_masks = ssup_masks * ssup_upbound_masks
        ssup_ratio = round(torch.mean(ssup_masks).item(), 3)
        consist_loss = self.consist_criterion(batch_pred_feats, batch_ano_feats)
        if self.pseudo_thres == -1:
            ssup_dice_loss = torch.zeros(1).to(batch_imgs.device)
            # ssup_ce_loss = -torch.mean(torch.sum(batch_ano_preds * torch.log(batch_preds.softmax(dim=1) + 1e-12), dim=1))
            ssup_ce_loss = smooth_slice_loss(batch_preds * ssup_masks, batch_ano_preds)
        else:
            ssup_dice_loss = one_minus_dice(batch_preds * ssup_masks, pseudo_targets, self.dice_weight.to(batch_imgs.device))
            ssup_ce_loss = F.cross_entropy(batch_preds * ssup_masks, pseudo_targets.long(), self.ce_weight.to(batch_imgs.device))
        consist_weight = sigmoid_rampup(self.global_step, self.rampup_length)
        #
        total_loss = dice_loss + ce_loss + consist_weight * (consist_loss + ssup_dice_loss + ssup_ce_loss)
        return (
            [
                'DiceLoss', 'CELoss',
                'W.Consist', 'Consist', 'SSupDiceLoss', 'SSupCELoss', 'SSupRatio',
                'Total'
            ],
            [
                round(dice_loss.item(), 3), round(ce_loss.item(), 3),
                round(consist_weight, 3), round(consist_loss.item(), 3), round(ssup_dice_loss.item(), 3), round(ssup_ce_loss.item(), 3), ssup_ratio,
                total_loss.item(),
            ],
            total_loss
        )

    def update(self, ):
        with torch.no_grad():
            update_ema_variables(self.unet_body, self.teacher_unet_body, self.global_step, self.alpha)
        self.global_step += 1

    def parameters(self, ):
        return list(self.unet_body.parameters()) + list(self.conv_pred.parameters())
