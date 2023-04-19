import numpy as np


class Metric:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    # '''
    def __call__(self, pred_mask, gt_mask):
        # pred_masks: (d, h, w)
        # gt_masks: (d, h, w)
        dice_dict = {}
        for class_index in range(self.num_classes):
            class_id = class_index + 1
            cur_pred = (pred_mask == class_id).astype(np.float)
            cur_gt = (gt_mask == class_id).astype(np.float)
            if np.sum(cur_gt) == 0:
                continue
            dice = 2 * np.sum(cur_pred * cur_gt) / (np.sum(cur_pred) + np.sum(cur_gt) + 1e-12)
            dice_dict[class_id] = dice
        return dice_dict, None