import torch


def iou(preds, targets):
    '''
    preds(targets): [n, 6]
        bbox: (xmin, ymin, zmin, xmax, ymax, zmax)
    '''
    ltu = torch.max(preds[:, : 3], targets[:, : 3])
    rbd = torch.min(preds[:, 3 :], targets[:, 3 :])
    whd = (rbd - ltu).clamp(min=0)
    overlap = whd[:, 0] * whd[:, 1] * whd[:, 2]
    volume1 = (
        (preds[:, 3] - preds[:, 0])
        * (preds[:, 4] - preds[:, 1])
        * (preds[:, 5] - preds[:, 2])
    )
    volume2 = (
        (targets[:, 3] - targets[:, 0])
        * (targets[:, 4] - targets[:, 1])
        * (targets[:, 5] - targets[:, 2])
    )
    iou = overlap / (volume1 + volume2 - overlap)
    return iou
