from collections import OrderedDict

import torch
import torch.nn as nn


def softIoU(out, target, e=1e-6):
    sm = nn.Softmax(dim=1)
    out = sm(out)

    target = target.float()

    out = out[:, 1, :, :]

    num = (out * target).sum()
    den = (out + target - out * target).sum() + e
    iou = num / den

    return iou.mean()


class softIoULoss(nn.Module):
    def __init__(self, e=1e-6):
        super(softIoULoss, self).__init__()
        self.e = e

    def forward(self, inputs, targets):
        return 1.0 - softIoU(inputs, targets, self.e)


def update_cm(cm, y_pred, y_true):
    y_pred = torch.argmax(y_pred, 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            cm[i, j] += ((y_pred == i) * (y_true == j)).sum().float()

    return cm


def compute_metrics(cm, ret_metrics, eps=1e-8):
    TP_perclass = cm.diag()
    FP_perclass = cm.sum(1) - TP_perclass
    FN_perclass = cm.sum(0) - TP_perclass

    ret_metrics['accuracy'] = TP_perclass.sum() / cm.sum()
    iou_perclass = TP_perclass / (TP_perclass + FP_perclass + FN_perclass + eps)
    ret_metrics['iou_perclass_0'] = iou_perclass[0]
    ret_metrics['iou_perclass_1'] = iou_perclass[1]
    ret_metrics['iou'] = iou_perclass.mean()

    return ret_metrics


def print_metrics(init, metrics, time=None):
    out_str = init
    metrics = OrderedDict(metrics)
    for k in metrics.keys():
        try:
            out_str += (k + ': {:.3f} | ' * len(metrics[k])).format(*metrics[k])
        except:
            out_str += (k + ': {:.3f} | ').format(metrics[k])

    if time is not None:
        out_str += ("time {:.3f}s").format(time)

    print(out_str)
