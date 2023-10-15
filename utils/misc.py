import os
import torch
import time
import numpy as np

def create_dirs(dirs: list):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu()
    return np.array(arr)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            n = val.size
            val = val.mean()
        elif isinstance(val, torch.Tensor):
            n = val.nelement()
            val = val.mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(AverageMeter):
    """A simple timer."""

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.update(self.diff)
        if average:
            return self.avg
        else:
            return self.diff

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print('Current model params has Nan value!')
                return False
            if torch.any(torch.isinf(param.grad)):
                print('Current model params has Infinite value!')
                return False
    return True


def chamfer_dist_loss(a, b):
    # x, y = a.transpose(2, 1).contiguous(), b.transpose(2, 1).contiguous()
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    loss = (torch.mean(P.min(1)[0])) + (torch.mean(P.min(2)[0]))
    return loss