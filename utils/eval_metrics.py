import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils import dcputil
from utils.misc import to_numpy


def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def calculate_ccd(src, dst):
    clip_val = torch.tensor(0.1, device=src.device, dtype=torch.float32)
    dist_src = torch.minimum(torch.min(torch.sqrt(square_distance(src, dst)), dim=-1)[0], clip_val)
    dist_ref = torch.minimum(torch.min(torch.sqrt(square_distance(dst, src)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=-1) + torch.mean(dist_ref, dim=-1)
    return clip_chamfer_dist


def npmat2euler(mats, seq='zyx'):
    eulers = []
    r = Rotation.from_matrix(mats)
    eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def compute_metrics(T_pre, T_gt, P1_gt, P2_gt):
    # compute r,t
    R_pre, t_pre = T_pre[:, :3, :3], T_pre[:, :3, 3]
    R_gt, t_gt = T_gt[:, :3, :3], T_gt[:, :3, 3]

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((t_gt - t_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(t_gt - t_pre), dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = dcputil.concatenate(dcputil.inverse(R_gt.cpu().numpy(), t_gt.cpu().numpy()),
                                       np.concatenate([R_pre.cpu().numpy(), t_pre.unsqueeze(-1).cpu().numpy()],
                                                      axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    # Chamfer distance
    P1_transformed = P1_gt @ R_pre.transpose(2, 1) + t_pre.unsqueeze(1)
    ccd = calculate_ccd(P1_transformed, P2_gt)

    metrics = {
        'r_mae': r_mae,
        't_mae': to_numpy(t_mae),
        'err_r_deg': to_numpy(residual_rotdeg),
        'err_t': to_numpy(residual_transmag),
        'ccd': to_numpy(ccd),
        'r_mse': r_mse,
        't_mse': to_numpy(t_mse)
    }

    return metrics

def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(np.array(metrics[k])**2))
        elif k.endswith('nomean'):
            summarized[k] = metrics[k]
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def metrics2msg(summary_metrics):
    """Prints out formated metrics to logger"""
    msg = ''

    msg+='\nDeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)\n'.format(summary_metrics['r_rmse'], summary_metrics['r_mae'],
                    summary_metrics['t_rmse'], summary_metrics['t_mae'])
    msg+='Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)\n'.format(summary_metrics['err_r_deg_mean'],
                 summary_metrics['err_r_deg_rmse'])
    msg+='Translation error {:.4g}(mean) | {:.4g}(rmse)\n'.format(summary_metrics['err_t_mean'],
                 summary_metrics['err_t_rmse'])
    msg+='Clip Chamfer error: {:.7f}(mean-sq)\n'.format(summary_metrics['ccd'])

    return msg
