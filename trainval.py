import torch
import numpy as np
import torch.nn.functional as F
import tqdm
from utils.misc import validate_gradient, chamfer_dist_loss
from utils.eval_metrics import compute_metrics
from collections import defaultdict
from utils.eval_metrics import summarize_metrics, metrics2msg
from utils.misc import Timer


def parsing_predicts(R_x, t_x, R_y, t_y):

    R_xy = R_x.transpose(2, 1) @ R_y
    R_yx = R_y.transpose(2, 1) @ R_x
    t_xy = (t_x - t_y) @ R_y
    t_yx = (t_y - t_x) @ R_x

    bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R_xy.shape[0], 1, 1).to(R_xy.device)
    pred_xy = torch.cat([torch.cat([R_xy.transpose(2, 1), t_xy.transpose(1, 2)], dim=2), bot_row], dim=1)
    pred_yx = torch.cat([torch.cat([R_yx.transpose(2, 1), t_yx.transpose(1, 2)], dim=2), bot_row], dim=1)

    return pred_xy, pred_yx


def forward_one_batch(model, px, py, compute_loss=True):
    R_x, t_x, R_y, t_y = model(px, py)
    pred_xy, pred_yx = parsing_predicts(R_x, t_x, R_y, t_y)
    if compute_loss:
        px_w = px @ R_x.transpose(2, 1) + t_x
        py_w = py @ R_y.transpose(2, 1) + t_y
        eye = torch.eye(4, dtype=torch.float, device=pred_xy.device).unsqueeze(0).repeat(pred_xy.shape[0], 1, 1)
        loss = 3 * F.mse_loss(pred_xy @ pred_yx, eye) + chamfer_dist_loss(px_w, py_w)
        return pred_xy, loss
    else:
        return pred_xy
    
def evaluate(model, dataloader, device, compute_loss=False):
    losses = []
    eval_metrics = defaultdict(list)
    eval_timer = Timer()
    total_time = 0.
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for iters, (p0, p1, gt) in enumerate(tqdm.tqdm(dataloader, leave=False, desc='Evaluating')):
            p0, p1, gt = p0.to(device), p1.to(device), gt.to(device)
            eval_timer.tic()
            forward_res = forward_one_batch(model, p0, p1, compute_loss=compute_loss)
            total_time += eval_timer.toc(average=False)
            num_samples += gt.shape[0]
            if compute_loss:
                pred_xy, loss = forward_res
                losses.append(loss.item())
            else:
                pred_xy = forward_res
            metrics = compute_metrics(pred_xy.detach(), gt, p0, p1)

            for k in metrics:
                eval_metrics[k].extend(metrics[k])
    avg_time = total_time / num_samples
    if compute_loss:
        return np.mean(losses), eval_metrics, avg_time
    else:
        return eval_metrics, avg_time
        

class Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        logger,
        writer,
        device='cuda'
        ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.writer = writer
        self.device = device

        self.model.to(device)
        self.logger.info(f"Running on {device}.")
        self.train_global_steps = 0
        self.min_loss = float('inf')
        self.min_rot = float('inf')

    def fit(self):
        self.logger.info(f"Training start!")
        for epoch in range(0, self.args.max_epochs):
            train_loss, train_rot, train_trans, train_ccd = self.train_one_epoch(epoch)
            train_log_msg = 'Train Epoch {:03d}/{:03d} ' \
                                'Loss: {:.5f} Rotation Error: {:.2f} ' \
                                'Translation Error: {:.4f} CCD: {:.4f}'.format(epoch, self.args.max_epochs,
                                                                                train_loss,
                                                                                train_rot, train_trans, train_ccd)
            # logging
            self.logger.info(train_log_msg)
            self.writer.add_scalar('Train/Epoch/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Epoch/RotationErr', train_rot, epoch)
            # self.writer.add_scalar('Train/Epoch/TranslationErr', train_trans, epoch)
            self.writer.add_scalar('Train/Epoch/CCD', train_ccd, epoch)
            self.writer.add_scalar('Train/Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            eval_loss, eval_metrics, avg_time = self.evaluate(epoch)
            val_log_msg = 'Evaluate Epoch {:03d}/{:03d} ' \
                      'Loss: {:.5f} Time: {:.5f}s'.format(epoch, self.args.max_epochs, eval_loss, avg_time)
            self.logger.info(val_log_msg)
            summary_metrics = summarize_metrics(eval_metrics)

            self.writer.add_scalar('Eval/Loss', eval_loss, epoch)
            self.writer.add_scalar('Eval/RotationErr', summary_metrics['err_r_deg_mean'], epoch)
            # self.writer.add_scalar('Eval/TranslationErr', summary_metrics['err_t_mean'], epoch)
            self.writer.add_scalar('Eval/CCD', summary_metrics['ccd'], epoch)

            show_detailed_msg = False
            if eval_loss <= self.min_loss:
                self.min_loss = eval_loss
                save_path = f"{self.args.ckpt_dir}/best_loss.pth"
                torch.save({'model': self.model.state_dict()}, save_path)
                self.logger.info(f"Got checkpoint with best loss {self.min_loss:.5f}, saving model state dict at {save_path}.")
                show_detailed_msg = True
            if summary_metrics['err_r_deg_mean'] <= self.min_rot:
                self.min_rot = summary_metrics['err_r_deg_mean']
                save_path = f"{self.args.ckpt_dir}/best_rot.pth"
                torch.save({'model': self.model.state_dict()}, save_path)
                self.logger.info(f"Got checkpoint with best rotation error {self.min_rot:.5f}, saving model state dict at {save_path}.")
                show_detailed_msg = True

            if show_detailed_msg:
                metrics_msg = metrics2msg(summary_metrics)
                self.logger.info(metrics_msg)

    def train_one_epoch(self, epoch):
        all_losses, all_rot, all_trans, all_ccd = [], [], [], []
        # self.logger.info(f"Start training Epoch {epoch}, LR is {self.optimizer.param_groups[0]['lr']:.5f}")
        self.model.train()
        
        for iters, (p0, p1, gt) in enumerate(tqdm.tqdm(self.train_loader, leave=False, desc=f"Epoch {epoch} LR {self.optimizer.param_groups[0]['lr']:.5f}")):
            # if iters > 10:
            #     break
            p0, p1, gt = p0.to(self.device), p1.to(self.device), gt.to(self.device)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            
            pred_xy, loss = forward_one_batch(self.model, p0, p1, compute_loss=True)
            loss.backward()
            metrics = compute_metrics(pred_xy.detach(), gt, p0, p1)
            
            self.train_global_steps += 1
            if validate_gradient(self.model):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            else:
                self.logger.info(f"Found invalid value in model parameters after Epoch {epoch}, Iteration {iters}.")
                self.optimizer.zero_grad()

            # logging
            self.writer.add_scalar('Train/Step/Loss', loss.detach().item(), self.train_global_steps)
            self.writer.add_scalar('Train/Step/Rotation', np.mean(metrics['err_r_deg']), self.train_global_steps)
            # self.writer.add_scalar('Train/Step/Translation', np.mean(metrics['err_t']), self.train_global_steps)
            self.writer.add_scalar('Train/Step/CCD', np.mean(metrics['ccd']), self.train_global_steps)

            all_losses.append(loss.detach().item())
            all_rot.extend(metrics['err_r_deg'])
            all_trans.extend(metrics['err_t'])
            all_ccd.extend(metrics['ccd'])
        self.scheduler.step()
        return np.mean(all_losses), np.mean(all_rot), np.mean(all_trans), np.mean(all_ccd)

    def evaluate(self, epoch):
        # self.logger.info(f"Evaluating Epoch {epoch}...")
        return evaluate(self.model, dataloader=self.val_loader, device=self.device, compute_loss=True)
