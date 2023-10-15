import torch
from tensorboardX import SummaryWriter
from model import UGMMReg
import argparse
from data.dataloader import get_datasets
from torch.utils.data import DataLoader
from datetime import datetime
import os
from utils.logger import *
from utils.misc import create_dirs

import open3d as o3d
from trainval import Trainer, evaluate
from utils.eval_metrics import summarize_metrics, metrics2msg

o3d_warn_level = o3d.utility.VerbosityLevel(0)
o3d.utility.set_verbosity_level(o3d_warn_level)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    # dataset & dataloader
    parser.add_argument('--dataset_type', default='modelnet')
    parser.add_argument('--dataset_path', default='./dataset/ModelNet40')
    parser.add_argument('--category_file', default='data/categories/modelnet40_half1.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--mag', type=float, default=3.14/4)
    parser.add_argument('--n_points', type=int, default=1024)

    # model
    parser.add_argument('--n_components', type=int, default=16)

    # optimizer
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.985)

    # training
    parser.add_argument('--max_epochs', type=int, default=200)

    # test only
    parser.add_argument('--test_ckpt_path',
                        default='./checkpoints/model.pth',
                        help='test checkpoint path')
    
    parser.add_argument('--gt_transforms', default='data/gt_transform_45.csv')
    # log
    parser.add_argument('--log_root', default='./logs')

    args = parser.parse_args()

    return args

def parsing_args(args):
    if args.mode == 'test':
        assert args.test_ckpt_path is not None
        assert os.path.exists(args.test_ckpt_path)
        assert os.path.exists(args.gt_transforms)
    # log
    start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.log_dir = f'{args.log_root}/{args.dataset_type}/{args.mode}/{start_time}'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.log_file_path = f"{args.log_dir}/{args.mode}.log"
    if args.mode == 'train':
        args.ckpt_dir = f"{args.log_dir}/checkpoints"
        args.tensorboard_dir = f"{args.log_dir}/tensorboard"
        create_dirs([args.log_dir, args.ckpt_dir, args.tensorboard_dir])
    else:
        create_dirs([args.log_dir])

def train(args, model, logger):
    writer = SummaryWriter(args.tensorboard_dir)

    # initiallize model and dataset
    train_dataset, val_dataset = get_datasets(args, dataset_type=args.dataset_type, mode='train')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, shuffle=False)

    # set optimizer and scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)

    trainer = Trainer(
                args,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                logger,
                writer,
                device=args.device
        )
    trainer.fit()

def test(args, model, logger):
    # loading state dict
    model.load_state_dict(torch.load(args.test_ckpt_path)['model'])
    test_set = get_datasets(args, dataset_type=args.dataset_type, mode='test')

    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)

    eval_metrics, avg_time = evaluate(model, test_loader, args.device, compute_loss=False)
    summary_metrics = summarize_metrics(eval_metrics)
    logger.info(metrics2msg(summary_metrics))
    logger.info(f"Avarage time of inference one sample: {avg_time:.4f}s")

    
def main(args):
    parsing_args(args)
    logger = get_root_logger(log_file=args.log_file_path, name='UGMMReg')

    logger.info(str(args))

    model = UGMMReg(args).to(args.device)

    if args.mode == 'train':
        train(args, model, logger)
    elif args.mode == 'test':
        test(args, model, logger)
    else:
        raise ValueError

    logger.info('Done!')
    
if __name__ == '__main__':
    args = get_args()
    main(args)

