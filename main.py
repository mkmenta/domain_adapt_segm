import argparse
import pickle
import sys

import os
from torch.backends import cudnn
from torchvision.transforms import transforms

from data_loaders.data_loader import get_loader
from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Transforms
    target = 'MNISTm' if config.source == 'MNISTthin' else 'MNISTthin'

    if config.mode == 'train':
        # logs to disk
        if config.log_term:
            print("Training logs will be saved to:", os.path.join(config.log_dir, 'train.log'))
            sys.stdout = open(os.path.join(config.log_dir, 'train.log'), 'w')
            sys.stderr = open(os.path.join(config.log_dir, 'train.err'), 'w')

        # save config
        pickle.dump(config, open(os.path.join(config.log_dir, 'config.pkl'), 'wb'))

        # Get datasets
        # Data loader
        mix_loader = get_loader(config.batch_size, 'mix', 'train', config.num_workers,
                                mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                mnistthinpath=config.mnist_thin_dir, source=config.source, colors=config.colors)
        mix_loader_val = get_loader(config.batch_size, 'mix', 'val', config.num_workers, mnistpath=config.mnist_dir,
                                    mnistmpath=config.mnist_m_dir, mnistthinpath=config.mnist_thin_dir,
                                    source=config.source, colors=config.colors)
        source_loader = get_loader(config.batch_size, config.source, 'train', config.num_workers,
                                   mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                   mnistthinpath=config.mnist_thin_dir, colors=config.colors)
        source_loader_val = get_loader(config.batch_size, config.source, 'val', config.num_workers,
                                       mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                       mnistthinpath=config.mnist_thin_dir, colors=config.colors)
        target_loader = get_loader(config.batch_size, target, 'train', config.num_workers,
                                   mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                   mnistthinpath=config.mnist_thin_dir, colors=config.colors)
        target_loader_val = get_loader(config.batch_size, target, 'val', config.num_workers,
                                       mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                       mnistthinpath=config.mnist_thin_dir, colors=config.colors)
        solver = Solver(config, mix_loader, source_loader, mix_loader_val, source_loader_val, target_loader,
                        target_loader_val)

        solver.train()
    else:
        # Get datasets
        # Data loader
        mix_loader = get_loader(config.batch_size, 'mix', 'test', config.num_workers,
                                mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                mnistthinpath=config.mnist_thin_dir, source=config.source,
                                colors=config.colors)
        source_loader = get_loader(config.batch_size, config.source, 'test', config.num_workers,
                                   mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                   mnistthinpath=config.mnist_thin_dir, colors=config.colors)
        target_loader = get_loader(config.batch_size, target, 'test', config.num_workers,
                                   mnistpath=config.mnist_dir, mnistmpath=config.mnist_m_dir,
                                   mnistthinpath=config.mnist_thin_dir, colors=config.colors)

        config.mode = 'test'

        solver = Solver(config, mix_loader, source_loader, None, target_loader)
        solver.test('source', condition_target='source')
        solver.test('target', condition_target='source')
        solver.test('target', condition_target='target')
        print(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--g_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--s_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=0, help='number of residual blocks in G')
    parser.add_argument('--s_repeat_num', type=int, default=0, help='number of strided conv layers in D')
    parser.add_argument('--g_num_init', type=int, default=1, help='number initial convolutions of G')
    parser.add_argument('--s_num_init', type=int, default=0, help='number initial convolutions of S')
    parser.add_argument('--g_num_down', type=int, default=2, help='number downsampling blocks of G')
    parser.add_argument('--d_num_down', type=int, default=4, help='number downsampling blocks of D')
    parser.add_argument('--s_num_down', type=int, default=0, help='number downsampling blocks of S')
    parser.add_argument('--df_num_down', type=int, default=0, help='number downsampling blocks of Df')
    parser.add_argument('--g_num_up', type=int, default=2, help='number upsampling blocks of G')
    parser.add_argument('--s_num_up', type=int, default=2, help='number upsampling blocks of S')
    parser.add_argument('--df_num_up', type=int, default=0, help='number upsampling blocks of Df')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_cycle', type=float, default=10, help='weight for cycle loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_segm', type=float, default=10, help='weight for segmentation loss')
    parser.add_argument('--lambda_id', type=float, default=0., help='weight for identity loss')
    parser.add_argument('--lambda_ffeat', type=float, default=0., help='weight for l1 feature matching')
    parser.add_argument('--lambda_fdom', type=float, default=0., help='weight for fdom loss in Df')
    parser.add_argument('--lambda_frf', type=float, default=0., help='weight for real vs fake loss in Df')
    parser.add_argument('--drop_d', type=float, default=0.2, help='dropout for D')
    parser.add_argument('--drop_g', type=float, default=0.2, help='dropout for G and S')
    parser.add_argument('--da_type', type=str, default='')

    # Training configuration.
    parser.add_argument('--source', type=str, default='MNISTthin')
    parser.add_argument('--colors', dest='colors', action='store_true')
    parser.add_argument('--fake_segm', dest='fake_segm', action='store_true')
    parser.set_defaults(fake_segm=False)
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=800000, help='number of total iterations for training D')
    parser.add_argument('--patience', type=int, default=50, help='patience (in epochs) for early stopping')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--s_lr', type=float, default=0.0001, help='learning rate for S')
    parser.add_argument('--df_lr', type=float, default=0.0001, help='learning rate for Df')
    parser.add_argument('--lr_decay', type=float, default=0.995, help='lr decay factor')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')  # 0.5
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')  # 0.999
    parser.add_argument('--oracle_cond', dest='oracle_cond', action='store_true')
    parser.add_argument('--load_pretrained', default=None, type=str)
    parser.add_argument('--modules_pretrained', default=['G', 'S', 'D'], type=str, nargs='+')
    parser.add_argument('--df_source_only', dest='df_source_only', action='store_true')
    parser.add_argument('--df_move_one', dest='df_move_one', action='store_true')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--mnist_m_dir', type=str, default='')
    parser.add_argument('--mnist_thin_dir', type=str, default='')
    parser.add_argument('--mnist_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--exp_name', type=str)

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--val_step', type=int, default=1500)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--log_term', dest='log_term', action='store_true')

    config = parser.parse_args()

    print(config)

    config.log_dir = os.path.join(config.log_dir, config.exp_name)

    main(config)
