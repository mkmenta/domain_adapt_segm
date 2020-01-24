import argparse
import pickle
import sys
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from data_loaders.data_loader import get_loader
from models.segmenter_baseline import Segmenter
from utils.metrics import softIoULoss, compute_metrics, update_cm, print_metrics
from utils.visualizer import Visualizer


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor


def restore_model(model, dir):
    """Restore the trained model."""
    print('Loading best models')
    path = os.path.join(dir, 'model.ckpt')
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return model


def train(args):
    # Create directories if not exist.
    model_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logs to disk
    if not args.log_term:
        print("Training logs will be saved to:", os.path.join(model_dir, 'train.log'))
        sys.stdout = open(os.path.join(model_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(model_dir, 'train.err'), 'w')
    # save args
    pickle.dump(args, open(os.path.join(model_dir, 'args.pkl'), 'wb'))

    curr_pat = 0

    # Data loader
    source_train = get_loader(args.batch_size, args.source, 'train', args.num_workers,
                              mnistpath=args.mnist_dir, mnistmpath=args.mnist_m_dir,
                              mnistthinpath=args.mnist_thin_dir, source=args.source, colors=False)
    source_val = get_loader(args.batch_size, args.source, 'val', args.num_workers,
                            mnistpath=args.mnist_dir, mnistmpath=args.mnist_m_dir,
                            mnistthinpath=args.mnist_thin_dir, source=args.source, colors=False)
    source_test = get_loader(args.batch_size, args.source, 'test', args.num_workers,
                             mnistpath=args.mnist_dir, mnistmpath=args.mnist_m_dir,
                             mnistthinpath=args.mnist_thin_dir, source=args.source, colors=False)
    target_test = get_loader(args.batch_size, args.target, 'test', args.num_workers,
                             mnistpath=args.mnist_dir, mnistmpath=args.mnist_m_dir,
                             mnistthinpath=args.mnist_thin_dir, source=args.target, colors=False)

    # Training criterion
    if args.criterion == 'softiou':
        criterion = softIoULoss()
    elif args.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss')

    # Build model
    model = Segmenter(conv_dim=args.conv_dim, repeat_num=args.repeat_num, num_down=args.num_down, bias=True,
                      n_classes=2, drop=args.drop)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # gpus
    model = model.cuda()
    cudnn.benchmark = True

    # Visualizer
    if args.use_tensorboard:
        visualizer = Visualizer(model_dir, name='visual_results')

    # Train the model
    for epoch in range(0, args.num_epochs):
        # reset visualizer
        visualizer.reset()

        # increase / decrase values for moving params
        set_lr(optimizer, args.lr_decay)

        # split loop
        for split in ['train', 'val']:

            if split == 'train':
                loader = source_train
                model.train()
            else:
                loader = source_val
                model.eval()

            metrics = {'loss': 0, 'iou': [], 'accuracy': []}
            cm = torch.from_numpy(np.zeros((2, 2))).float().cuda()

            total_step = len(loader)
            torch.cuda.synchronize()
            start = time.time()

            # minibatch loop
            for i, (images, gts) in enumerate(loader):
                global_iter = total_step * epoch + i

                # send to cuda
                images = images.cuda()
                gts = gts.cuda()

                loss_dict = {}

                if split == 'val':
                    with torch.no_grad():
                        outputs = model(images)
                else:
                    outputs = model(images)

                # loss computation
                loss = criterion(outputs, gts)

                # update confusion matrix
                cm = update_cm(cm, outputs, gts)

                # update dicts
                loss_dict['loss'] = loss.data
                metrics['loss'] += loss_dict['loss']

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    visualizer.scalar_summary(mode=split, epoch=global_iter, **loss_dict)

            # end of epoch
            metrics['loss'] /= total_step
            str_endepoch = 'total epoch %d; split: %s; loss: %.4f; time: %s' % (
            epoch, split, metrics['loss'], time.time() - start)
            print(str_endepoch)

            torch.cuda.synchronize()
            start = time.time()

            # compute metrics and visualize them
            metrics = compute_metrics(cm, metrics)

            if split == 'train':
                visualizer.scalar_summary(mode=split, epoch=epoch,
                                          **{k: v for k, v in metrics.items() if v and k != 'loss'})

            if split == 'val':
                visualizer.scalar_summary(mode=split, epoch=epoch, **metrics)

        # Save the model checkpoints if performance was improved
        if epoch == 0 or metrics['loss'] < es_best:
            es_best = metrics['loss']
            torch.save(model.state_dict(), os.path.join(
                model_dir, 'model.ckpt'))
            torch.save(optimizer.state_dict(), os.path.join(
                model_dir, 'optim.ckpt'))

            curr_pat = 0
        else:
            curr_pat += 1

        if curr_pat > args.patience:
            break

    visualizer.close()

    # restore model
    model = restore_model(model, model_dir)

    # test on source domain
    model.eval()
    cm = torch.from_numpy(np.zeros((2, 2))).float().cuda()
    metrics = {'loss': 0, 'iou': [], 'accuracy': []}
    for i, (images, gts) in enumerate(source_test):
        # send to cuda
        images = images.cuda()
        gts = gts.cuda()

        with torch.no_grad():
            outputs = model(images)

        # loss computation
        loss = criterion(outputs, gts)

        # update confusion matrix
        cm = update_cm(cm, outputs, gts)

        # update dicts
        metrics['loss'] += loss.data

    # compute metrics and visualize them
    metrics['loss'] /= len(source_test)
    metrics = compute_metrics(cm, metrics)

    print_metrics('TEST SOURCE: ', metrics)

    # test on target domain
    model.eval()
    cm = torch.from_numpy(np.zeros((2, 2))).float().cuda()
    metrics = {'loss': 0, 'iou': [], 'accuracy': []}
    for i, (images, gts) in enumerate(target_test):
        # send to cuda
        images = images.cuda()
        gts = gts.cuda()

        with torch.no_grad():
            outputs = model(images)

        # loss computation
        loss = criterion(outputs, gts)

        # update confusion matrix
        cm = update_cm(cm, outputs, gts)

        # update dicts
        metrics['loss'] += loss.data

    # compute metrics and visualize them
    metrics['loss'] /= len(target_test)
    metrics = compute_metrics(cm, metrics)

    print_metrics('TEST TARGET: ', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Domains args
    parser.add_argument('--source', type=str, default='MNISTthin', help='Source domain - dataset name)')
    parser.add_argument('--target', type=str, default='MNISTm', help='Target domain - dataset name)')

    # Segmenter args
    parser.add_argument('--conv_dim', type=int, default=32, help='dimension of first convolution')
    parser.add_argument('--repeat_num', type=int, default=0, help='number of bottleneck layers')
    parser.add_argument('--num_down', type=int, default=2,
                        help='number of downsampling operations (same will be used for upsampling)')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout probability')

    # Training args
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of total epochs for training')
    parser.add_argument('--patience', type=int, default=50, help='number of total epochs for training')
    parser.add_argument('--lr_decay', type=int, default=0.995, help='lr decay factor')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for S')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    # parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--criterion', type=str, default='softiou', help='Training criterion',
                        choices=['softiou', 'crossentropy'])

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_tensorboard', type=str, default=True)
    parser.add_argument('--log_term', type=bool, default=True)

    # Directories.
    parser.add_argument('--mnist_dir', type=str, default='')
    parser.add_argument('--mnist_thin_dir', type=str, default='')
    parser.add_argument('--mnist_m_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='segmentation_baseline_mnistthin')

    args = parser.parse_args()
    print(args)
    train(args)
