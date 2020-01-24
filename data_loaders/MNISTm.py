import glob

import numpy as np
import os
import struct
import torch
from PIL import Image
from torch.nn.functional import pad
from torch.utils import data
from torchvision import transforms as T

__all__ = ["MNISTm"]


class MNISTm(data.Dataset):
    def __init__(self, which_set, path_mnistm, path_mnist, transform_list=[T.ToTensor()], split=0.84,
                 require_labels=False):

        # Paths to images and labels
        if which_set == 'train':
            self.path_images = os.path.join(path_mnistm, 'mnist_m_train')
            self.path_labels = os.path.join(path_mnistm, 'mnist_m_train_labels.txt')
            self.nimages = int(len(glob.glob(self.path_images + '/*.png')) * split)
        elif which_set == 'val':
            self.path_images = os.path.join(path_mnistm, 'mnist_m_train')
            self.path_labels = os.path.join(path_mnistm, 'mnist_m_train_labels.txt')
            self.ntrain = int(len(glob.glob(self.path_images + '/*.png')) * split)
            self.nimages = len(glob.glob(self.path_images + '/*.png')) - self.ntrain
        else:
            self.path_images = os.path.join(path_mnistm, 'mnist_m_test')
            self.path_labels = os.path.join(path_mnistm, 'mnist_m_test_labels.txt')
            self.nimages = len(glob.glob(self.path_images + '/*.png'))

        # Load Labels
        labels = []
        with open(self.path_labels, 'r') as f:
            for line in f:
                labels.append(int(line.split(' ')[1]))

        # Load Masks
        if which_set in ['train', 'val']:
            path_images_mnist = os.path.join(path_mnist, 'train-images-idx3-ubyte')
        else:
            path_images_mnist = os.path.join(path_mnist, 't10k-images-idx3-ubyte')

        with open(path_images_mnist, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            masks = (np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols) > 0).astype(np.uint8)
            masks = pad(torch.LongTensor(masks), (2, 2, 2, 2), mode='constant', value=0)

        # split
        if which_set == 'train':
            self.labels = labels[:self.nimages]
            self.masks = masks[:self.nimages]
        elif which_set == 'val':
            self.labels = labels[self.ntrain:]
            self.masks = masks[self.ntrain:len(labels)]
        else:
            self.labels = labels
            self.masks = masks[:len(labels)]

        print('Number of images: ', self.nimages, which_set)
        print('Number of labels: ', len(self.labels), which_set)
        print('Number of masks: ', len(self.masks), which_set)

        self.n_classes = 2
        self.which_set = which_set
        self.require_labels = require_labels
        self.transform = T.Compose(transform_list)

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):

        # adjust idx for validation set
        if self.which_set == 'val':
            idx_img = idx + self.ntrain
        else:
            idx_img = idx

        # Load image
        fname = '00000000' + str(idx_img) + '.png'
        fname = fname[-12:]
        image = Image.open(os.path.join(self.path_images, fname))  # Range [0,255]
        image = self.transform(image)

        # Mask
        mask = self.masks[idx]

        # Label
        label = self.labels[idx]

        if not self.require_labels:
            return image, mask
        else:
            return image, mask, label


if __name__ == '__main__':
    train_data = MNISTm('train', 'path_to_mnistm', 'path_to_mnist', split=1.0)
    test_data = MNISTm('test', 'path_to_mnistm', 'path_to_mnist', split=1.0)

    im_tr, mask_tr = train_data[0]
    im_te, mask_te = test_data[0]

    print(im_tr.size)
    print(im_te.size)

    print(mask_tr.size)
    print(mask_te.size)
