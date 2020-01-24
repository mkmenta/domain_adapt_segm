import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

__all__ = ["MNISTthin"]


class MNISTthin(Dataset):
    def __init__(self, which_set, path, transform_list=[T.ToTensor()], colors=False, split=1.0, require_labels=False):
        if which_set in ['train', 'val']:
            self.images = np.load(os.path.join(path, 'train-images.npy'))
            self.labels = np.load(os.path.join(path, 'train-labels.npy'))
        else:
            self.images = np.load(os.path.join(path, 'test-images.npy'))
            self.labels = np.load(os.path.join(path, 'test-labels.npy'))

        # split
        ntotal = self.images.shape[0]
        ntrain = int(ntotal * split)
        if which_set == 'train':
            self.images = self.images[:ntrain]
            self.labels = self.labels[:ntrain]
        elif which_set == 'val':
            self.images = self.images[ntrain:]
            self.labels = self.labels[ntrain:]

        print('Number of images: ', len(self.images), which_set)
        print('Number of labels: ', len(self.labels), which_set)

        # class attributes
        self.transform = T.Compose(transform_list)
        self.n_classes = 2
        self.colors = colors
        self.require_labels = require_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        mask = image[:, :, 0].copy()
        mask = (mask > 128).astype(np.uint8)

        # add colors
        if self.colors:
            background = np.random.randint(0, 255, 3)
            digit = np.random.randint(0, 255, 3)
            image[mask == 0] = background
            image[mask == 1] = digit

        # to tensors & transformations
        image = self.transform(Image.fromarray(image))
        mask = torch.LongTensor(mask)

        if not self.require_labels:
            return image, mask
        else:
            return image, mask, label


if __name__ == '__main__':
    train_data = MNISTthin('train', 'path_to_mnist', split=1.0, colors=False)
    test_data = MNISTthin('test', 'path_to_mnist', split=1.0, colors=False)

    im_tr, mask_tr = train_data[0]
    im_te, mask_te = test_data[0]

    print(im_tr.size)
    print(im_te.size)

    print(mask_tr.size)
    print(mask_te.size)
