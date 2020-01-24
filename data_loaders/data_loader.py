from torch.utils import data
from torchvision import transforms as T

from data_loaders.DatasetMix import DatasetMix
from .MNISTm import MNISTm
from .MNISTthin import MNISTthin


def get_loader(batch_size=16, dataset='MNISTm', which_set='train', num_workers=1, mnistpath=None, mnistmpath=None,
               mnistthinpath=None, source='MNISTthin', colors=False):
    """Build and return a data loader."""
    transform = []

    # dataset transformations
    # if which_set == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    # dataset
    if dataset == 'MNISTthin':
        dataset = MNISTthin(which_set, mnistthinpath,
                            split=0.84 if which_set in ['train', 'val'] else 1.,
                            transform_list=transform, colors=colors)
    elif dataset == 'MNISTm':
        dataset = MNISTm(which_set, mnistmpath,
                         mnistpath, split=0.9 if which_set in ['train', 'val'] else 1.,
                         transform_list=transform)
    elif dataset == 'mix':
        mnistthin = MNISTthin(which_set, mnistthinpath,
                              split=0.84 if which_set in ['train', 'val'] else 1.,
                              transform_list=transform, colors=colors)
        mnistm = MNISTm(which_set, mnistmpath,
                        mnistpath, split=0.9 if which_set in ['train', 'val'] else 1.,
                        transform_list=transform)
        dataset = DatasetMix(mnistthin if source == 'MNISTthin' else mnistm,
                             mnistthin if source != 'MNISTthin' else mnistm)
    # loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(which_set == 'train' or dataset == 'mix'),
                                  num_workers=num_workers)

    return data_loader


if __name__ == '__main__':
    mnistthin_dataloader_tr = get_loader(16, 'MNISTthin', 'train')
    mnistm_dataloader_tr = get_loader(16, 'MNISTm', 'train')

    mnistthin_dataloader_v = get_loader(16, 'MNISTthin', 'val')
    mnistm_dataloader_v = get_loader(16, 'MNISTm', 'val')

    mnistthin_dataloader_te = get_loader(16, 'MNISTthin', 'test')
    mnistm_dataloader_te = get_loader(16, 'MNISTm', 'test')
