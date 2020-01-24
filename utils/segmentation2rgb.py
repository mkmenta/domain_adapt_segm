import numpy as np
import matplotlib.pyplot as plt
import torch

def segmentation2rgb(labels, original_im=None, n_labels=None, alpha=.5, cmap='jet'):
    # labels.shape must be (N,C,H,W) (same for original_im)
    # output.shape = (N,3,H,W) in RGB mode
    # Get the number of labels
    n_labels = len(np.unique(labels)) if n_labels is None else n_labels

    # Get colormap
    colors = plt.get_cmap(cmap)

    # Generate the label image
    label_image = torch.zeros((3,) + labels.shape)
    if labels.is_cuda:
        label_image = label_image.cuda()
    for i in range(n_labels):
        label_image[0][(labels == i)] = colors(i / (n_labels - 1))[0]
        label_image[1][(labels == i)] = colors(i / (n_labels - 1))[1]
        label_image[2][(labels == i)] = colors(i / (n_labels - 1))[2]
    label_image = label_image.permute([1, 0, 2, 3])

    # Add overlay with original image
    if original_im is not None:
        gray_im = original_im.mean(1).unsqueeze(1).repeat([1, 3, 1, 1])
        output = gray_im * alpha + label_image * (1 - alpha)
    else:
        output = label_image
    return output