import torch.nn as nn

from .utils import ResidualBlock


class Segmenter(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6, num_down=2, num_up=2, num_init=1, bias=False, n_classes=2, drop=0.0,
                 in_channels=3):
        super(Segmenter, self).__init__()

        # initial transformation
        layers = []
        for i in range(num_init):
            layers.append(nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=1, padding=1, bias=bias))
            layers.append(nn.Dropout(drop))
            layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim if num_init > 0 else in_channels
        for i in range(num_down):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers.append(nn.Dropout(drop))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, bias=bias, drop=drop))

        # Up-sampling layers.
        for i in range(num_up):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers.append(nn.Dropout(drop))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Classifier
        layers.append(nn.Conv2d(curr_dim, n_classes, kernel_size=3, stride=1, padding=1, bias=bias))
        # layers.append(nn.Softmax(dim=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
