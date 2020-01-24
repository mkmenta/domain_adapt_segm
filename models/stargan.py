import numpy as np
import torch
import torch.nn as nn

from .utils import ResidualBlock


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6, num_down=2, num_up=2, num_init=1, bias=False, drop=0.0):
        super(Generator, self).__init__()

        # initial transformation
        layers_encoder = []
        for i in range(num_init):
            layers_encoder.append(nn.Conv2d(4, conv_dim, kernel_size=3, stride=1, padding=1, bias=bias))
            layers_encoder.append(nn.Dropout(drop))
            layers_encoder.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False))
            layers_encoder.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(num_down):
            layers_encoder.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers_encoder.append(nn.Dropout(drop))
            layers_encoder.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
            layers_encoder.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.bottleneck_dim = curr_dim

        # Bottleneck layers.
        for i in range(repeat_num):
            layers_encoder.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, bias=bias, drop=drop))

        self.encoder = nn.Sequential(*layers_encoder)

        # Up-sampling layers.
        layers_decoder = []
        for i in range(num_up):
            layers_decoder.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers_decoder.append(nn.Dropout(drop))
            layers_decoder.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
            layers_decoder.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Output
        layers_decoder.append(nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=bias))
        layers_decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers_decoder)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        h = self.encoder(x)
        return self.decoder(h), h


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6, drop=0.0):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Dropout(drop))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.Dropout(drop))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
