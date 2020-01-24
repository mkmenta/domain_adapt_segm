import torch
import torch.nn as nn


class FeatureDiscriminator(nn.Module):
    """Discriminator network to perform domain adaptation."""

    def __init__(self, inplanes, seg_nclasses=0, num_ups_feat=2, num_downs=4, normalization=False,
                 leaky_relu=0.01, bias=True, drop=0.):
        super(FeatureDiscriminator, self).__init__()
        self.num_ups_feat = num_ups_feat
        self.seg_nclasses = seg_nclasses

        # Feature upsampling
        layers_ups = []
        curr_dim = inplanes
        for i in range(self.num_ups_feat):
            layers_ups.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers_ups.append(nn.Dropout(drop))
            if normalization:
                layers_ups.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
            layers_ups.append(nn.LeakyReLU(leaky_relu))
            curr_dim = curr_dim // 2
        self.upsampler = nn.Sequential(*layers_ups)

        # Downsampling
        layers = []
        layers.append(nn.Conv2d(curr_dim + self.seg_nclasses, curr_dim * 2, kernel_size=4, stride=2, padding=1,
                                bias=bias))
        layers.append(nn.Dropout(drop))
        if normalization:
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
        layers.append(nn.LeakyReLU(leaky_relu))
        curr_dim = curr_dim * 2

        for i in range(1, num_downs):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=bias))
            layers.append(nn.Dropout(drop))
            if normalization:
                layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
            layers.append(nn.LeakyReLU(leaky_relu))
            curr_dim = curr_dim * 2

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.main = nn.Sequential(*layers)

        # Output
        self.conv_src = nn.Linear(curr_dim, 1, bias=bias)
        self.conv_cls = nn.Linear(curr_dim, 1, bias=bias)

    def forward(self, x, seg=None):
        x = self.upsampler(x)
        if self.seg_nclasses > 0 and seg is not None:
            x = torch.cat([x, seg], dim=1)
        h = self.main(x)
        h = h.view(h.size(0), -1)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls


class PixelwiseFeatureDiscriminator(nn.Module):
    def __init__(self, inplanes, num_ups=2, extra_conv=False, leaky_relu=0.01, drop=0.):
        super(PixelwiseFeatureDiscriminator, self).__init__()

        # Upsampling
        curr_dim = inplanes
        layers = []
        for i in range(num_ups):
            if extra_conv:
                layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Dropout(drop))
                layers.append(nn.LeakyReLU(leaky_relu))
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.Dropout(drop))
            layers.append(nn.LeakyReLU(leaky_relu))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        # Output
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)
        self.conv_cls = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls
