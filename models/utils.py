import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, bias=False, drop=0.0):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Dropout(drop),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Dropout(drop),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False))

    def forward(self, x):
        return x + self.main(x)
