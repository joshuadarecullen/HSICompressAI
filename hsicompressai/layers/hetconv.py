from torch import Tensor, cat, nn


class Het3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4, ratio=0.5):
        super(Het3DConv, self).__init__()

        # Number of filters for full and grouped convolutions
        full_out = int(out_channels * ratio)  # Full convolution filters
        grouped_out = out_channels - full_out  # Grouped convolution filters
        group_size = in_channels // groups  # Channels per group

        # Full Convolution (spatial processing)
        self.full_conv = nn.Conv3d(
            in_channels, full_out, kernel_size, stride, padding, bias=False
        )

        # Grouped Convolution (spectral processing)
        self.grouped_conv = nn.Conv3d(
            in_channels, grouped_out, kernel_size,
            stride, padding, groups=groups, bias=False
        )

        # Fusion layer (1x1x1 Conv)
        self.fusion = nn.Conv3d(out_channels, out_channels,
                                kernel_size=1, bias=False)

    def forward(self, x):
        full_out = self.full_conv(x)  # Standard 3D conv (spatial)
        grouped_out = self.grouped_conv(x)  # Grouped 3D conv (spectral)
        # Concatenate along channel dimension
        out = cat([full_out, grouped_out], dim=1)
        out = self.fusion(out)  # Fuse features

        return out


class HetConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 bias: Tensor = None,
                 groups_p: int = 220,
                 groups_g: int = 220) -> None:

        super().__init__()

        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size,
                             groups=groups_g,
                             padding=kernel_size//3,
                             stride=stride)

        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             groups=groups_p,
                             stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)
