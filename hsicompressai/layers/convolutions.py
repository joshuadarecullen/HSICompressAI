"""Specialized convolution layers for hyperspectral image processing.

This module provides custom convolution layers designed for efficient processing
of hyperspectral imagery, including multi-head self-attention, heterogeneous 
convolutions, and mixed spatial-spectral processing.

Classes:
    MHSA3D: Multi-head self-attention for 3D hyperspectral data
    Het3DConv: Heterogeneous 3D convolution combining full and grouped convolutions
    HetConv: Heterogeneous convolution for mixed pointwise and grouped processing

Example:
    >>> import torch
    >>> # 3D Multi-head self-attention
    >>> x = torch.randn(2, 16, 64, 32, 32)  # (B, T, C, H, W)
    >>> mhsa = MHSA3D(channels=64, num_heads=8)
    >>> output = mhsa(x)
    
    >>> # Heterogeneous 3D convolution
    >>> x = torch.randn(2, 128, 64, 32, 32)  # (B, C, D, H, W)
    >>> het_conv = Het3DConv(128, 256, ratio=0.4)
    >>> output = het_conv(x)
"""

from torch import Tensor, cat, nn
import torch.nn.functional as F
import torch


__all__ = [
        "MHSA3D",
        "Het3DConv",
        "HetConv",
        ]


class MHSA3D(nn.Module):
    """Multi-Head Self-Attention for 3D hyperspectral data.
    
    Implements 3D multi-head self-attention specifically designed for hyperspectral
    image cubes. The layer processes spatio-spectral features by computing attention
    across all spatial and spectral dimensions simultaneously.
    
    The attention mechanism helps capture long-range dependencies across both spatial
    locations and spectral bands, which is crucial for hyperspectral image analysis.
    
    Attributes:
        num_heads (int): Number of attention heads
        temperature (nn.Parameter): Learnable temperature parameter for attention scaling
        qkv (nn.Conv3d): 1x1x1 convolution to generate query, key, value tensors
        qkv_conv (nn.Conv3d): Spatial convolution for QKV features
        project_out (nn.Conv3d): Output projection layer
        
    Args:
        channels (int, optional): Number of input channels. Defaults to 16.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        
    Example:
        >>> # Input: (batch, time/depth, channels, height, width)
        >>> x = torch.randn(2, 16, 64, 32, 32)
        >>> mhsa = MHSA3D(channels=64, num_heads=4)
        >>> output = mhsa(x)  # Same shape as input
        
    Note:
        Input tensor should have 5 dimensions: (B, T, C, H, W) where T can represent
        temporal frames or spectral depth.
    """

    def __init__(self, channels=16, num_heads=1) -> None:

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(channels,
                             channels * 3,
                             kernel_size=(1, 1, 1),
                             bias=False)

        self.qkv_conv = nn.Conv3d(channels * 3,
                                  channels * 3,
                                  kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1),
                                  groups=channels * 3,
                                  bias=False)  # 331

        self.project_out = nn.Conv3d(channels,
                                     channels,
                                     kernel_size=(1, 1, 1),
                                     bias=False)

    def forward(self, x: Tensor) -> Tensor:

        B, T, C, H, W = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, -1,  H * W * T)
        k = k.reshape(B, self.num_heads, -1,  H * W * T)
        v = v.reshape(B, self.num_heads, -1,  H * W * T)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)  # Why do we normalise?
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(B, T, -1, H, W))

        return out


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
