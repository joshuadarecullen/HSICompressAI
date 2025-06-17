import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import torch


class MHSA3D(nn.Module):

    def __init__(self, channels=16, num_heads=1) -> None:

        super(MHSA3D, self).__init__()

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
