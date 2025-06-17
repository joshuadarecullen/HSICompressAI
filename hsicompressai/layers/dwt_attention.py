import torch.nn as nn
from torch import Tensor
from .pytorch_dwt import DWT_3D
from .MHSA3D import MHSA3D


class DWT_3D_Attention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 wave='haar') -> None:

        super().__init__()

        self.dwt = DWT_3D()
        self.attention = MHSA3D(channels=in_channels)

    def forward(self, x) -> Tensor:
        return self.attention(self.dwt(x))
