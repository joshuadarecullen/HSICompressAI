import torch
from torch import nn
from torch.functional import F
from einops import rearrange

__all__ = [
        "ClusterX"
        ]


class ClusterX(nn.Module):
    """
    Creates specified clusters of contininous bands from a Hyperspectral Image.
    The HSI is padded if number of bands is not divisble by cluster_size.
    """
    def __init__(self, cluster_size: int) -> None:

        super().__init__()
        self.cluster_size = cluster_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_size = (self.cluster_size - x.shape[0] % self.cluster_size) % self.cluster_size
        x = self._pad_spectral_to_divisible(x, pad_size)

        # cluster each image into 3 bands
        x = rearrange(x, '(n c) h w -> n c h w', c=self.cluster_size)
        return x

    def _pad_spectral_to_divisible(self, x: torch.Tensor, pad_size: int):
        """
        Pads the spectral (channel) dimension (dim=1) of a 4D tensor [B, C, H, W]
        so that C is divisible by `divisor`. Pads with zeros at the end.
        Returns:
            x_padded: the padded tensor
            pad_size: number of channels padded (to undo later if needed)
        """
        return F.pad(x, (0, 0, 0, 0, 0, pad_size))  # Only pad the channels
