import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
# from torch.nn.modules.utils import _ntuple
import math

__all__ = [
        "SignalConv1D",
        "SignalConv2D",
        "SignalConv3D",
        ]

class SignalConvND(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: str = "same",  # "same", "valid", "reflect", "circular"
        transpose: bool = False,
        bias: bool = True,
        activation: Union[None, nn.Module] = None,  # Optional activation
    ):
        super().__init__()
        assert ndim in [1, 2, 3], f"Unsupported dimension: {ndim}"
        self.ndim = ndim
        self.kernel_size = self._to_tuple(kernel_size, ndim)
        self.stride = self._to_tuple(stride, ndim)
        self.padding_mode = padding.lower()
        self.transpose = transpose
        self.activation = activation

        Conv = {
            (1, False): nn.Conv1d,
            (2, False): nn.Conv2d,
            (3, False): nn.Conv3d,
            (1, True): nn.ConvTranspose1d,
            (2, True): nn.ConvTranspose2d,
            (3, True): nn.ConvTranspose3d,
        }[(ndim, transpose)]

        if self.transpose:
            self.conv = Conv(
                in_channels,
                out_channels,
                self.kernel_size,
                stride=self.stride,
                bias=bias,
                **self._get_transpose_padding()
            )
        else:
            self.conv = Conv(
                in_channels,
                out_channels,
                self.kernel_size,
                stride=self.stride,
                bias=bias,
            )

        # Xavier uniform like TensorFlow
        nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def _to_tuple(self, val, ndim):
        """
        Returns correct tuple shape for kernel and stride given the
        type of conv. E.g it would return for Conv2d: kernel=(k, k),
        Conv1d=(k)
        """
        return (val,) * ndim if isinstance(val, int) else val

    def _get_pad(self):
        """
        Returns
        """
        return tuple(
            max(k - s, 0) for k, s in zip(self.kernel_size, self.stride)
            )

    def _get_transpose_padding(self) -> Tuple[int, int]:
        """
        Returns (padding, output_padding) needed for ConvTranspose2d
        to double size for stride=2, or preserve size for stride=1.
        Only works for odd kernel for preserving when stride=1, needs fixing
        """
        padding = math.ceil((self.kernel_size[0] - self.stride[0]) / 2)
        output_padding = self.stride[0] + 2 * padding - self.kernel_size[0]
        return {'padding': padding,
                'output_padding': output_padding,}

    def _pad_input(self, x: torch.Tensor, pad: Tuple[int]) -> torch.Tensor:
        # Reverse order and split each dim into (before, after)
        pad_list = [[p//2, p-p//2] for p in reversed(pad)]
        pad_list = [p for x in pad_list for p in x]
        return F.pad(x, pad_list, mode="constant" if self.padding_mode == "same" else self.padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "valid" or self.transpose:
            out = self.conv(x)
        else:
            pad = self._get_pad()
            x = self._pad_input(x, pad)
            out = self.conv(x)

        return self.activation(out) if self.activation else out


def _conv_class_factory(name: str, ndim: int):
    """Create a subclass of SignalConvND with a fixed ndim."""
    def __init__(self, *args, **kwargs):
        # Now pass the fixed ndim and the remaining args/kwargs to the superclass
        super(cls, self).__init__(ndim=ndim, *args, **kwargs)

    cls = type(name, (SignalConvND,), {
        "__init__": __init__,
        "__doc__": f"{name} is a {ndim}D version of SignalConvND.",
    })
    return cls

# Create fixed-dimension classes
SignalConv1D = _conv_class_factory("SignalConv1D", 1)
SignalConv2D = _conv_class_factory("SignalConv2D", 2)
SignalConv3D = _conv_class_factory("SignalConv3D", 3)
