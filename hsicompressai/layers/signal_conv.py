import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


__all__ = [
        "SignalConv1D",
        "SignalConv2D",
        "SignalConv3D"
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
        self.stride = self._to_tuple(stride, ndim)
        self.kernel_size = self._to_tuple(kernel_size, ndim)
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
        return (val,) * ndim if isinstance(val, int) else val

    def _get_pad(self):
        return tuple(
            max(k - s, 0) for k, s in zip(self.kernel_size, self.stride)
        )

    def _pad_input(self, x: torch.Tensor, pad: Tuple[int]) -> torch.Tensor:
        # Reverse order and split each dim into (before, after)
        pad_list = []
        for p in reversed(pad):
            pad_list.extend([p // 2, p - p // 2])

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
