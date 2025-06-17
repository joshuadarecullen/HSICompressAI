from typing import Any, Dict, List

import torch.nn as nn

from torch import Tensor

__all__ = [
    "LatentCodec",
]


class LatentCodec(nn.Module):
    def forward(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def compress(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def decompress(
        self, strings: List[List[bytes]], shape: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError

