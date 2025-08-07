"""Models module for HSICompressAI.

This module contains various compression models for hyperspectral images,
organized into conventional and neural approaches.

Submodules:
    conventional: Traditional compression methods (e.g., KLT+JPEG)
    neural: Deep learning-based compression models
    
Classes:
    HSN11LitModule: PyTorch Lightning module wrapper for HySpecNet-11k models
    
Example:
    >>> from hsicompressai.models import HSN11LitModule
    >>> from hsicompressai.models.neural import CAE1DModule
    >>> model = HSN11LitModule(net=CAE1DModule(), ...)
"""

from . import conventional
from . import neural
from .hsn11_module import HSN11LitModule

__all__ = [
        "conventional",
        "neural",
        "HSN11LitModule"
        ]
