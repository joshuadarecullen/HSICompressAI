"""Neural compression models for hyperspectral images.

This module contains various deep learning-based compression architectures
specifically designed for hyperspectral image data.

Models:
    ConvolutionalAutoencoder1D: 1D convolutional autoencoder for spectral compression
    ModifiedConvolutionalAutoencoder1D: Enhanced version of the 1D CAE
    ConvolutionalAutoencoder3D: 3D convolutional autoencoder for spatial-spectral compression
    HyperspectralCompressionTransformer: Transformer-based compression model
    SpectralSignalsCompressorNetwork: Network for compressing spectral signals
    ScalableReduceComplexityEntropyModel: Entropy-based compression with reduced complexity
    
Example:
    >>> from hsicompressai.models.neural import ConvolutionalAutoencoder1D
    >>> model = ConvolutionalAutoencoder1D(src_channels=103)
    >>> compressed = model.compress(hyperspectral_data)
    >>> reconstructed = model.decompress(compressed)
"""

from .cae1d import ConvolutionalAutoencoder1D
from .cae1dm import ModifiedConvolutionalAutoencoder1D
from .cae3d import ConvolutionalAutoencoder3D
from .hycot import HyperspectralCompressionTransformer
# from .mambacomp import MambaHSICompression
from .sscnet import SpectralSignalsCompressorNetwork
from .srcmodel import ScalableReduceComplexityEntropyModel


__all__ = [
    "ConvolutionalAutoencoder1D",
    "ConvolutionalAutoencoder3D",
    "HyperspectralCompressionTransformer",
    # "MambaHSICompression",
    "ModifiedConvolutionalAutoencoder1D",
    "SpectralSignalsCompressorNetwork",
    "ScalableReduceComplexityEntropyModel",
    ]
