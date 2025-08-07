"""HSICompressAI - A PyTorch Lightning-based framework for hyperspectral image compression.

This package provides comprehensive tools and models for compressing hyperspectral images
using various deep learning approaches including convolutional autoencoders, transformers,
and state-of-the-art architectures like Mamba.

Key Features:
    - Multiple neural compression models (CAE1D, CAE3D, HyCOT, SrcModel, etc.)
    - Hyperspectral-specific datasets and data loading
    - Custom losses, metrics, and callbacks for compression evaluation
    - Extensible registry system for easy model/transform registration
    - CLI interface for training and evaluation
    - Support for various hyperspectral formats and transformations

Example:
    >>> import hsicompressai as hsica
    >>> from hsicompressai.models.neural import CAE1DModule
    >>> model = CAE1DModule()

Author: [Your Name]
Version: 1.0.0
License: [Your License]
"""

from hsicompressai import (
    callbacks,
    datamodules,
    datasets,
    entropy_models,
    latent_codecs,
    layers,
    losses,
    metrics,
    models,
    ops,
    optimizers,
    registry,
    transforms,
    typing,
    utils,
    zoo,
)


__all__ = [
    "callbacks",
    "datamodules",
    "datasets",
    "entropy_models",
    "latent_codecs",
    "layers",
    "losses",
    "metrics",
    "models",
    "ops",
    "optimizers",
    "registry",
    "transforms",
    "typing",
    "utils",
    "zoo",
    ]
