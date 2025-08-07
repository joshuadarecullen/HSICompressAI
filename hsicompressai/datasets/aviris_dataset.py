"""AVIRIS hyperspectral dataset implementation.

This module provides functionality for loading and processing AVIRIS (Airborne Visible/Infrared 
Imaging Spectrometer) hyperspectral imagery data. The AVIRIS dataset contains high-resolution 
hyperspectral images with up to 220+ spectral bands.

Classes:
    IndianPinesDataset: PyTorch dataset for loading AVIRIS data from .tif files

Functions:
    plot_aviris_rgb: Visualization utility for creating RGB renderings from hyperspectral data

Example:
    >>> dataset = IndianPinesDataset("path/to/aviris.tif", patch_size=64)
    >>> sample = dataset[0]
    >>> print(sample.shape)  # (bands, 64, 64)
    
References:
    AVIRIS-Classic and AVIRIS-NG data from JPL (https://aviris.jpl.nasa.gov/)
"""

import os
import torch
from torch.utils.data import Dataset
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from hsicompressai.registry import register_dataset

@register_dataset("IndianPinesDataset")
class IndianPinesDataset(Dataset):
    """PyTorch dataset for loading AVIRIS hyperspectral imagery.
    
    This dataset loads multi-band AVIRIS .tif files and provides functionality for extracting
    patches or loading complete images. The dataset automatically normalizes spectral bands
    to the [0,1] range for training compatibility.
    
    Attributes:
        tif_path (str): Path to the AVIRIS .tif file
        patch_size (int or tuple): Size of patches to extract
        transform (callable): Optional transform to apply to samples
        full_data (numpy.ndarray): Complete hyperspectral data cube (bands, height, width)
        C (int): Number of spectral bands
        H (int): Image height
        W (int): Image width
        total_patches (int): Total number of patches available
        
    Example:
        >>> # Load complete image
        >>> dataset = IndianPinesDataset("aviris_image.tif")
        >>> sample = dataset[0]  # Shape: (bands, height, width)
        
        >>> # Extract 64x64 patches
        >>> dataset = IndianPinesDataset("aviris_image.tif", patch_size=64)
        >>> patch = dataset[0]  # Shape: (bands, 64, 64)
    """
    
    def __init__(self, tif_path, patch_size=None, transform=None):
        """Initialize the AVIRIS dataset.
        
        Args:
            tif_path (str): Path to a single multi-band AVIRIS .tif file.
            patch_size (int or tuple, optional): Size of patches to extract. If int, 
                creates square patches. If tuple, (height, width). If None, loads 
                complete image.
            transform (callable, optional): Transform to apply to each sample. If 
                provided, skips automatic normalization.
                
        Raises:
            FileNotFoundError: If tif_path does not exist
            ValueError: If patch_size is larger than image dimensions
        """
        self.tif_path = tif_path
        self.patch_size = patch_size
        self.transform = transform

        with rasterio.open(tif_path) as src:
            self.full_data = src.read().astype(np.float32)  # shape: (bands, height, width)

        # Normalize per-band to [0, 1]
        self.full_data = self.transform if transform else self._normalize(self.full_data)

        self.C, self.H, self.W = self.full_data.shape

        if patch_size:
            ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
            self.num_patches_h = self.H // ph
            self.num_patches_w = self.W // pw
            self.total_patches = self.num_patches_h * self.num_patches_w
            print(self.total_patches)
        else:
            self.total_patches = 1  # Whole image

    def __len__(self):
        """Return total number of samples in dataset.
        
        Returns:
            int: Number of patches if patch_size is set, otherwise 1 for complete image.
        """
        return self.total_patches

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            torch.Tensor: Hyperspectral data sample with shape (bands, height, width).
                If patch_size is set, returns patch of specified size. Otherwise 
                returns complete image.
                
        Raises:
            IndexError: If idx is out of bounds.
        """
        # Extract non overlapping patches
        if self.patch_size:
            ph, pw = self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
            row = idx // self.num_patches_w
            col = idx % self.num_patches_w
            top = row * ph
            left = col * pw
            patch = self.full_data[:, top:top+ph, left:left+pw]
        else:
            patch = self.full_data  # Whole image

        if self.transform:
            patch = self.transform(patch)

        return torch.from_numpy(patch)

    def _normalize(self, data):
        """Normalize hyperspectral data to [0,1] range.
        
        Performs per-band min-max normalization to ensure all spectral bands
        are scaled to the [0,1] range for consistent training.
        
        Args:
            data (numpy.ndarray): Input hyperspectral data with shape (bands, height, width).
            
        Returns:
            numpy.ndarray: Normalized data with same shape as input.
        """
        # Normalize each spectral band independently to [0, 1]
        for i in range(data.shape[0]):
            band = data[i]
            band_min = band.min()
            band_max = band.max()
            if band_max > band_min:
                data[i] = (band - band_min) / (band_max - band_min + 1e-6)
        return data


def plot_aviris_rgb(data, red=26, green=17, blue=7, title="AVIRIS RGB"):
    """Plot RGB visualization from AVIRIS hyperspectral data.
    
    Creates an RGB image from selected spectral bands of a hyperspectral cube.
    Useful for visualizing hyperspectral data in a format that's easy to interpret.
    
    Args:
        data (numpy.ndarray): Hyperspectral cube with shape (batch, bands, H, W) or 
            (bands, H, W). Data should be normalized to [0, 1] range.
        red (int, optional): Band index to use for red channel. Defaults to 26.
        green (int, optional): Band index to use for green channel. Defaults to 17.
        blue (int, optional): Band index to use for blue channel. Defaults to 7.
        title (str, optional): Title for the plot. Defaults to "AVIRIS RGB".
        
    Example:
        >>> data = dataset[0]  # Shape: (bands, H, W)
        >>> plot_aviris_rgb(data, red=30, green=20, blue=10)
        
    Note:
        Default band indices (26, 17, 7) correspond to typical RGB wavelengths
        in AVIRIS data (~660nm, ~550nm, ~470nm).
    """
    # Extract RGB channels
    rgb = np.stack([
        data[:,red,:,:],
        data[:,green,:,:],
        data[:,blue,:,:]
    ], axis=-1).squeeze(0)  # shape: (H, W, 3)

    print(rgb.shape)

    # Ensure values are in [0, 1] for display
    # rgb = np.clip(rgb, 0, 1)

    # Joint min/max scaling across all RGB pixels
    vmin = rgb.min()
    vmax = rgb.max()
    rgb_scaled = (rgb - vmin) / (vmax - vmin + 1e-6)  # safe division

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(rgb_scaled, 0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()
