import os
import torch
from torch.utils.data import Dataset
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from hsicompressai.registry import register_dataset

@register_dataset("IndianPinesDataset")
class IndianPinesDataset(Dataset):
    def __init__(self, tif_path, patch_size=None, transform=None):
        """
        Args:
            tif_path (str): Path to a single multi-band AVIRIS .tif file.
            patch_size (int or tuple, optional): Extracts random patches if set.
            transform (callable, optional): Transform to apply to each sample.
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
        return self.total_patches

    def __getitem__(self, idx):
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
        # Normalize each spectral band independently to [0, 1]
        for i in range(data.shape[0]):
            band = data[i]
            band_min = band.min()
            band_max = band.max()
            if band_max > band_min:
                data[i] = (band - band_min) / (band_max - band_min + 1e-6)
        return data


def plot_aviris_rgb(data, red=26, green=17, blue=7, title="AVIRIS RGB"):
    """
    Plots an RGB image from a normalized AVIRIS hyperspectral cube.

    Args:
        data (np.ndarray): Hyperspectral cube of shape (bands, H, W), normalized to [0, 1].
        red, green, blue (int): Band indices to use for RGB.
        title (str): Title for the plot.
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
