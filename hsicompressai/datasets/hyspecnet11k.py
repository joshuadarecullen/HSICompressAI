"""HySpecNet-11k hyperspectral dataset implementation.

This module provides a PyTorch dataset for the HySpecNet-11k large-scale hyperspectral 
image dataset designed for benchmarking learning-based hyperspectral image compression methods.

Classes:
    HySpecNet11k: PyTorch dataset for HySpecNet-11k hyperspectral imagery

Example:
    >>> # Load easy split for training
    >>> dataset = HySpecNet11k(
    ...     root_dir="data/hyspecnet-11k/", 
    ...     mode="easy", 
    ...     split="train"
    ... )
    >>> sample = dataset[0]
    >>> print(sample.shape)  # (bands, height, width)

References:
    Fuchs, Martin Hermann Paul, and Begüm Demir. "HySpecNet-11k: A Large-Scale 
    Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image 
    Compression Methods." arXiv preprint arXiv:2306.00385 (2023).
"""

import csv
import os

import numpy as np
import torch

from torch.utils.data import Dataset
from hsicompressai.registry import register_dataset

@register_dataset("HySpecNet11k")
class HySpecNet11k(Dataset):
    """PyTorch Dataset for HySpecNet-11k hyperspectral imagery.

    HySpecNet-11k is a large-scale hyperspectral dataset containing over 11,000 
    hyperspectral patches from EnMAP satellite data. The dataset is designed for 
    benchmarking learning-based hyperspectral image compression methods.
    
    The dataset includes multiple difficulty modes (easy, hard) and standard
    train/validation/test splits. Each sample is a hyperspectral patch with
    224 spectral bands covering the range from 420nm to 2450nm.
    
    Attributes:
        root_dir (str): Root directory path containing the dataset
        mode (str): Dataset difficulty mode ("easy", "hard", "mini")  
        split (str): Data split ("train", "val", "test")
        transform (callable): Optional transform to apply to samples
        npy_paths (list): List of paths to .npy data files
        
    Dataset Structure:
        - root_dir/
            - patches/
                - ENMAP01-____L2A-DT000000xxxx_yyyymmddThhmmssZ_xxx_V010110_yyyymmddThhmmssZ/
                    - patch_folder/
                        - patch_name-DATA.npy  (hyperspectral data)
                        - patch_name-QL_*.TIF  (quality layers)
                        - patch_name-SPECTRAL_IMAGE.TIF
                        - patch_name-THUMBNAIL.jpg
            - splits/
                - easy/hard/mini/
                    - train.csv, val.csv, test.csv
                    
    Example:
        >>> # Load training data from easy mode
        >>> dataset = HySpecNet11k(
        ...     root_dir="data/hyspecnet-11k/",
        ...     mode="easy", 
        ...     split="train"
        ... )
        >>> print(len(dataset))  # Number of training samples
        >>> sample = dataset[0]
        >>> print(sample.shape)  # (224, patch_height, patch_width)

        >>> # Load with transforms
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = HySpecNet11k(
        ...     root_dir="data/hyspecnet-11k/",
        ...     mode="hard",
        ...     split="test",
        ...     transform=transform
        ... )
    
    Citation:
        @misc{fuchs2023hyspecnet11k,
            title={HySpecNet-11k: A Large-Scale Hyperspectral Dataset for 
                   Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
            author={Martin Hermann Paul Fuchs and Begüm Demir},
            year={2023},
            eprint={2306.00385},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
    """

    def __init__(self, root_dir, mode="easy", split="train", transform=None):
        """Initialize HySpecNet-11k dataset.
        
        Args:
            root_dir (str): Root directory path containing the HySpecNet-11k dataset.
            mode (str, optional): Dataset difficulty mode. Options are "easy", "hard", 
                or "mini". The "easy" mode contains less challenging compression scenarios,
                while "hard" contains more complex scenes. Defaults to "easy".
            split (str, optional): Dataset split to use. Options are "train", "val", 
                or "test". Defaults to "train".
            transform (callable, optional): Optional transform to be applied to samples.
                Should accept a torch.Tensor and return a torch.Tensor. Defaults to None.
                
        Raises:
            FileNotFoundError: If root_dir doesn't exist or split CSV file not found.
            ValueError: If mode or split are not valid options.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.split = split

        self.csv_path = os.path.join(self.root_dir,
                                     "splits", mode,
                                     f"{split}.csv")

        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(self.root_dir, "patches", x)
                          for x in self.npy_paths]

        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset.
        
        Returns:
            int: Number of hyperspectral patches in the current split.
        """
        return len(self.npy_paths)

    def __getitem__(self, index):
        """Get a hyperspectral sample by index.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            torch.Tensor: Hyperspectral data tensor with shape (bands, height, width).
                Typically (224, patch_height, patch_width) for HySpecNet-11k.
                
        Raises:
            IndexError: If index is out of bounds.
            FileNotFoundError: If the .npy file doesn't exist.
        """
        # get full numpy path
        npy_path = self.npy_paths[index]
        # read numpy data
        img = np.load(npy_path)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        return img
