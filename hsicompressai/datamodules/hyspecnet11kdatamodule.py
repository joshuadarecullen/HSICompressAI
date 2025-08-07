"""HySpecNet-11k Lightning DataModule for hyperspectral image compression.

This module provides a PyTorch Lightning DataModule for the HySpecNet-11k dataset,
designed specifically for training neural hyperspectral image compression models.
It extends TorchGeo's NonGeoDataModule with support for data augmentation and 
different training modes.

Classes:
    HySpecNet11kDataModule: Lightning DataModule for HySpecNet-11k dataset

Example:
    >>> # Basic usage
    >>> datamodule = HySpecNet11kDataModule(
    ...     data_root="data/hyspecnet-11k/",
    ...     dataset_mode="easy",
    ...     batch_size=8,
    ...     num_workers=4
    ... )
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()
    
    >>> # With data augmentation
    >>> import kornia.augmentation as K
    >>> aug = K.AugmentationSequential(
    ...     K.RandomHorizontalFlip(p=0.5),
    ...     K.RandomVerticalFlip(p=0.5)
    ... )
    >>> datamodule = HySpecNet11kDataModule(
    ...     data_root="data/hyspecnet-11k/",
    ...     aug=aug
    ... )
"""

from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule

import albumentations as A

import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

from hsicompressai.datasets import HySpecNet11k
from hsicompressai.registry import register_pldatamodule

@register_pldatamodule("HySpecNet11k")
class HySpecNet11kDataModule(NonGeoDataModule):
    """Lightning DataModule for HySpecNet-11k hyperspectral dataset.
    
    This DataModule provides a standardized interface for loading the HySpecNet-11k
    dataset for training hyperspectral image compression models. It extends TorchGeo's 
    NonGeoDataModule with specific configurations for hyperspectral data processing.
    
    The module supports multiple dataset difficulty modes (easy, hard, mini) and 
    provides data augmentation capabilities using both Albumentations and Kornia.
    
    Attributes:
        data_root (str): Root path to the HySpecNet-11k dataset
        dataset_class (type): Dataset class to instantiate (defaults to HySpecNet11k)
        dataset_mode (str): Dataset difficulty mode ("easy", "hard", "mini")
        batch_size (int): Batch size for all dataloaders
        num_workers (int): Number of worker processes for data loading
        aug (AugmentationSequential): Kornia augmentation pipeline
        
    Example:
        >>> # Basic setup
        >>> datamodule = HySpecNet11kDataModule(
        ...     data_root="data/hyspecnet-11k/",
        ...     dataset_mode="easy",
        ...     batch_size=16
        ... )
        >>> datamodule.setup("fit")
        >>> 
        >>> # Get training data
        >>> for batch in datamodule.train_dataloader():
        ...     print(batch.shape)  # (batch_size, bands, height, width)
        ...     break
    """
    def __init__(
        self,
        data_root: str,
        dataset_class: Optional[type] = None,  # Accept class, not instance
        dataset_mode: str = "easy",
        batch_size: int = 1,
        num_workers: int = 0,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HySpecNet-11k DataModule.
        
        Args:
            data_root (str): Root directory path containing the HySpecNet-11k dataset.
            dataset_class (type, optional): Custom dataset class to use. If None,
                defaults to HySpecNet11k. Defaults to None.
            dataset_mode (str, optional): Dataset difficulty mode. Options are 
                "easy", "hard", or "mini". Defaults to "easy".
            batch_size (int, optional): Batch size for all dataloaders. 
                Defaults to 1.
            num_workers (int, optional): Number of worker processes for data loading.
                Defaults to 0.
            transform (A.Compose | None | list[A.BasicTransform], optional): 
                Albumentations transforms to apply. Defaults to None.
            aug (AugmentationSequential, optional): Kornia augmentation pipeline
                for data augmentation. Defaults to None.
            **kwargs: Additional arguments passed to parent NonGeoDataModule.
        """
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        self.dataset_class = dataset_class or HySpecNet11k  # default fallback
        self.data_root = data_root
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Data loaders
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.test_batch_size = batch_size
        self.predict_batch_size = batch_size

        # Data augmentation
        self.aug = transform

        self.train_aug = aug
        self.val_aug = aug
        self.test_aug = aug
        self.predict_aug = aug

        # Initialize dataset placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use.
        
        Called only once and on single process. Used for downloading or 
        preprocessing that should be done once. Currently no preprocessing
        is required for HySpecNet-11k.
        """
        pass  # Optional: download or preprocess

    def setup(self, stage: str = None) -> None:
        """Set up datasets for training, validation, and testing.
        
        Called on every process in distributed training. Creates dataset 
        instances for the specified stage.
        
        Args:
            stage (str, optional): Training stage. Can be "fit", "validate", 
                "test", or "predict". If None, sets up for all stages.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                self.data_root, mode=self.dataset_mode, split="train", transform=self.aug
            )
            self.val_dataset = self.dataset_class(
                self.data_root, mode=self.dataset_mode, split="val", transform=self.aug
            )
        elif stage == "validate":
            self.val_dataset = self.dataset_class(
                self.data_root, mode=self.dataset_mode, split="val", transform=self.aug
            )
        elif stage == "test":
            self.test_dataset = self.dataset_class(
                self.data_root, mode=self.dataset_mode, split="test", transform=self.aug
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_class(
                self.data_root, mode=self.dataset_mode, split="test", transform=self.aug
            )
