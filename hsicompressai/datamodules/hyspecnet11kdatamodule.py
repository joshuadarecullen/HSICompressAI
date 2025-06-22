from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule

import albumentations as A

import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

from matplotlib.figure import Figure

from hsicompressai.datasets import HySpecNet11k
from hsicompressai.registry import register_pldatamodule

# print("DEBUG:", register_pldatamodule)  # ← Add this line
# """
# Configure function that hydra uses
# """

# @register_pldatamodule("HySpecNet11k")
class HySpecNet11kDataModule(NonGeoDataModule):
    def __init__(
        self,
        dataset_class: NonGeoDataset,# HySpecNet11k,
        data_root: str,
        dataset_mode: str = "easy",
        batch_size: int = 1,
        num_workers: int = 0,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
        ) -> None:

        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

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


    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(self.data_root,
                                           mode=dataset_mode,
                                           split="train",
                                           transform=self.aug)
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(self.data_root,
                                         mode=self.dataset_mode,
                                         split="val",
                                         transform=self.aug)
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(self.data_dir,
                                          mode=self.dataset_mode,
                                          split="test",
                                          transform=self.aug)
        if stage in ["predict"]:
            self.test_dataset = self.dataset_class(self.data_dir,
                                          mode=self.dataset_mode,
                                          split="test",
                                          transform=self.aug)
