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

# print("DEBUG:", register_pldatamodule)  # ← Add this line
# """
# Configure function that hydra uses
# """


class HySpecNet11kDataModule(NonGeoDataModule):
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
        pass  # Optional: download or preprocess

    def setup(self, stage: str = None) -> None:
        """Set up datasets for different stages."""
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
# # @register_pldatamodule("HySpecNet11k")
# class HySpecNet11kDataModule(NonGeoDataModule):
#     def __init__(
#         self,
#         data_root: str,
#         dataset_class: Optional[NonGeoDataset] = None,# HySpecNet11k,
#         dataset_mode: str = "easy",
#         batch_size: int = 1,
#         num_workers: int = 0,
#         transform: A.Compose | None | list[A.BasicTransform] = None,
#         aug: AugmentationSequential = None,
#         **kwargs: Any,
#         ) -> None:

#         super().__init__(dataset_class, batch_size, num_workers, **kwargs)

#         self.data_root = data_root
#         self.dataset_mode = dataset_mode
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         # Data loaders
#         self.train_batch_size = batch_size
#         self.val_batch_size = batch_size
#         self.test_batch_size = batch_size
#         self.predict_batch_size = batch_size

#         # Data augmentation
#         self.aug = transform

#         self.train_aug = aug
#         self.val_aug = aug
#         self.test_aug = aug
#         self.predict_aug = aug

#     def prepare_data(self) -> None:
#         pass

#     def setup(self) -> None:
#         """Set up datasets.

#         Args:
#             stage: Either fit, validate, test, or predict.
#         """
#         # load and split datasets only if not loaded already
#         if not self.train_dataset and not self.val_dataset and not self.test_dataset:

#             self.train_dataset = HySpecNet11k(self.data_root,
#                                            mode=self.dataset_mode,
#                                            split="train",
#                                            transform=None)

#             self.val_dataset = HySpecNet11k(self.data_root,
#                                          mode=self.dataset_mode,
#                                          split="val",
#                                          transform=None)

#             self.test_dataset = HySpecNet11k(self.data_root,
#                                           mode=self.dataset_mode,
#                                           split="test",
#                                           transform=None)
