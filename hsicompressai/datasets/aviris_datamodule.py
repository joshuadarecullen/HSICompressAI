"""AVIRIS Lightning DataModule for hyperspectral image compression training.

This module provides a PyTorch Lightning DataModule for loading and processing AVIRIS
hyperspectral imagery data in a format suitable for training neural compression models.

Classes:
    AVIRISDataModule: Lightning DataModule for AVIRIS hyperspectral data

Example:
    >>> datamodule = AVIRISDataModule(
    ...     data_dir="path/to/aviris/data",
    ...     batch_size=4,
    ...     num_workers=2
    ... )
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()
"""

from typing import Any, Dict, Optional, Tuple, Callable
from torch import Tensor

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .aviris_dataset import AVIRISTiffDataset


class AVIRISDataModule(LightningDataModule):
    """Lightning DataModule for AVIRIS hyperspectral imagery.
    
    This DataModule provides a standardized interface for loading AVIRIS hyperspectral
    data for training, validation, and testing of neural compression models. It handles
    data loading, preprocessing, and batch creation.
    
    The module implements the standard Lightning DataModule interface with methods for:
    - Data preparation and setup
    - Train/validation/test dataloaders
    - State management for checkpointing
    
    Attributes:
        data_dir (str): Path to the AVIRIS data directory
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory in DataLoader
        transform (callable): Optional transform to apply to samples
        
    Example:
        >>> # Basic usage
        >>> datamodule = AVIRISDataModule(data_dir="data/aviris/", batch_size=4)
        >>> datamodule.setup()
        >>> 
        >>> # With custom transforms
        >>> transform = lambda x: x * 0.5  # Example transform
        >>> datamodule = AVIRISDataModule(
        ...     data_dir="data/aviris/",
        ...     batch_size=8,
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:
        """Initialize AVIRIS DataModule.
        
        Args:
            data_dir (str, optional): Path to the AVIRIS data directory. 
                Defaults to "data/".
            train_val_test_split (Tuple[float, float, float], optional): 
                Proportions for train/validation/test splits. Defaults to (0.6, 0.2, 0.2).
            batch_size (int, optional): Batch size for data loading. Defaults to 1.
            num_workers (int, optional): Number of worker processes for data loading. 
                Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory in DataLoader for 
                faster GPU transfer. Defaults to False.
            transform (callable, optional): Transform to apply to samples. 
                Defaults to None.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.data_dir = data_dir
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Number of classes in dataset.
        
        Returns:
            int: Always returns 0 as this is a self-supervised compression task.
        """
        return 0

    def prepare_data(self) -> None:
        """Prepare data for use.
        
        Called only once and on single GPU. Used for downloading or 
        preprocessing that should be done once.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.
        
        Creates dataset instances and handles distributed training considerations.
        Called on every process in distributed training.
        
        Args:
            stage (str, optional): The stage being set up. Can be "fit", "validate", 
                "test", or "predict". If None, sets up all stages. Defaults to None.
                
        Raises:
            RuntimeError: If batch size is not divisible by number of devices.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not\
                            divisible by the number of devices\
                            ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_data and not self.val_data and not self.test_data:

            # self.train_data = AVIRISTiffDataset(self.data_dir,
            #                                     split="train",
            #                                     transform=None)

            # self.val_data = AVIRISTiffDataset(self.data_dir,
            #                                   split="val",
            #                                   transform=None)

            self.test_data = AVIRISTiffDataset(self.data_dir,
                                               split="test",
                                               transform=None)

            # dataset = ConcatDataset(datasets=[trainset, valset, testset])
            # self.train_data, self.val_data, self.test_data = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the training dataloader.
        
        Returns:
            DataLoader: Training dataloader with shuffling enabled.
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader without shuffling.
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.
        
        Returns:
            DataLoader: Test dataloader without shuffling.
        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training/validation/testing.
        
        Lightning hook for cleaning up resources after training stages complete.
        
        Args:
            stage (str, optional): The stage being torn down. Can be "fit", 
                "validate", "test", or "predict". Defaults to None.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Get datamodule state for checkpointing.
        
        Returns:
            Dict[Any, Any]: Dictionary containing datamodule state to save.
                Currently returns empty dict as no persistent state needed.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load datamodule state from checkpoint.
        
        Args:
            state_dict (Dict[str, Any]): The datamodule state dictionary
                returned by state_dict().
        """
        pass


if __name__ == "__main__":
    _ = HSN11DataModule()
