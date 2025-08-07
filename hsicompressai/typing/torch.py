from typing import Any, Dict, Optional, Protocol, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT

# Basic PyTorch types
TCriterion = nn.Module
TDataLoader = DataLoader
TDataset = Dataset
TModel = nn.Module
TModule = nn.Module
TOptimizer = Union[Optimizer, Dict[str, Optimizer]]
TScheduler = Union[ReduceLROnPlateau, _LRScheduler]

# PyTorch Lightning base types
PLModule = LightningModule
PLDataModule = LightningDataModule
PLCallback = Callback
PLTrainer = Trainer
PLLogger = Logger

# Generic type variables for Lightning modules
T = TypeVar('T')
PLModuleT = TypeVar('PLModuleT', bound=LightningModule)
PLDataModuleT = TypeVar('PLDataModuleT', bound=LightningDataModule)

# Lightning step outputs
TStepOutput = STEP_OUTPUT

# Common Lightning method signatures
class LightningStepProtocol(Protocol):
    """Protocol for Lightning step methods."""
    def __call__(self, batch: Any, batch_idx: int) -> TStepOutput: ...

class LightningHookProtocol(Protocol):
    """Protocol for Lightning hook methods."""
    def __call__(self) -> None: ...

# Optimizer and scheduler configuration types
TOptimizerConfig = Dict[str, Union[Optimizer, Dict[str, Any]]]
TLRSchedulerConfig = Dict[str, Union[_LRScheduler, Dict[str, Any]]]

# Common batch types for HSI applications
THSIBatch = Union[Tensor, tuple[Tensor, ...], Dict[str, Tensor]]
THSIStepOutput = Dict[str, Union[Tensor, float]]
