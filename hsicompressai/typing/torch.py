from typing import Dict, Union

import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, Dataset

TCriterion = nn.Module
TDataLoader = DataLoader
TDataset = Dataset
TModel = nn.Module
TModule = nn.Module
TOptimizer = Union[Optimizer, Dict[str, Optimizer]]
TScheduler = Union[ReduceLROnPlateau, _LRScheduler]
