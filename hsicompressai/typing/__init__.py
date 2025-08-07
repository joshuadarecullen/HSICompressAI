from typing import Callable

from .torch import (
    # Basic PyTorch types
    TCriterion,
    TDataLoader,
    TDataset,
    TModel,
    TModule,
    TOptimizer,
    TScheduler,
    # PyTorch Lightning base types
    PLModule,
    PLDataModule,
    PLCallback,
    PLTrainer,
    PLLogger,
    # Generic type variables
    T,
    PLModuleT,
    PLDataModuleT,
    # Lightning step outputs
    TStepOutput,
    # Protocols
    LightningStepProtocol,
    LightningHookProtocol,
    # Configuration types
    TOptimizerConfig,
    TLRSchedulerConfig,
    # HSI-specific types
    THSIBatch,
    THSIStepOutput,
)

__all__ = [
    # Basic PyTorch types
    "TCriterion",
    "TDataLoader",
    "TDataset",
    "TModel",
    "TModule",
    "TOptimizer",
    "TScheduler",
    "TTransform",
    # PyTorch Lightning base types
    "PLModule",
    "PLDataModule",
    "PLCallback",
    "PLTrainer",
    "PLLogger",
    # Generic type variables
    "T",
    "PLModuleT",
    "PLDataModuleT",
    # Lightning step outputs
    "TStepOutput",
    # Protocols
    "LightningStepProtocol",
    "LightningHookProtocol",
    # Configuration types
    "TOptimizerConfig",
    "TLRSchedulerConfig",
    # HSI-specific types
    "THSIBatch",
    "THSIStepOutput",
]

TTransform = Callable
