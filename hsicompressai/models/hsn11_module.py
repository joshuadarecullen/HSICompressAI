"""HSN11 Lightning Module for hyperspectral image compression.

This module provides a PyTorch Lightning wrapper for hyperspectral compression models,
specifically designed for the HySpecNet-11k dataset. It handles training, validation,
and testing workflows with support for both custom loss functions and built-in model losses.

Classes:
    HSN11LitModule: Main Lightning module for hyperspectral compression
    
Example:
    >>> from hsicompressai.models import HSN11LitModule
    >>> from hsicompressai.models.neural import CAE1DModule
    >>> net = CAE1DModule()
    >>> model = HSN11LitModule(
    ...     net=net,
    ...     optimizer=torch.optim.Adam,
    ...     scheduler=None,
    ...     criterion=None,
    ...     state_dict={},
    ...     compile=False
    ... )
"""

from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from hsicompressai.registry import register_plmodule


@register_plmodule("HySpecNet11k")
class HSN11LitModule(LightningModule):
    """PyTorch Lightning module for hyperspectral image compression.
    
    This module provides a standardized interface for training, validating, and testing
    hyperspectral compression models. It supports both external loss functions and
    models with built-in loss computation.
    
    Attributes:
        net: The underlying compression model
        criterion: Optional external loss function
        train_loss: Training loss metric tracker
        val_loss: Validation loss metric tracker  
        test_loss: Test loss metric tracker
        
    Note:
        This module is registered as "HySpecNet11k" in the PyTorch Lightning registry.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Union[torch.nn.Module, None],
        state_dict: Dict[str, Tensor],
        compile: bool,
    ) -> None:
        """Initialize the HSN11 Lightning Module.
        
        Args:
            net: The compression model to train (e.g., CAE1D, CAE3D, etc.)
            optimizer: Optimizer class to use for training
            scheduler: Learning rate scheduler class
            criterion: Optional external loss function. If None, uses model's built-in loss
            state_dict: Pre-trained model state dictionary
            compile: Whether to compile the model with torch.compile for optimization
            
        Note:
            If criterion is None, the model is expected to have a loss() method
            that returns a dictionary with at least a 'loss' key.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = criterion if criterion else None

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of recon.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, x: Tuple[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data and compute
            the torch loss or custom

        :param batch: A batch of data (a tuple) containing the input tensor of HSI.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of reconstructions.
        """

        if self.criterion:
            x_hat = self(x)
            loss = self.criterion(x_hat, x)
        else:
            outputs = self(x)
            loss = self.net.loss(outputs, x)["loss"]
            x_hat = outputs["x_hat"]

        return loss, x_hat

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between input and reconstructions.
        """
        # print(torch.cuda.memory_summary(device=0, abbreviated=True))
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return {'loss': loss,
                'x_hat': x_hat}

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss,
                'x_hat': x_hat}

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss,
                'x_hat': x_hat}

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
