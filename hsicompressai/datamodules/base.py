# lightning_wrappers.py
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Callable
from hsicompressai.registry import register_module


@register_module("GenericTrainer")
class GenericModelTrainer(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable = F.mse_loss,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
