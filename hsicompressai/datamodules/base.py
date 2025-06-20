import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning import LightningDataModule
from typing import Callable
from hsicompressai.registry import register_pldatamodule


@register_pldatamodule("base")
class BaseModule(LightningDataModule):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
