from typing import Dict

import torch
from torch import Tensor
from torchmetrics import MeanMetric
from pytorch_lightning import Callback, Trainer, LightningModule

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hsicompressai.metrics.psnr import PeakSignalToNoiseRatio
from hsicompressai.metrics.sa import SpectralAngle
from hsicompressai.metrics.ssim import StructuralSimilarity


class Metrics(Callback):
    def __init__(self):
        super().__init__()
        """
        A callback to compute the Peak Signal to Noise Ratio (PSNR)
        on the indvidual bands of the hyperspectral image
        """

        self.psnr = PeakSignalToNoiseRatio()
        self.ssim = StructuralSimilarity()
        self.sa = SpectralAngle()

    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule) -> None:

        self.psnr_train = MeanMetric()
        self.sa_train = MeanMetric()
        self.ssim_train = MeanMetric()

        self.psnr_val = MeanMetric()
        self.sa_val = MeanMetric()
        self.ssim_val = MeanMetric()

        self.move_to_device(pl_module)

    def on_test_start(self,
                      trainer: Trainer,
                      pl_module: LightningModule) -> None:

        self.psnr_test = MeanMetric()
        self.sa_test = MeanMetric()
        self.ssim_test = MeanMetric()

        self.move_to_device(pl_module)

    def on_train_start(self,
                       trainer: Trainer,
                       pl_module: LightningModule) -> None:

        self.psnr_val.reset()
        self.ssim_val.reset()
        self.sa_val.reset()

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: Dict[str, Tensor],
                           batch_idx: int) -> None:

        metrics = self.generate_metrics(outputs['x_hat'].detach(), batch)

        self.psnr_train.update(metrics['psnr'])
        self.ssim_train.update(metrics['ssim'])
        self.sa_train.update(metrics['sa'])

    def on_train_epoch_end(self, trainer, pl_module):
        """ Compute and log final epoch metrics for training """
        pl_module.log("train/psnr", self.psnr_train.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("train/ssim", self.ssim_train.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("train/sa", self.sa_train.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset the metrics after logging to prepare for the next epoch
        self.psnr_train.reset()
        self.ssim_train.reset()
        self.sa_train.reset()

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: Dict[str, Tensor],
                                batch_idx: int,
                                ) -> None:

        metrics = self.generate_metrics(outputs['x_hat'].detach(), batch)

        self.psnr_val.update((metrics['psnr']))
        self.ssim_val.update((metrics['ssim']))
        self.sa_val.update((metrics['sa']))

    def on_validation_epoch_end(self, trainer, pl_module):
        """ Compute and log final epoch metrics for validation """
        pl_module.log("val/psnr", self.psnr_val.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("val/ssim", self.ssim_val.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("val/sa", self.sa_val.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)

        self.psnr_val.reset()
        self.ssim_val.reset()
        self.sa_val.reset()

    def on_test_batch_end(self,
                          trainer: Trainer,
                          pl_module: LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: Dict[str, Tensor],
                          batch_idx: int,
                          ) -> None:

        metrics = self.generate_metrics(outputs['x_hat'].detach(), batch)

        self.psnr_test.update((metrics['psnr']))
        self.ssim_test.update((metrics['ssim']))
        self.sa_test.update((metrics['sa']))

    def on_test_epoch_end(self,
                          trainer: Trainer,
                          pl_module: LightningModule) -> None:
        """ Compute and log final epoch metrics for testing """
        pl_module.log("test/psnr", self.psnr_test.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("test/ssim", self.ssim_test.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log("test/sa", self.sa_test.compute(), on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)

        self.psnr_test.reset()
        self.ssim_test.reset()
        self.sa_test.reset()

    def move_to_device(self,
                       pl_module: LightningModule) -> None:
        """Move all metrics to the correct device when training starts"""
        device = pl_module.device
        for metric in vars(self).values():
            if isinstance(metric, torch.nn.Module):
                metric.to(device)

    def generate_metrics(self,
                         x_hat: Tensor,
                         batch: Dict[str, Tensor]) -> Dict[str, float]:

        psnr = self.psnr(batch, x_hat)
        ssim = self.ssim(batch, x_hat)
        sa = self.sa(batch, x_hat)

        return {'psnr': psnr,
                'sa': sa,
                'ssim': ssim}
