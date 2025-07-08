"""
A Scalable Reduced-Complexity Compression of Hyperspectral
Remote Sensing Images Using Deep Learning

Sebastià Mijares i Verdú, Johannes Ballé , Valero Laparra, Joan Bartrina-Rapesta,
Miguel Hernández-Cabronero and Joan Serra-Sagristà. 
"""

from typing import List, Union, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from compressai.layers import GDN
from compressai.entropy_models import GaussianConditional
from compressai.entropy_models import EntropyBottleneck

from hsicompressai.registry import register_model
from hsicompressai.entropy_models import ConditionalHyperpriorAutoencoderBase
from hsicompressai.layers import SignalConv2D
from hsicompressai.losses import RateDistortionLoss

#TODO
"""
[] - change this file in signal conv to this folders in hsicompressai
[] - check what the bottle neck is doing
[] - GaussianConditioningModel


[] - figure out why the checkpoint files looks for the src in spatio spectral
"""
3
class AnalysisTransform(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 N: int=192,
                 kernel_size: int=5):

        super().__init__()

        self.net = nn.Sequential(
            SignalConv2D(in_channels=in_channels,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         activation=GDN(N)),
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         activation=GDN(N)),
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         activation=GDN(N)),
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same")
        )

    def forward(self, x):
        return self.net(x)


class SynthesisTransform(nn.Module):
    def __init__(self,
                 out_channels: int=3,
                 N: int=192,
                 kernel_size=5):

        super().__init__()

        self.net = nn.Sequential(
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         transpose=True,
                         activation=GDN(N, inverse=True)),
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         transpose=True,
                         activation=GDN(N, inverse=True)),
            SignalConv2D(in_channels=N,
                         out_channels=N,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         transpose=True,
                         activation=GDN(N, inverse=True)),
            SignalConv2D(in_channels=N,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         transpose=True)
        )

    def forward(self, y_hat):
        return self.net(y_hat)


class HyperAnalysisTransform(nn.Module):
    def __init__(self,
                 N: int=192,
                 M: int=192,
                 kernel_size: int=5):

        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(in_channels=N,
                         out_channels=M,
                         kernel_size=3,
                         stride=1,
                         padding="same",
                         activation=nn.ReLU(inplace=True)),
            SignalConv2D(in_channels=M,
                         out_channels=M,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         activation=None),
        )

    def forward(self, y: Tensor) -> Tensor:
        return self.net(torch.abs(y))


class HyperSynthesisTransform(nn.Module):
    def __init__(self,
                 N: int=192,
                 M: int=192,
                 kernel_size: int=5):

        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(in_channels=M,
                         out_channels=M,
                         kernel_size=kernel_size,
                         stride=2,
                         padding="same",
                         transpose=True,
                         activation=nn.ReLU(inplace=True)),
            SignalConv2D(in_channels=M,
                         out_channels=N,
                         kernel_size=3,
                         stride=1,
                         padding="same",
                         transpose=True)
        )

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.net(z_hat)


@register_model("ScalableReduceComplexityEntropyModel")
class ScalableReduceComplexityEntropyModel(ConditionalHyperpriorAutoencoderBase):
    def __init__(self,
                 src_channels: int=3,
                 cluster_size: int=3,
                 N: int=192,
                 M: int=192,
                 loss_metric: str="mse",
                 loss_return: str="all",
                 target_bpppc: float=1.0
                 ) -> None:

        super().__init__()

        self.analysis = AnalysisTransform(in_channels=cluster_size, N=N)
        self.synthesis = SynthesisTransform(out_channels=cluster_size, N=N)
        self.hyper_analysis = HyperAnalysisTransform(N=N, M=M)
        self.hyper_synthesis = HyperSynthesisTransform(N=N, M=M)

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)  # Uses learned scales

        # self.pad_size = src_channels % cluster_size
        self.cluster_size = cluster_size
        self.pad_size = (cluster_size - src_channels % cluster_size) % cluster_size

        self.criterion = RateDistortionLoss(lmbda=target_bpppc,
                                            metric=loss_metric,
                                            return_type=loss_return)

    def forward(self,
                x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:

        # Encoder
        y = self.analysis(x)

        # hyperprior encoder
        z = self.hyper_analysis(y)

        # entropy encode hyperprior
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # sythesis anaylsis transformer output from z_hat
        scales_hat = self.hyper_synthesis(z_hat)

        # measure hyperprior sythesis
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)

        # sythesis orginal from hyperprior sythesis
        x_hat = self.synthesis(y_hat)

        # compose original hsi shape
        # x_hat = x_hat[:, :C, :, :] #  remove padded

        return {
                'x_hat': x_hat,
                'likelihoods': {
                    'y': y_likelihoods,
                    'z': z_likelihoods
                    }
                }

    def loss(self, outputs: Dict[str, Tensor], batch: Tensor) -> Tensor:
        return self.criterion(outputs, batch)

    def compress(self, batch: Tensor) -> Tensor:
        pass

    def decompress(self, batch: Tensor) -> Tensor:
        pass
