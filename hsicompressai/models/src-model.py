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

from compressai.layers import GDN
from compressai.entropy_models import GaussianConditional

from hsicompressai.registry import register_model
from hsicompressai.entropy_models import ConditionalHyperpriorAutoencoderBase
from hsicompressai.layers import SignalConv2D

class AnalysisTransform(nn.Module):
    def __init__(self, N: int=192):
        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(3, N, kernel_size=5, stride=2, padding="same", activation=GDN(N)),
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same", activation=GDN(N)),
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same", activation=GDN(N)),
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same")
        )

    def forward(self, x):
        return self.net(x)


class SynthesisTransform(nn.Module):
    def __init__(self, N: int=192):
        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same", transpose=True, activation=GDN(N, inverse=True)),
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same", transpose=True, activation=GDN(N, inverse=True)),
            SignalConv2D(N, N, kernel_size=5, stride=2, padding="same", transpose=True, activation=GDN(N, inverse=True)),
            SignalConv2D(N, 3, kernel_size=5, stride=2, padding="same", transpose=True)
        )

    def forward(self, y_hat):
        return self.net(y_hat)


class HyperAnalysisTransform(nn.Module):
    def __init__(self,
                 N: int=192,
                 M: int=192):

        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(N, M, kernel_size=3, stride=1, padding="same", activation=nn.ReLU(inplace=True)),
            SignalConv2D(M, M, kernel_size=5, stride=2, padding="same", activation=None),
        )

    def forward(self, y: Tensor) -> Tensor:
        return self.net(torch.abs(y))


class HyperSynthesisTransform(nn.Module):
    def __init__(self,
                 N: int=192,
                 M: int=192):

        super().__init__()
        self.net = nn.Sequential(
            SignalConv2D(M, M, kernel_size=5, stride=2, padding="same", transpose=True, activation=nn.ReLU(inplace=True)),
            SignalConv2D(M, N, kernel_size=3, stride=1, padding="same", transpose=True)
        )

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.net(z_hat)


@register_model("ScalableReduceComplexityEntropyModel")
class ScalableReduceComplexityEntropyModel(ConditionalHyperpriorAutoencoderBase):
    def __init__(self,
               N: int=192,
               M: int=192) -> None:

        super().__init__()

        self.analysis = AnalysisTransform(N)
        self.synthesis = SynthesisTransform(N)
        self.hyper_analysis = HyperAnalysisTransform(N, M)
        self.hyper_synthesis = HyperSynthesisTransform(N, M)

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)  # Uses learned scales

    def forward(self,
                x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:

        # Encoder
        y = self.analysis(x)
        z = self.hyper_analysis(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.hyper_synthesis(z_hat)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.synthesis(y_hat)
        return {
                'x_hat': x_hat,
                'likelihoods': {
                    'y': y_likelihoods,
                    'z': z_likelihoods
                    }
                }


    def compress(self, batch: Tensor) -> Tensor:
        pass

    def decompress(self, batch: Tensor) -> Tensor:
        pass


if __name__ == "__main__":
    image = torch.randn(1, 3, 256, 256)  # Replace with real image
    model = ScalableReduceComplexityEntropyModel()
    output = model(image)
    reconstructed = output['x_hat']
    y_likelihoods = output['likelihoods']['y']
    z_likelihoods = output['likelihoods']['z']
