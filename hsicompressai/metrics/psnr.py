import torch

from torch import nn


class PeakSignalToNoiseRatio(nn.Module):
    def __init__(self):
        super(PeakSignalToNoiseRatio, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, a, b):
        return -10 * torch.log10(self.mse(a, b))
