from torch import nn

from hsicompressai.metrics import sa
from hsicompressai.registry import register_criterion

@register_criterion("SpectralAngleLoss")
class SpectralAngleLoss(nn.Module):
    def __init__(self):
        super(SpectralAngleLoss, self).__init__()
        self.metric = sa.SpectralAngle()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)
