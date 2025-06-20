from torch import nn

from hsicompressai.metrics import mse

from hsicompressai.registry import register_criterion

@register_criterion("MeanSquaredErrorLoss")
class MeanSquaredErrorLoss(nn.Module):
    def __init__(self):
        super(MeanSquaredErrorLoss, self).__init__()
        self.metric = mse.MeanSquaredError()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)
