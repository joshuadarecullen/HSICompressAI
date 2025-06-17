from torch import nn

from metrics import ssim


class StructuralSimilarityLoss(nn.Module):
    def __init__(self, data_range=1.0, channels=202):
        super(StructuralSimilarityLoss, self).__init__()
        self.metric = ssim.StructuralSimilarity(data_range, channels)

    def forward(self, x, x_hat):
        return 1 - self.metric(x, x_hat)
