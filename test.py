import torch
import hsicompressai
from hsicompressai.models.neural import ScalableReduceComplexityEntropyModel
from hsicompressai.metrics import PeakSignalToNoiseRatio

psnr = PeakSignalToNoiseRatio()

src_channels = 6
batch = torch.randn(5, src_channels, 256, 256)  # Replace with real image
model = ScalableReduceComplexityEntropyModel(src_channels=src_channels)
outputs = model(batch)

loss = model.loss(outputs, batch)

x_hat = outputs['x_hat']
y_likelihoods = outputs['likelihoods']['y']
z_likelihoods = outputs['likelihoods']['z']

print(psnr(batch, x_hat))
print(loss)
