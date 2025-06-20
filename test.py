import torch
from hsicompressai.models import ScalableReduceComplexityEntropyModel


image = torch.randn(1, 3, 256, 256)  # Replace with real image
model = ScalableReduceComplexityEntropyModel()
output = model(image)
reconstructed = output['x_hat']
y_likelihoods = output['likelihoods']['y']
z_likelihoods = output['likelihoods']['z']
