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

# print(psnr(batch, x_hat))
# print(loss)

lambdas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
#build a function that will lopp varying lambdas and return the psnr

def loop_lambdas(model, batch, lambdas):
    psnrs = []
    for l in lambdas:
        model.criterion.lmbda = l
        outputs = model(batch)
        x_hat = outputs['x_hat']
        psnrs.append(psnr(batch, x_hat))
    return psnrs

psnrs = loop_lambdas(model, batch, lambdas)

for psnr in psnrs:
    print(psnr)
