"""Dynamic Embedding from DOFA paper.
Reference:
- https://arxiv.org/abs/2403.15356
- https://github.com/zhu-xlab/DOFA
"""
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

from hsicompressai.layers import Transformer

__all__ = [
        "DynamicEmbedding",
        ]

def posemb_sincos_1d(waves: Tensor,
                     dim: int,
                     temperature: int = 10000,
                     dtype=torch.float32) -> Tensor:
    """
    Args:
        waves: has shape [num_bands, 1], this will contain the wavelength
        for each band in the input hyperspectral image.

        dim: projection dimension for omega encoding size

    Process:
        Takes the wavelength tensor, and creates an omega vector of shape
        [1, dim], when the dot product is applied we get a vector of shape
        [num_bands, dim/2]. This matrix then has its values put through cos
        and sin separately. These two matrices are concatenated to create
        a matrix of size [num_bands, dim]

    """

    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)


class FCBlock(nn.Module):
    def __init__(self, size):
        """
        Maintains the same shape of the input matrix and
        adds residuals
        """
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        wave_dim,
        output_dim,
        num_latent_tokens,
        embed_dim,
        num_heads=4,
        num_layers=1,) -> None:

        super().__init__()

        self.num_latent_tokens = num_latent_tokens

        self.encoder = Transformer(input_dim=wave_dim,
                                   qkv_dim=wave_dim,
                                   num_layers=num_layers,
                                   heads=num_heads,
                                   dim_feedforward=embed_dim,
                                   dropout=0.1,)

        # layer = nn.TransformerEncoderLayer(
        #     d_model=wave_dim,
        #     nhead=num_heads,
        #     activation="gelu",
        #     dropout=0,
        #     norm_first=False,
        #     batch_first=True,
        # )

        # self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None

        self.weight_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, wave_dim) * 0.02
        )
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The weight_tokens and wavelengths (input) are concatenated into one
        tensor that is fed through the transformer encoder.

        The resulting output is sliced from the num_latent_tokens to the end
        of the output, which removes the weight_tokens encoder output.
        The contatenated x is sliced in the same way, to retreive original
        input x, and added to the sliced encoder output (adding residuals).
        This is put through a fully connected linear layer that has the shape
        [wave_dim, output_dim]
        """

        print("\nWaveTransformer...\n")
        print(f"x: {x.shape}")
        print(f"weight_tokens: {self.weight_tokens.shape}")
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        print(f"cat x and weight_tokens: {x.shape}")
        out = self.encoder(x.unsqueeze(0)).squeeze(0)
        print(f"encoder output: {out.shape}")

        weights = self.fc_weight(
            out[self.num_latent_tokens: -1] + x[self.num_latent_tokens: -1]
        )
        print(f"Slice encoder out: {out[self.num_latent_tokens: -1].shape}")
        print(f"Slice x: {x[self.num_latent_tokens: -1].shape}")
        print(f"weights: {weights.shape}\n")
        bias = None
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        wave_dim,
        num_latent_tokens,
        patch_size,
        embed_dim,
    ):
        """
        This dynamic embeddings layer generates weights via a transformer
        encoder to match the dimensions of the of the input shape.

        Args:

            wave_dim: Dimension of the resulting output
            num_latent_tokens:
            patch_size:
            embed_dim:

        """
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = waves.to(batch.device)
        waves = self.fclayer(waves)
        dynamic_weight, bias = self.weight_generator(waves)

        if bias is not None:
            bias = rearrange(bias, "b -> (b)")

        dynamic_out = F.linear(batch, dynamic_weight.T, bias=bias)
        x = dynamic_out

        return x, waves

    def initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if "__main__" == __name__:

    patch_embed = DynamicEmbedding(wave_dim=128,
                                   num_latent_tokens=128,
                                   patch_size=1,
                                   embed_dim=256,)

    print("First Input\n")
    batch_size, height, width, num_channels = 5, 256, 256, 202
    ainput = torch.rand((batch_size, height, width, num_channels))
    print(f"original input: {ainput.shape}")
    ainput = ainput.reshape((batch_size * height * width, num_channels))
    print(f"reshape input: {ainput.shape}")
    awaves = torch.rand(num_channels)
    x, waves = patch_embed(ainput, awaves)
    print(f"Output: {x.shape}")
