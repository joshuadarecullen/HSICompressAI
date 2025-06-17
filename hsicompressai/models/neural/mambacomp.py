import math
import torch
import torch.nn.functional as f
from torch import nn
from einops import rearrange, repeat

from hsicompressai.layers import SpectralMamba

from hsicompressai.latent_codec.base import LatentCodec

class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features: int = 50,
                 hidden_features: int = 1024,
                 final_features: int = 202) -> None:

        super().__init__()

        self.patch_deembed = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=hidden_features,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_features,
                out_features=final_features,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_deembed(x)
class MLPEncoder(nn.Module):
    def __init__(self,
                 in_features: int = 64,
                 hidden_features: int = 1024,
                 latent_features: int = 50) -> None:

        super().__init__()

        self.latent = nn.Sequential(
                nn.Linear(in_features=in_features,
                          out_features=hidden_features),
                nn.GELU(),
                nn.Linear(in_features=hidden_features,
                          out_features=latent_features),
                nn.GELU(),
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.latent(x)



class PatchEmbedding(nn.Module):
    def __init__(self,
                 patch_dim: int,
                 embed_dim: int) -> None:

        super().__init__()

        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        # self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> torch.Tensor:
        return self.patch_to_embedding(x)


# TODO:
'''
1. Either prep input in SpecMamba then reconstruct, or do it before,
2. fiure out adaptiveNorm if we dont reconstruct
3. figure out compresssion token if we reconstruct
4. figure out the proj term if we do dont reconstruct to original image dimensions
5. fiure out adaptiveNorm if we dont reconstruct
6. figure out compresssion token if we reconstruct

'''


class MambaHSICompression(LatentCodec):
    def __init__(self,
                 mamba_params: dict,
                 dim: int = 16,
                 src_channels: int = 202,
                 target_compression_ratio: int = 4,
                 hidden_dim: int = 1024,
                 patch_depth: int = 4,
                 num_mamba_layers: int = 1,
                 ) -> None:
        """
        Key concepts:
            1. Spectral Compression only
            2.

        Args:
            target_compression_ratio: Compression ratio of input image channels
            hidden_dim: Hidden feature dimension of encoder and decoder
            token_num:
            group_num:
            use_att: If attention is applied after spectral mamba
        """

        super().__init__()

        self.src_channels = src_channels

        self.dim = dim

        self.latent_channels = int(
                math.ceil(src_channels / target_compression_ratio))

        self.compression_ratio = src_channels / self.latent_channels

        self.bpppc = 32 / self.compression_ratio

        self.delta_pad = int(math.ceil(src_channels / patch_depth)) * patch_depth - src_channels

        self.num_patches = (src_channels + self.delta_pad) // patch_depth

        self.patch_dim = (src_channels + self.delta_pad) // self.num_patches

        self.patch_to_embedding = PatchEmbedding(patch_dim=self.patch_dim,
                                                 embed_dim=self.dim)

        self.comp_token = nn.Parameter(torch.randn(1, 1, self.dim))

        self.mamba = nn.Sequential(
                    *[SpectralMamba(**mamba_params)
                      for _ in range(num_mamba_layers)]
                )

        self.encoder = MLPEncoder(in_features=dim,
                               hidden_features=hidden_dim,
                               latent_features=self.latent_channels)

        self.decoder = MLPDecoder(in_features=self.latent_channels,
                               hidden_features=hidden_dim,
                               final_features=src_channels)

    def compress(self, x):

        _, _, h, w = x.shape

        if self.delta_pad > 0:
            x = f.pad(x, (0, 0, 0, 0, self.delta_pad, 0))

        x = rearrange(x, 'b (n pd) w h -> (b w h) n pd',
                      n=self.num_patches,
                      pd=self.patch_dim,)

        x = self.patch_to_embedding(x)

        b, n, _ = x.shape
        comp_tokens = repeat(self.comp_token, '() n d -> b n d', b=b)

        # Now shape is [batch*128*128, token_num+1, patch_dim]
        x = torch.cat([x, comp_tokens], dim=1)

        x = self.mamba(x)

        # Extract last token (compression representation)
        comp_token_out = x[:, -1, :]

        b, n, _ = x.shape

        z = self.encoder(comp_token_out)

        z = rearrange(z, '(b w h) d -> b d w h',
                      d = self.latent_channels,
                      w = w,
                      h = h,
                      )

        return z

    def decompress(self, z):

        z = rearrange(z, 'b d w h -> b w h d')

        x_hat = self.decoder(z)

        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        return x_hat

    def forward(self, x):
        z = self.compress(x)
        x_hat = self.decompress(z)
        return x_hat


if "__main__" == __name__:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input = torch.randn((2, 202, 128, 128)).to(device)

    dim = 16
    src_channels = 202
    dim = 128

    mamba_params = {'use_residual': True,
                    'd_state': 16,
                    'd_conv': 4,
                    'expand': 2,
                    'group_num': dim}

    model = MambaHSICompression(mamba_params=mamba_params,
                                dim=dim,
                                patch_depth=32).to(device)


    output = model.compress(input)

    print(f"model input: {input.shape}")
    print(f"model output: {output.shape}")
