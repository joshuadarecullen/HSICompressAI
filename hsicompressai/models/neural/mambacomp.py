import math
import torch
import torch.nn.functional as f
from torch import nn
from einops import rearrange, repeat

from hsicompressai.layers import SpectralMamba, DynamicEmbedding
from hsicompressai.latent_codecs import LatentCodec
from hsicompressai.registry import register_model

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
4. figure out the proj term if we do dont reconstruct to original image dimensions
5. fiure out adaptiveNorm if we dont reconstruct
6. figure out compresssion token if we reconstruct
7. PatchEmbed so we can plug in different patch embeddings
'''

@register_model("MambaComp")
class MambaHSICompression(LatentCodec):
    def __init__(self,
                 metadata_path: str = None,
                 mamba_params: dict,
                 platform: str = 'hyspek',
                 dynamic_embed_params: dict,
                 src_channels: int = 202,
                 target_compression_ratio: int = 4,
                 hidden_dim: int = 1024,
                 num_token: int = 8,
                 num_mamba_layers: int = 1,
                 ) -> None:
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

        self.platform = platform

        if metadata_path:
            self.metadata = Box(yaml.safe_load(open(metadata_path)))
            self.waves = torch.tensor(list(self.metadata[self.platform].bands.wavelengths.values()))
        else:
            # self.metadata = None
            self.waves = None

        self.src_channels = src_channels
        self.latent_channels = int(
                math.ceil(src_channels / target_compression_ratio))
        self.compression_ratio = src_channels / self.latent_channels
        self.bpppc = 32 / self.compression_ratio
        self.num_tokens = num_token
        self.patch_dim = dynamic_embed_params["embed_dim"] // num_token #  dim of each token

        self.comp_token = nn.Parameter(torch.randn(1, 1, self.patch_dim))

        self.patch_to_embedding = DynamicEmbedding(**dynamic_embed_params)

        self.mamba = nn.Sequential(
                    *[SpeMamba(**mamba_params.update({'group_num': self.patch_dim}))
                      for _ in range(num_mamba_layers)]
                )

        self.encoder = MLPEncoder(in_features=patch_dim,
                                  hidden_features=hidden_dim,
                                  latent_features=self.latent_channels)

        self.decoder = MLPDecoder(in_features=self.latent_channels,
                                  hidden_features=hidden_dim,
                                  final_features=src_channels)

        self.to_out = DynamicEmbedding(**dynamic_embed_params)

    def compress(self, x):

        _, _, h, w = x.shape

        # if self.delta_pad > 0:
        #     x = f.pad(x, (0, 0, 0, 0, self.delta_pad, 0))

        x = rearrange(x, 'b c h w -> (b h w) c')

        if self.waves
            x, _ = self.patch_to_embedding(x, waves)
        else
            x = self.patch_to_embedding(x)

        x = rearrange(x, 'b (n pd) -> (b) n pd',
                      n=self.num_tokens,
                      pd=self.patch_dim,)

        b, n, _ = x.shape
        comp_tokens = repeat(self.comp_token, '() n d -> b n d', b=b)

        ' I have to figure this out, maybe check the mamba paper'
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


# if "__main__" == __name__:

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     input = torch.randn((2, 202, 128, 128)).to(device)

#     patch_dim = 64
#     src_channels = 202

#     mamba_params = {'use_residual': True,
#                     'd_state': 16,
#                     'd_conv': 4,
#                     'expand': 2,}

#     dynamic_embed_params = {"wave_dim": 128,
#                             "num_latent_tokens": 128,
#                             "patch_size": 1,
#                             "embed_dim": 256,
#                             "is_decoder": True}

#     # metadata_path = '/home/jd983/Documents/phd/code/spatio-spectral-hsi/data/metadata.yaml'
#     # metadata = Box(yaml.safe_load(open(metadata_path)))
#     # print(metadata)
#     # waves = torch.tensor(list(metadata['hyspek'].bands.wavelengths.values()))
#     # print(waves.shape)


#     model = MambaHSICompression(metadata_path='/home/jd983/Documents/phd/code/spatio-spectral-hsi/data/metadata.yaml',
#                                 mamba_params=mamba_params,
#                                 dynamic_embed_params=dynamic_embed_params,
#                                 patch_dim=patch_dim,
#                                 num_token=4).to(device)

#     # summary(model)

#     output = model.compress(input)

#     print(f"model input: {input.shape}")
#     print(f"model output: {output.shape}")
