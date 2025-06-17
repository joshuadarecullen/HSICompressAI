import math
import torch
import torch.nn.functional as f
from torch import nn

from einops import rearrange, repeat

from hsicompressai.layers import Transformer
from hsicompressai.latent_codec.base import LatentCodec


def hycot_cr4(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=4,
    )


def hycot_cr8(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=8,
    )


def hycot_cr16(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=16,
    )


def hycot_cr32(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=32,
    )


class HyperspectralCompressionTransformer(LatentCodec):
    def __init__(
            self,
            src_channels=202,
            target_compression_ratio=4,
            patch_depth=4,
            hidden_dim=1024,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=8,
            dim_head=16,
            dropout=0.,
            emb_dropout=0.,
        ):
        super().__init__()

        self.src_channels = src_channels

        self.dim = dim

        latent_channels = int(math.ceil(src_channels / target_compression_ratio))
        self.latent_channels = latent_channels

        self.compression_ratio = src_channels / latent_channels
        self.bpppc = 32 / self.compression_ratio

        self.delta_pad = int(math.ceil(src_channels / patch_depth)) * patch_depth - src_channels

        num_patches = (src_channels + self.delta_pad) // patch_depth
        self.num_patches = num_patches

        patch_dim = (src_channels + self.delta_pad) // num_patches
        self.patch_dim = patch_dim
        
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.comp_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)

        self.to_latent = nn.Sequential(
            nn.Linear(
                in_features=dim,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=latent_channels,
            ),
            nn.Sigmoid(),
        )

        self.patch_deembed = nn.Sequential(
            nn.Linear(
                in_features=latent_channels,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=src_channels,
            ),
            nn.Sigmoid(),
        )

    def compress(self, x):
        _, _, h, w = x.shape

        if self.delta_pad > 0:
            x = f.pad(x, (0, 0, 0, 0, self.delta_pad, 0))

        x = rearrange(x, 'b (n pd) w h -> (b w h) n pd',
                      n = self.num_patches,
                      pd = self.patch_dim,
                      )

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # concat compression tokens
        comp_tokens = repeat(self.comp_token, '() n d -> b n d', b = b)
        x = torch.cat((comp_tokens, x), dim = 1)

        # add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x)

        # extract transformed comp_tokens
        y = x[:, 0]
        
        y = self.to_latent(y)

        y = rearrange(y, '(b w h) d -> b d w h',
                      d = self.latent_channels,
                      w = w,
                      h = h,
                      )

        return y

    def decompress(self, y):
        y = rearrange(y, 'b d w h -> b w h d')
        
        x_hat = self.patch_deembed(y)
        
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        return x_hat
    
    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat




if __name__ == '__main__':
    import torch

    model = HyperspectralCompressionTransformer()
    # print(model)

    in_tensor = torch.randn(2, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)

    latent_tensor = model.compress(in_tensor)
    print("latent shape:\t\t", latent_tensor.shape)
    
    out_tensor = model(in_tensor)
    print("out shape:\t\t", out_tensor.shape)

    print("in shape = out shape:\t", out_tensor.shape == in_tensor.shape)

    print("real CR:\t\t", torch.numel(in_tensor) / torch.numel(latent_tensor))
    print("model parameter CR:\t", model.compression_ratio)

    print("real bpppc:\t\t", 32 * torch.numel(latent_tensor) / torch.numel(in_tensor))
    print("model parameter bpppc:\t", model.bpppc)
