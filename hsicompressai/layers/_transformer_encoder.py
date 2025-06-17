import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor

from typing import Union, Optional


class MLP_Block(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1) -> None:

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 qkv_dim: int,
                 heads: int = 8,
                 dropout=0.1):

        super().__init__()

        self.heads = heads
        self.scale = qkv_dim ** -0.5  # 1/sqrt(dim)
        inner_dim = qkv_dim * heads

        # Wq,Wk,Wv for each vector, thats why *3
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=True)
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.to_out = nn.Linear(inner_dim, input_dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.to_out(out)
        out = self.dropout(out)

        # print(f'out: {out.shape}')

        return out


class LayerNormalise(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: Union[Attention, MLP_Block]) -> None:

        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    def __init__(self, fn: LayerNormalise):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 qkv_dim: int,
                 num_layers: int,
                 heads: int,
                 dim_feedforward: int,
                 dropout: float) -> None:

        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(
                    nn.ModuleList([
                        Residual(LayerNormalise(input_dim,
                                                Attention(input_dim,
                                                          qkv_dim,
                                                          heads=heads,
                                                          dropout=dropout))),
                        Residual(LayerNormalise(input_dim,
                                                MLP_Block(input_dim,
                                                          dim_feedforward,
                                                          dropout=dropout)))
                                                ])
                    )

    def forward(self, x, mask: Optional[Tensor] = None) -> Tensor:
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


if "__main__" == __name__:
    input = torch.randn((1, 64, 200))

    dict = {"input_dim": 200,
            "qkv_dim": 100,
            "num_layers": 1,
            "heads": 1,
            "dim_feedforward": 200,
            "dropout": 0.1}

    model = Transformer(**dict)

    x_hat = model(input)
