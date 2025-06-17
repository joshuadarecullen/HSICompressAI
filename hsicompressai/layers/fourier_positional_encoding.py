import torch
import torch.nn as nn

class FourierSpectralEmbedding(nn.Module):
    def __init__(self, num_bands, embed_dim, scale=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_bands, embed_dim // 2) * scale)  # Learnable frequencies
        print(self.W.shape)

    def forward(self, x):
        # Compute Fourier embeddings
        pos = torch.arange(x.shape[1]).to(x.device).float().unsqueeze(1)  # Spectral indices
        print(pos.shape)
        emb = torch.cat([torch.sin(pos * self.W), torch.cos(pos * self.W)], dim=-1)  # [num_bands, embed_dim]
        print(f'{pos.shape}')

        # Add embeddings
        x = x + emb #.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x


if __name__ == "__main__":
    FPE = FourierSpectralEmbedding(num_bands=202, embed_dim=202)

    input = torch.randn((128*128, 202))

    output = FPE(input)

    print(output.shape)
