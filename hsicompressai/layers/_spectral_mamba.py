from torch import nn
from mamba_ssm import Mamba


class SpectralMamba(nn.Module):
    """
    This block assumes you are passing in tokenised sequences
    """

    def __init__(self,
                 use_residual: bool = True,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 group_num: int = 64):

        super().__init__()

        self.use_residual = use_residual

        # This module uses roughly 3 * expand * d_model^2 parameters
        self.mamba = Mamba(
                            d_model=group_num,  # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            d_conv=d_conv,  # Local convolution width
                            expand=expand,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.LayerNorm(group_num),
            nn.SiLU()
        )

    def forward(self, x):
        x = x.contiguous()
        x_mamba = self.mamba(x).contiguous()
        x_proj = self.proj(x_mamba)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj
