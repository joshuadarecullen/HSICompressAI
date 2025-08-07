"""Spectral Mamba layer for hyperspectral sequence modeling.

This module implements a Spectral Mamba layer designed specifically for processing
hyperspectral imagery. The layer leverages the Mamba state space model architecture
to efficiently model long-range dependencies across spectral dimensions.

Classes:
    SpectralMamba: Mamba-based layer for spectral sequence modeling

Example:
    >>> import torch
    >>> # Input: (batch_size, sequence_length, features)
    >>> x = torch.randn(2, 224, 64)  # 224 spectral bands, 64 features per band
    >>> layer = SpectralMamba(group_num=64, d_state=16)
    >>> output = layer(x)
    >>> print(output.shape)  # (2, 224, 64)

References:
    Gu, Albert, and Tri Dao. "Mamba: Linear-Time Sequence Modeling with 
    Selective State Spaces." arXiv preprint arXiv:2312.00752 (2023).
"""

from torch import nn
from mamba_ssm import Mamba


__all__ = [
        "SpectralMamba",
        ]


class SpectralMamba(nn.Module):
    """Spectral Mamba layer for hyperspectral sequence modeling.
    
    This layer applies Mamba state space model to hyperspectral data sequences,
    enabling efficient modeling of long-range spectral dependencies. The layer
    includes optional residual connections and layer normalization for stable training.
    
    The Mamba architecture provides linear computational complexity in sequence length,
    making it particularly suitable for hyperspectral data with hundreds of spectral bands.
    
    Attributes:
        use_residual (bool): Whether to use residual connections
        mamba (Mamba): Core Mamba state space model
        proj (nn.Sequential): Post-processing projection layers
        
    Args:
        use_residual (bool, optional): Enable residual connections. Defaults to True.
        d_state (int, optional): SSM state expansion factor. Higher values capture
            more complex dependencies but increase memory usage. Defaults to 16.
        d_conv (int, optional): Local convolution width for the SSM. Defaults to 4.
        expand (int, optional): Block expansion factor controlling model capacity.
            Defaults to 2.
        group_num (int, optional): Model dimension (number of features per token).
            Should match the feature dimension of input sequences. Defaults to 64.
            
    Example:
        >>> # Basic usage
        >>> layer = SpectralMamba(group_num=128, d_state=32)
        >>> x = torch.randn(4, 224, 128)  # (batch, spectral_bands, features)
        >>> output = layer(x)
        >>> 
        >>> # Without residual connections
        >>> layer = SpectralMamba(use_residual=False, group_num=64)
        >>> x = torch.randn(2, 176, 64)
        >>> output = layer(x)
        
    Note:
        Input should be tokenized hyperspectral data with shape 
        (batch_size, sequence_length, group_num) where sequence_length
        typically corresponds to the number of spectral bands.
    """

    def __init__(self,
                 use_residual: bool = True,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 group_num: int = 64):
        """Initialize SpectralMamba layer."""
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
        """Forward pass through SpectralMamba layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, group_num).
                Typically represents tokenized hyperspectral data where sequence_length
                corresponds to spectral bands and group_num is the feature dimension.
                
        Returns:
            torch.Tensor: Output tensor with same shape as input. If use_residual=True,
                returns x + processed(x), otherwise returns processed(x).
                
        Note:
            Input tensor is made contiguous before processing to ensure optimal
            memory layout for the Mamba operations.
        """
        x = x.contiguous()
        x_mamba = self.mamba(x).contiguous()
        x_proj = self.proj(x_mamba)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj
