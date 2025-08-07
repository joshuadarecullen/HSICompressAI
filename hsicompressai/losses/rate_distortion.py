"""Rate-distortion loss functions for hyperspectral image compression.

This module implements rate-distortion optimization losses used in neural image
compression systems. The losses balance reconstruction quality (distortion) 
against compressed bitstream size (rate) using Lagrangian optimization.

Classes:
    RateDistortionLoss: Lagrangian rate-distortion loss for compression training

Example:
    >>> criterion = RateDistortionLoss(lmbda=0.01, metric="mse")
    >>> # Model output with reconstructed image and likelihoods
    >>> output = {
    ...     "x_hat": reconstructed_image,  # (B, C, H, W) 
    ...     "likelihoods": {"y": likelihood_y, "z": likelihood_z}
    ... }
    >>> target = original_image  # (B, C, H, W)
    >>> loss_dict = criterion(output, target)
    >>> total_loss = loss_dict["loss"]

References:
    Ballé, Johannes, Valero Laparra, and Eero P. Simoncelli. "End-to-end 
    optimized image compression." arXiv preprint arXiv:1611.01704 (2016).
"""

import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim
from hsicompressai.registry import register_criterion

@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Lagrangian rate-distortion loss for neural image compression.
    
    This loss function implements the rate-distortion optimization objective
    used in neural image compression. It combines a distortion term (measuring 
    reconstruction quality) with a rate term (measuring compressed size) using
    a Lagrangian multiplier.
    
    The loss is formulated as: L = λ * D + R
    where D is distortion (MSE or MS-SSIM), R is rate (bits per pixel), 
    and λ controls the rate-distortion trade-off.
    
    Attributes:
        metric (nn.Module or callable): Distortion metric (MSELoss or ms_ssim)
        lmbda (float): Lagrangian multiplier for rate-distortion trade-off
        return_type (str): What to return ("all", "loss", "bpp_loss", etc.)
        
    Args:
        lmbda (float, optional): Lagrangian multiplier. Higher values prioritize
            rate reduction over quality. Defaults to 0.01.
        metric (str, optional): Distortion metric. Options: "mse", "ms-ssim".
            Defaults to "mse".
        return_type (str, optional): Return format. "all" returns dictionary
            with all loss components, otherwise returns specific component.
            Defaults to "all".
            
    Example:
        >>> # Standard MSE-based rate-distortion loss
        >>> rd_loss = RateDistortionLoss(lmbda=0.01, metric="mse")
        >>> 
        >>> # MS-SSIM based loss for perceptual quality
        >>> rd_loss = RateDistortionLoss(lmbda=0.001, metric="ms-ssim")
        >>> 
        >>> # Using the loss
        >>> model_output = {
        ...     "x_hat": reconstructed,
        ...     "likelihoods": {"y": y_likelihood, "z": z_likelihood}
        ... }
        >>> losses = rd_loss(model_output, target_image)
        >>> print(losses.keys())  # ['bpp_loss', 'mse_loss', 'loss']
        
    Raises:
        NotImplementedError: If unsupported metric is specified.
    """

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        """Initialize rate-distortion loss."""
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        """Compute rate-distortion loss.
        
        Args:
            output (dict): Model output containing:
                - "x_hat": Reconstructed image tensor with shape (N, C, H, W)
                - "likelihoods": Dict of likelihood tensors for rate estimation
            target (torch.Tensor): Original target image with shape (N, C, H, W)
            
        Returns:
            dict or torch.Tensor: If return_type="all", returns dictionary with:
                - "bpp_loss": Bits per pixel (rate term)
                - "mse_loss" or "ms_ssim_loss": Distortion term
                - "loss": Combined rate-distortion loss
                Otherwise returns the specific loss component.
                
        Note:
            Rate is computed from likelihood tensors using negative log-likelihood
            scaled by bits per pixel. Distortion uses either MSE or MS-SSIM.
        """
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            # distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
