import torch
from torch import Tensor, nn
from torch.autograd import Function
import pywt


'''
    @staticmethod
    def backward(ctx, dx) -> Tensor:
        if ctx.needs_input_grad[0]:
            w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 8, -1, H // 2, W // 2)  # Reshape based on 8 subbands

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = cat([w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh], dim=0).repeat(C, 1, 1, 1)
            dx = nn.functional.conv_transpose3d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None, None, None, None, None
'''


class DWT_Function_3D(Function):
    @staticmethod
    def forward(ctx, x: Tensor, filters: Tensor) -> Tensor:
        x = x.contiguous()
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        C = x.shape[1]  # Number of channels
        num_filters = filters.shape[0]  # 8 filters

        # Apply all wavelet filters at once
        filters = filters.repeat(C, 1, 1, 1, 1)  # Expand for grouped conv
        x = torch.nn.functional.conv3d(x, filters, stride=(2, 2, 2), groups=C)

        # Reshape to separate subbands
        x = x.view(x.shape[0], num_filters, -1, x.shape[-2], x.shape[-1])
        return x

    @staticmethod
    def backward(ctx: Tensor, dx: Tensor) -> Tensor:
        if ctx.needs_input_grad[0]:
            filters, = ctx.saved_tensors
            B, _, C, H, W = ctx.shape

            dx = dx.view(B, 8 * C, H // 2, W // 2)
            dx = torch.nn.functional.conv_transpose3d(dx,
                                                      filters,
                                                      stride=(2, 1, 1),
                                                      groups=C)

        return dx, None


class DWT_3D(nn.Module):
    def __init__(self, wave: str = 'haar') -> None:
        super().__init__()

        # Load wavelet filters from PyWavelets
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        # Create 3D wavelet filters
        dec_lo_3d = dec_lo.view(-1, 1, 1)
        dec_hi_3d = dec_hi.view(-1, 1, 1)

        # Construct 3D wavelet filters
        filters = [
            dec_lo_3d * dec_lo_3d.transpose(0, 1) * dec_lo_3d.transpose(0, 2),
            dec_lo_3d * dec_lo_3d.transpose(0, 1) * dec_hi_3d.transpose(0, 2),
            dec_lo_3d * dec_hi_3d.transpose(0, 1) * dec_lo_3d.transpose(0, 2),
            dec_lo_3d * dec_hi_3d.transpose(0, 1) * dec_hi_3d.transpose(0, 2),
            dec_hi_3d * dec_lo_3d.transpose(0, 1) * dec_lo_3d.transpose(0, 2),
            dec_hi_3d * dec_lo_3d.transpose(0, 1) * dec_hi_3d.transpose(0, 2),
            dec_hi_3d * dec_hi_3d.transpose(0, 1) * dec_lo_3d.transpose(0, 2),
            dec_hi_3d * dec_hi_3d.transpose(0, 1) * dec_hi_3d.transpose(0, 2),
        ]

        # Stack filters into one tensor (8 filters, 1 input channel, D, H, W)
        self.register_buffer("filters",
                             torch.stack(filters, dim=0).unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        return DWT_Function_3D.apply(x, self.filters)


def test_filters(output: Tensor):

    input = torch.rand((8, 1, 220, 145, 145))

    dwt = DWT_3D()

    output = dwt(input)

    dwt = DWT_3D()
    out = dwt(input)

    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            if i != j:
                print(torch.equal(out[:,i,:,:,:], out[:,j,:,:,:]))


if "__main__" == __name__:
    input = torch.rand((8, 1, 220, 145, 145))

    dwt = DWT_3D()

    output = dwt(input)

    print(output.shape)
