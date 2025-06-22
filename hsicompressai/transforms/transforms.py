from einops import rearrange

__all__ = [
        "TensorToMatrix"
        ]

class TensorToMatrix:
    def __call__(self, x):
        return rearrange(x, 'c h w -> (h w) c')

