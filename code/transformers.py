import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int,
                 device: torch.device=None, dtype: torch.dtype=None):
        '''
        Parameters:
            d_in: int final dimension of the input
            d_out: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=math.sqrt(2/(d_out + d_in)), a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

