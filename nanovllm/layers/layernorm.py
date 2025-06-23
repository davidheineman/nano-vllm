import torch
from torch import nn


# TODO: I removed in-place operations from RMS norm (went from 490 -> 440 TPS)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.eps)
        x_norm = x_norm.to(orig_dtype) * self.weight
        return x_norm

    def add_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32) + residual.to(torch.float32)
        residual_out = x_f32.to(orig_dtype)
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.eps)
        x_norm = x_norm.to(orig_dtype) * self.weight
        return x_norm, residual_out

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
