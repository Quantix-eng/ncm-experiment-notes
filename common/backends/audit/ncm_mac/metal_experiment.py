"""Experimental MPS pooling; does not alter the validated CPU runtime."""
from types import MethodType
import torch
from torch import nn


class AreaPool1d(nn.Module):
    """Adaptive-average bins floor(i*N/M):ceil((i+1)*N/M), entirely on device."""
    def __init__(self, input_length, output_length):
        super().__init__()
        starts = torch.arange(output_length) * input_length // output_length
        ends = ((torch.arange(output_length) + 1) * input_length + output_length - 1) // output_length
        width = int((ends - starts).max())
        indices = starts[:, None] + torch.arange(width)
        valid = indices < ends[:, None]
        self.register_buffer("indices", indices.clamp_max(input_length - 1).flatten())
        self.register_buffer("valid", valid.float())
        self.register_buffer("counts", (ends - starts).float())
        self.output_length, self.width = output_length, width

    def forward(self, x):
        gathered = x.index_select(-1, self.indices).reshape(*x.shape[:-1], self.output_length, self.width)
        return (gathered * self.valid).sum(-1) / self.counts


def _next_input(vq, level_idx, z_hat, z_q_level):
    if level_idx == len(vq.levels) - 1:
        z_hat.add_(vq._apply_quant_resi(z_q_level, level_idx))
        return z_hat, z_hat
    up = vq._apply_quant_resi(vq._up_to_full(z_q_level, level_idx, vq.levels[-1]), level_idx)
    z_hat.add_(up)
    pooled = vq.metal_area_pools[str(level_idx)](z_hat.transpose(-1, -2)).transpose(-1, -2)
    return z_hat, pooled


def install_area_pool(vq):
    """Instance-local adapter: only the experimental model receives this method."""
    vq.metal_area_pools = nn.ModuleDict({str(k): AreaPool1d(vq.levels[-1], vq.levels[k + 1])
                                       for k in range(len(vq.levels) - 1)})
    vq.get_next_autoregressive_input = MethodType(_next_input, vq)
