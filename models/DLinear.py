"""DLinear: decomposition into trend + seasonal, a linear layer each."""
import torch
import torch.nn as nn

from helpers import SEQ_LEN, PRED_LEN


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """Decomposition-Linear: split into trend + seasonal, a linear layer each."""
    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, kernel_size=25):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):                       # x: [B, seq_len, 1]
        trend = self.decomp(x)
        seasonal = x - trend
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend = self.Linear_Trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)
