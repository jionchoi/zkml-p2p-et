"""NLinear: Linear with last-value normalization, to absorb distribution shift."""
import torch.nn as nn

from helpers import SEQ_LEN, PRED_LEN


class NLinear(nn.Module):
    """Linear with last-value normalization, to absorb distribution shift."""
    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):                       # x: [B, seq_len, 1]
        last = x[:, -1:, :].detach()
        x = x - last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + last
