"""LTSF-Linear: one linear layer mapping the history window to the forecast."""
import torch.nn as nn

from helpers import SEQ_LEN, PRED_LEN


class Linear(nn.Module):
    """One linear layer mapping the history window to the forecast."""
    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):                       # x: [B, seq_len, 1]
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
