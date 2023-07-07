import numpy as np
import torch
from torch import nn
from functools import partial


class DemoModel(nn.Module):
    def __init__(self, hid, *args, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(3, hid)
        self.linear2 = nn.Linear(hid, 3)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.permute(0, 3, 1, 2)
        return x
