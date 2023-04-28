
import torch
import torch.nn as nn


class HyperParam(nn.Module):
    def __init__(self):
        super(HyperParam, self).__init__()
        self.tau_temp = nn.Parameter(torch.tensor([10.]))
        self.tau_dist = nn.Parameter(torch.tensor([10.]))