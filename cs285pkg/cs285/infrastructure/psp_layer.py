import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class BinaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        # self.key_pick = key_pick
        # w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        # rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        # o = torch.from_numpy(rand_01*2 - 1)
        # self.n_in = n_in
        # self.n_out = n_out
        #
        # self.w = nn.Parameter(w)
        # self.bias = nn.Parameter(torch.zeros(n_out))
        # self.o = nn.Parameter(o)
        self.linear = nn.Linear(n_in, n_out)
        # if not learn_key:
        #     self.o.requires_grad = False

    def forward(self, x, time=0):
        # o = self.o[:, int(time)]
        # m = x*o
        r = self.linear(x)
        return r
