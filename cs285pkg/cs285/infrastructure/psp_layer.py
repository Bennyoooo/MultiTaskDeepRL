import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class BinaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)]
        m = x*o
        r = torch.mm(m, self.w)
        return r


class HashLinear(nn.Module):
    '''Complex layer with complex diagonal contexts'''
    def __init__(self, n_in, n_out, period=2, key_pick='hash', learn_key=True):
        super(HashLinear, self).__init__()
        self.key_pick = key_pick
        w_r = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        w_phi = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi)
        o_r = torch.ones(period, n_in)
        o_phi = torch.Tensor(period, n_in).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(torch.stack(from_polar(o_r, o_phi)))
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x_a, x_b, time):
        net_time = int(time) % self.o.shape[1]
        o = self.o[:, net_time]
        o_a = o[0].unsqueeze(0)
        o_b = o[1].unsqueeze(0)
        m_a = x_a*o_a - x_b*o_b
        m_b = x_b*o_a + x_a*o_b

        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(m_a, w_a) - torch.mm(m_b, w_b)
        r_b = torch.mm(m_b, w_a) + torch.mm(m_a, w_b)
        return r_a + self.bias, r_b


def from_polar(r, phi):
    a = r*torch.cos(phi)
    b = r*torch.sin(phi)
    return a, b
