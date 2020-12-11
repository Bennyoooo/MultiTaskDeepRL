import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class RouteLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(RouteLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        rots = []
        
        for key_i in range(period):
            row_idx = np.random.permutation(n_in)
            random_route = np.eye(n_in)[row_idx].astype('float32')
            rots.append(torch.from_numpy(random_route))

        rots = torch.stack(rots)
        self.rots = nn.Parameter(rots)
        
        if not learn_key:
            self.rots.requires_grad = False
    
    def forward(self, x, time):
        m = torch.mm(x, self.rots[int(time)])
        ret = torch.mm(m, self.w) + self.bias
        return ret