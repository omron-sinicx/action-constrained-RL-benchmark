# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import torch
import torch.nn as nn
import torch.nn.functional as F

from .analytic_center import calc_analytic_center

class SquashLayer(torch.nn.Module):
    def __init__(self, cons):
        super(SquashLayer, self).__init__()
        self.cons = cons

    def forward(self, actions, states, centers):
        v = actions - centers
        L = self.cons.getL(states, centers, v)
        return centers + (torch.tanh(L) / (L+1e-9))[:,None] * v

# unused layer
class SmoothSquashLayer(torch.nn.Module):
    def __init__(self, cons):
        super(SmoothSquashLayer, self).__init__()
        self.cons = cons

    def forward(self, actions, states, centers):
        C=self.cons.C(states)
        d=self.cons.d(states)
        v = actions - centers
        b = torch.maximum((v[:,None,:]*C).sum(axis=2) / (d-(centers[:,None,:]*C).sum(axis=2)), torch.tensor(0.))
        r = torch.linalg.norm(v, axis = 1)
        p = torch.minimum(2*torch.exp(r).detach(),torch.tensor(10.))
        return centers + (torch.pow(torch.pow(b,p[:,None]).sum(axis=1),-1/p) * torch.tanh(torch.linalg.norm(b, axis = 1)))[:,None] * v

if __name__ == "__main__":
    import numpy as np
    from ...half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint


    cons = HalfCheetahDynamicConstraint()
    state = np.repeat(10.0, 17)

    C=cons.C(state)
    d=cons.d(state)
    layer=SquashLayer(cons)
    print(C,d)
    for t in range(100):
        action=np.random.rand(6)
        print(action)
        result = layer.forward(action, state)
        print(result)
        assert cons.isConstraintSatisfied(state, result)
