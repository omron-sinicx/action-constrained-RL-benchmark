# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import torch
import torch.nn as nn
import torch.nn.functional as F

from .analytic_center import calc_analytic_center
from .chebyshev_center import calc_chebyshev_center

class AlphaProjectionLayer(torch.nn.Module):
    def __init__(self, cons):
        super(AlphaProjectionLayer, self).__init__()
        self.cons = cons

    def forward(self, actions, states, centers):
        v = actions - centers
        L = self.cons.getL(states, centers, v)
        ret = centers + v / torch.maximum(L, torch.tensor(1))[:,None]
        return centers + v / torch.maximum(L, torch.tensor(1))[:,None]

if __name__ == "__main__":
    import numpy as np
    from ...half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint


    cons = HalfCheetahDynamicConstraint()
    states = torch.tensor(10.0).repeat(10,17)

    C=cons.C(states)
    d=cons.d(states)
    layer=AlphaProjectionLayer(cons)
    print(C,d)
    actions = torch.rand((10,6))
    print(actions)
    results = layer.forward(actions, states)
    print(results)
    for t in range(10):
        assert cons.isConstraintSatisfied(states[t], results[t])
        assert not cons.isConstraintSatisfied(states[t], actions[t]) or torch.allclose(actions[t], results[t])
