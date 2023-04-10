# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

from ...cvxpy_variables import CVXPYVariables

class OptLayer(torch.nn.Module):
    """
    wrapper of cvxpylayer
    """
    def __init__(self, cons):
        super(OptLayer, self).__init__()
        cvxpy_vars = cons.cvxpy_variables()

        if cvxpy_vars.s is None:
            self.state_dependence = False
            self.layer = CvxpyLayer(cvxpy_vars.prob, parameters=[cvxpy_vars.q], variables=[cvxpy_vars.x])
        else:
            self.state_dependence = True
            self.layer = CvxpyLayer(cvxpy_vars.prob, parameters=[cvxpy_vars.q, cvxpy_vars.s], variables=[cvxpy_vars.x])

    def forward(self, x, s):
        if self.state_dependence:
            return self.layer(x, s)[0]
        else:
            return self.layer(x)[0]
        

if __name__ == "__main__":
    import numpy as np
    from ...half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint

    cons = HalfCheetahDynamicConstraint()
    layer = OptLayer(cons)

    x = torch.rand((100, 6))
    s = torch.rand((100, 17))
    y = layer(x,s)
    for t in range(x.shape[0]):
        assert cons.isConstraintSatisfied(s[t], y[t])
        assert not cons.isConstraintSatisfied(s[t], x[t]) or torch.allclose(x[t], y[t], rtol = 1e-2)

    # import diffcp

    # diffcp.solve_and_derivative_batch()
        
    for param in layer.parameters():
        print(param)

