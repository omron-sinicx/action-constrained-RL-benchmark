# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .constraint import LinearConstraint
import torch

import cvxpy as cp
import gurobipy as gp

from ..cvxpy_variables import CVXPYVariables
from .power_constraint import make_compatible

class BoxConstraint(LinearConstraint):

    """
    Action Constraints with the from
    $|a_i| \leq 1`$ 
    """

    def __init__(self, a_dim):
        super().__init__(a_dim, -1)
        eyes = torch.eye(self.a_dim)
        self.C_value = torch.concat((eyes, -eyes), axis = 0)

        self.d_value = torch.ones(2*self.a_dim)

    def tensor_C(self, state):
        size = state.shape[0]
        self.C_value = make_compatible(self.C_value, state)
        return self.C_value.repeat(size, 1, 1)
    
    def tensor_d(self, state):
        size = state.shape[0]
        self.d_value = make_compatible(self.d_value, state)
        return self.d_value.repeat(size, 1)


    def numConstraints(self):
        return 2 * self.a_dim

    def E(self, state):
        return self.C(state)

    def f(self, state):
        return self.d(state)

    def cvxpy_constraints(self, x, state):
        cons = []
        for i in range(self.a_dim):
            cons.append(x[i] <= 1.)
            cons.append(-x[i] <= 1.)
        return cons

    def gp_constraints(self, model, x, s):
        pass
