# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
import numpy as np
import torch as th
import math
from abc import ABC, abstractmethod

from .normalize_constraint import normalizeConstraint
from ..nn.additional_layers.chebyshev_center import calc_chebyshev_center

import cvxpy as cp
from ..cvxpy_variables import CVXPYVariables
from .constraint import Constraint

import gurobipy as gp

EPS=1e-9

class CombinedConstraint(Constraint):
    """
    class for combined two constraints
    """
    def __init__(self, cons1, cons2):
        super().__init__(cons1.a_dim, cons1.s_dim)
        self.cons1=cons1
        self.cons2=cons2
        
    def isConstraintSatisfied(self, state, a):
        return self.cons1.isConstraintSatisfied(state, a) and self.cons2.isConstraintSatisfied(state,a)
    
    def numConstraints(self):
        return self.cons1.numConstraints() + self.cons2.numConstraints()
    
    def getL(self, states, centers, v, get_grad:bool = False):
        if get_grad:
            L1, grad1 = self.cons1.getL(states, centers, v, get_grad = True)
            L2, grad2 = self.cons2.getL(states, centers, v, get_grad = True)
            L = th.maximum(L1, L2)
            grad = th.where(th.ge(L1,L2)[:,None], grad1, grad2)
            return L, grad
        else:
            L1 = self.cons1.getL(states, centers, v)
            L2 = self.cons2.getL(states, centers, v)
            return th.maximum(L1, L2)

    def get_center(self, state):
        return self.cons2.get_center(state)

    def cvxpy_constraints(self, x, state = None):
        return self.cons1.cvxpy_constraints(x, state)+self.cons2.cvxpy_constraints(x,state)

    def gp_constraints(self, model, x, state = None):
        self.cons1.gp_constraints(model, x, state)
        self.cons2.gp_constraints(model, x, state)
        
    def constraintViolation(self, state, a, err=1e-2, normalize=False):
        return self.cons1.constraintViolation(state, a, err, normalize)+self.cons2.constraintViolation(state, a, err, normalize)

    def constraintViolationBatch(self, states, actions):
        return self.cons1.constraintViolationBatch(states, actions)+self.cons2.constraintViolationBatch(states, actions)



if __name__ == "__main__":
    from .power_constraint import OrthoplexConstraint
    from .sin2_constraint import Sin2Constraint
    offset_p = 2
    scale = (1., 1., 1., 1., 1., 1.)
    indices_p = list(range(offset_p, offset_p+len(scale)))
    offset_v = 11
    indices_v = list(range(offset_v, offset_v+len(scale)))
    s_dim = 17
    cons = CombinedConstraint(OrthoplexConstraint(indices_v, scale, 10., s_dim),
                              Sin2Constraint(indices_p, 0.1, s_dim))
    state=np.array([8.07938070e-01,  1.66712368e-01, -3.53352869e-01, -2.05882562e+00, 7.96720395e-01, -8.13423423e-01, -5.51193071e-01, -2.03170841e-01,  9.15822510e-01, -1.39466434e+00, -2.43234536e+00,  3.60834351e+00, -1.00000000e+01,  2.21396068e-06,  9.28677242e-01, -9.96848688e+00,  7.87601474e+00])
    action=np.array([0.26813674, -0.16751897, 0.7599944, -0.00297695, -0.07156372, -0.1420452 ])
    print(cons.get_center(state))
