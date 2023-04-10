# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .constraint import Constraint

import numpy as np
import torch as th
import cvxpy as cp
import gurobipy as gp

class SphericalConstraint(Constraint):

    """
    Action Constraints with the from $|a| \leq M$
    """
    
    def __init__(self, a_dim, r2):
        super().__init__(a_dim)
        self.r2 = r2
        self.r = r2 ** 0.5
    
    def isConstraintSatisfied(self, state, a, err=1e-3):
        return np.sum(np.square(a)) <= self.r2 + err

    def enforceConstraint(self, state, a):
        return min(self.r / np.linalg.norm(a), 1.) * a

    def numConstraints(self):
        return 1
    
    def getL(self, states, centers, v, get_grad:bool = False):
        L = v.norm(dim=1)/self.r
        if not get_grad:
            return L
        else:
            return L, v/L[:,None]/self.r**2

    def constraintViolation(self, state, a, err=1e-3, normalize=False):
        return np.expand_dims(np.maximum(0.0, np.sqrt(np.sum(np.square(a))) - self.r - err),0)
        
    def constraintViolationBatch(self, states, actions):
        return th.maximum(actions.norm(dim=1)-self.r, th.tensor(0.))
        
    def get_center(self, state):
        return np.zeros(self.a_dim)

    def cvxpy_constraints(self, x, state = None):
        return [cp.sum_squares(x) <= self.r2]

    def gp_constraints(self, model, x, state = None):
        Sq = gp.QuadExpr()
        for i in range(self.a_dim):
            Sq+=x[i]*x[i]
        model.addConstr(Sq <= self.r2)
