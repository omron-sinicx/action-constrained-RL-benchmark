# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .quadratic_constraint import QuadraticConstraint
import torch as th

import cvxpy as cp
import gurobipy as gp
import math

from ..cvxpy_variables import CVXPYVariables
from .power_constraint import make_compatible

class Sin2Constraint(QuadraticConstraint):

    """
    State-dependent Action Constraints with the from
    $\sum a_i^2\sin^2\theta_i  \leq M$ where $\theta_i$ is the angle corresponding to $a_i$
    """

    def __init__(self, index, max_M, s_dim, **kargs):
        self.index = index
        super().__init__(max_M, len(index), s_dim, **kargs)
        
    def getTensorQ(self, states):
        Q=th.zeros((states.shape[0],self.a_dim,self.a_dim),device = states.device)
        for i in range(self.a_dim):
            sin2 = th.sin(states[:,self.index[i]])**2
            Q[:,i,i] = sin2
        return Q

    
    def cvxpy_constraints(self, x, state = None):
        pass

    def gp_constraints(self, model, x, s):
        Sq = gp.QuadExpr()
        for i in range(self.a_dim):
            sin2 = math.sin(s[self.index[i]])**2
            Sq+=sin2*x[i]*x[i]
        model.addConstr(Sq <= self.max_M)
