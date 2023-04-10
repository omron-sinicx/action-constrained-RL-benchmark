# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .constraint import LinearConstraint
import torch as th

import cvxpy as cp
import gurobipy as gp

import math

class MAConstraint(LinearConstraint):

    """
    State-dependent Action Constraints with the from
    $w_0a_0\sin(theta_0+theta_1+theta_2)+w_3a_3\sin(\theta_3+\theta_4+\theta_5) \leq M, |a_i| \leq 1`$
    """

    def __init__(self, max_power, **kargs):
        super().__init__(6, 17, **kargs)
        self.max_power = max_power

        self.d_value = th.hstack((self.max_power * th.ones(1), th.ones(2*self.a_dim)))

    def tensor_C(self, state):
        size = state.shape[0]
        device = state.device
        C = th.zeros((size, 1, 6), device = device)
        C[:,0,0] = state[:,11]*th.sin(state[:,2]+state[:,3]+state[:,4])
        C[:,0,3] = state[:,14]*th.sin(state[:,5]+state[:,6]+state[:,7])
        eyes = th.eye(self.a_dim, device = device).repeat(size,1,1) 
        return th.concat((C, eyes, -eyes), axis = 1)
    
    def tensor_d(self, state):
        size = state.shape[0]
        if self.d_value.device != state.device:
            self.d_value = self.d_value.to(state.device)
        return self.d_value.repeat(size, 1)

    def numConstraints(self):
        return 1 + 2 * self.a_dim

    def E(self, state):
        return self.C(state)
    
    def f(self, state):
        return self.d(state)
    
    def cvxpy_constraints(self, x, state):
        pass
    
    def numConstraints(self):
        return 1 + 2 * self.a_dim

    def gp_constraints(self, model, x, s):
        model.addConstr(s[11]*math.sin(s[2]+s[3]+s[4])*x[0]
                        + s[14]*math.sin(s[5]+s[6]+s[7])*x[3]<= self.max_power)
