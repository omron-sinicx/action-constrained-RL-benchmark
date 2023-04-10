# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .constraint import Constraint, to_tensors
import torch as th
import numpy as np
import math
from abc import abstractmethod

import cvxpy as cp
import gurobipy as gp

from ..cvxpy_variables import CVXPYVariables
from .power_constraint import make_compatible

class QuadraticConstraint(Constraint):

    """
    State-dependent Action Constraints with the from
    $`\sum Q_ij a_i a_j \leq M`$ 
    """

    def __init__(self, max_M, a_dim, s_dim, **kargs):
        super().__init__(a_dim, s_dim, **kargs)
        self.max_M = max_M
        self.sr_max_M = math.sqrt(max_M)

    @abstractmethod
    def getTensorQ(self, state):
        pass
    
    def getQ(self, state):
        if isinstance(state, np.ndarray):
            return self.getTensorQ(to_tensors(state)).numpy()[0]
        return self.getTensorQ(state)    

    def isConstraintSatisfied(self, state, a, err=1e-2):
        Q = self.getQ(state)
        return a.transpose()@Q@a <= self.max_M + err

    def enforceConstraint(self, state, a):
        Q = self.getQ(state)
        value = a.transpose()@Q@a
        if value <= self.max_M:
            return a
        else:
            return math.sqrt(self.max_M / value) * a

    def numConstraints(self):
        return 1
    
    def getL(self, states, centers, v, get_grad:bool = False):
        Q = self.getQ(states)
        value = (v[:,:,None]*Q*v[:,None,:]).sum(dim=2).sum(dim=1).clamp(min=1e-3)
        L = th.sqrt(value/self.max_M)
        
        if not get_grad:
            return L
        else:
            return L, (Q*v[:,None,:]).sum(dim=2)/L[:,None]/self.max_M

    def constraintViolation(self, state, a, err=1e-3, normalize=False):
        Q = self.getQ(state)
        scale = np.sqrt(self.a_dim / np.trace(Q)+1e-6)
        value = a.transpose()@Q@a
        return np.expand_dims(np.maximum(0.0, scale*(np.sqrt(value) - self.sr_max_M) - err),0)
                
    def constraintViolationBatch(self, states, actions):
        Q = self.getQ(states)
        scale = th.sqrt(self.a_dim / Q.diagonal(dim1=1, dim2=2).sum(axis=1)[:,None,None]+1e-6)
        value = (actions[:,:,None]*Q*actions[:,None,:]).sum(dim=2).sum(dim=1).clamp(min=1e-3)
        return th.maximum(scale*(th.sqrt(value)-self.sr_max_M), th.tensor(0.))

    def get_center(self, state):
        return np.zeros(self.a_dim)
