# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
from .constraint import LinearConstraint
import torch

import cvxpy as cp
import gurobipy as gp

from ..cvxpy_variables import CVXPYVariables

def make_compatible(a, b):
    if a.device != b.device:
        a=a.to(b.device)
    if a.dtype != b.dtype:
        a=a.to(b.dtype)
    return a

class PowerConstraint(LinearConstraint):

    """
    State-dependent Action Constraints with the from
    $`\sum max{w_i a_i, 0} \leq M, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """

    def __init__(self, indices, scale, max_power, s_dim, **kargs):
        super().__init__(len(scale), s_dim, **kargs)
        self.indices = torch.tensor(indices)
        self.K = torch.zeros((2 ** self.a_dim -1, self.a_dim))
        self.scale = scale
        self.s_dim = s_dim
        for i in range(2 ** self.a_dim -1):
            for j in range(self.a_dim):
                if i // (2 ** j) % 2 == 0:
                    self.K[i,j] = scale[j]
        self.max_power = max_power

        self.d_value = torch.hstack((self.max_power * torch.ones(self.K.shape[0]), torch.ones(2*self.a_dim)))

    def tensor_C(self, state):
        size = state.shape[0]
        device = state.device
        self.K = make_compatible(self.K, state)
        if self.indices.device != state.device:
            self.indices=self.indices.to(state.device)
        C = self.K[None, :, :] * torch.index_select(state, 1, self.indices)[:,None,:]
        eyes = torch.eye(self.a_dim, device = device).repeat(size,1,1) 
        return torch.concat((C, eyes, -eyes), axis = 1)
    
    def tensor_d(self, state):
        size = state.shape[0]
        device = state.device
        self.d_value = make_compatible(self.d_value, state)
        return self.d_value.repeat(size, 1)


    def numConstraints(self):
        return self.K.shape[0] + 2 * self.a_dim

    def E(self, state):
        return self.C(state)

    def f(self, state):
        return self.d(state)
    
    def cvxpy_constraints(self, x, state):
        cons = [sum([cp.maximum(self.scale[j] * x[j] * state[self.indices[j].item()], 0.) for j in range(self.a_dim)]) <= self.max_power]
        for i in range(self.a_dim):
            cons.append(x[i] <= 1.)
            cons.append(-x[i] <= 1.)
        return cons

    def gp_constraints(self, model, x, s):
        max_vars = []
        for i in range(self.a_dim):
            mul_var = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,
                                   vtype = gp.GRB.CONTINUOUS)
            model.addConstr(mul_var == self.scale[i]*x[i]*s[self.indices[i].item()])
            max_var = model.addVar(lb=0, ub = gp.GRB.INFINITY,
                                   vtype = gp.GRB.CONTINUOUS)
            model.addConstr(max_var == gp.max_(mul_var, 0))
            max_vars.append(max_var)
        model.addConstr(sum(max_vars) <= self.max_power)
        
class OrthoplexConstraint(LinearConstraint):

    """
    State-dependent Action Constraints with the from
    $`\sum |w_i a_i| \leq M, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """

    def __init__(self, indices, scale, max_power, s_dim, **kargs):
        super().__init__(len(scale), s_dim, **kargs)
        self.indices = torch.tensor(indices)
        self.K = torch.zeros((2 ** self.a_dim, self.a_dim))
        self.scale = scale
        self.s_dim = s_dim
        for i in range(2 ** self.a_dim):
            for j in range(self.a_dim):
                if i // (2 ** j) % 2 == 0:
                    self.K[i,j] = scale[j]
                else:
                    self.K[i,j] = -scale[j]
        self.max_power = max_power

        self.d_value = torch.hstack((self.max_power * torch.ones(self.K.shape[0]), torch.ones(2*self.a_dim)))

    def tensor_C(self, state):
        size = state.shape[0]
        device = state.device
        self.K = make_compatible(self.K, state)
        if self.indices.device != state.device:
            self.indices=self.indices.to(state.device)
        C = self.K[None, :, :] * torch.index_select(state, 1, self.indices)[:, None, :]
        eyes = torch.eye(self.a_dim, device = device).repeat(size,1,1) 
        return torch.concat((C, eyes, -eyes), axis = 1)
    
    def tensor_d(self, state):
        size = state.shape[0]
        device = state.device
        self.d_value = make_compatible(self.d_value, state)
        return self.d_value.repeat(size, 1)


    def numConstraints(self):
        return self.K.shape[0] + 2 * self.a_dim

    def E(self, state):
        return self.C(state)

    def f(self, state):
        return self.d(state)

    def cvxpy_constraints(self, x, state):
        cons = [sum([cp.abs(self.scale[j] * x[j] * state[self.indices[j].item()]) for j in range(self.a_dim)]) <= self.max_power]
        for i in range(self.a_dim):
            cons.append(x[i] <= 1.)
            cons.append(-x[i] <= 1.)
        return cons

    def gp_constraints(self, model, x, s):
        abs_vars = []
        for i in range(self.a_dim):
            mul_var = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,
                                   vtype = gp.GRB.CONTINUOUS)
            model.addConstr(mul_var == self.scale[i]*x[i]*s[self.indices[i].item()])
            abs_var = model.addVar(lb=0, ub = gp.GRB.INFINITY,
                                   vtype = gp.GRB.CONTINUOUS)
            model.addGenConstrAbs(abs_var, mul_var)
            abs_vars.append(abs_var)
        model.addConstr(sum(abs_vars) <= self.max_power)
    
class DecelerationConstraint(LinearConstraint):

    """
    State-dependent Action Constraints with the from
    $`\sum w_i a_i \leq M - \sum |w_i|, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """

    def __init__(self, indices, scale, max_power, s_dim, **kargs):
        super().__init__(len(scale), s_dim, **kargs)
        self.indices = torch.tensor(indices)
        self.scale = torch.tensor(scale)
        self.s_dim = s_dim
        self.max_power = max_power

    def tensor_C(self, state):
        size = state.shape[0]
        device = state.device
        self.scale = make_compatible(self.scale, state)
        if self.indices.device != state.device:
            self.indices=self.indices.to(state.device)
        C = (self.scale[None,:] * torch.index_select(state, 1, self.indices)).unsqueeze(1)
        eyes = torch.eye(self.a_dim, device = device).repeat(size,1,1) 
        return torch.concat((C, eyes, -eyes), axis = 1)
    
    def tensor_d(self, state):
        size = state.shape[0]
        device = state.device
        self.scale = make_compatible(self.scale, state)
        d = (self.max_power * torch.ones(size, device = device) - torch.abs(self.scale[None,:] * torch.index_select(state, 1, self.indices)).sum(1)).unsqueeze(1)
        return torch.concat((d, torch.ones((size, 2*self.a_dim), device = device)), 1)


    def numConstraints(self):
        return 1 + 2 * self.a_dim

    def E(self, state):
        return self.C(state)

    def f(self, state):
        return self.d(state)

    def cvxpy_variables(self):
        q = cp.Parameter(self.a_dim)  # input action

        s = cp.Parameter(self.s_dim)  # input: state

        x = cp.Variable(self.a_dim)  # output
        obj = cp.Minimize(0.5*cp.sum_squares(x) - q.T @ x)
        cons = []
        for i in range(1<<self.a_dim):
            sg = []
            for j in range(self.a_dim):
                if i // (2 ** j) % 2 == 0:
                    sg.append(1)
                else:
                    sg.append(-1)
            cons.append(sum([x[j]*self.scale[j]*s[self.indices[j].item()] for j in range(self.a_dim)])
                        <= self.max_power - sum([sg[j]*self.scale[j]*s[self.indices[j].item()] for j in range(self.a_dim)]))
        prob = cp.Problem(obj, cons)
        return CVXPYVariables(x, q, s, cons, obj, prob)

if __name__ == "__main__":
    import numpy as np
    cons = PowerConstraint(6, (1., 1.), 1., 11)
    action = np.random.rand(2)
    state = 5*np.random.rand(11)

    print(cons.get_center(state))
    x = cp.Variable(2)  # output
    r = cp.Variable()
    obj = cp.Maximize(r)
    C = cons.C(state)
    d = cons.d(state)
    norm=np.linalg.norm(C, axis =1)
    cons = [C @ x + norm * r <= d]
    prob = cp.Problem(obj, cons)
    prob.solve(solver = cp.GUROBI)
    print(x.value)
    exit()
    
    p_action = cons.enforceConstraintIfNeed(state, action)
    print(p_action, 0.5*np.sum(p_action**2)-p_action.dot(action), p_action.dot(state[6:8]))
    
    x = cp.Variable(2)  # output
    obj = cp.Minimize(0.5*cp.sum_squares(x) - action.T @ x)
    cons = [cp.maximum(x[0]*state[6],0.)+cp.maximum(x[1]*state[7],0)<=1., x[0]<=1., -x[0] <= 1., x[1]<=1., -x[1]<=1.]
    prob = cp.Problem(obj, cons)
    prob.solve(solver = cp.GUROBI)
    print(x.value, 0.5*np.sum(x.value**2)-x.value.dot(action), x.value.dot(state[6:8]))
    
