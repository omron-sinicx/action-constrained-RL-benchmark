import numpy as np
import torch as th
import math
from abc import ABC, abstractmethod

from .normalize_constraint import normalizeConstraint
from ..nn.additional_layers.chebyshev_center import calc_chebyshev_center

import cvxpy as cp
from ..cvxpy_variables import CVXPYVariables

import gurobipy as gp

def to_tensors(a):
    return th.tensor(np.expand_dims(a,0))

EPS=1e-9

class Constraint(ABC):
    """
    Abstract class for all action constraints
    """
    def __init__(self, a_dim:int, s_dim:int = -1):
        self.a_dim = a_dim
        self.s_dim = s_dim
    @abstractmethod
    def isConstraintSatisfied(self, state, a):
        pass

    def enforceConstraint(self, state, a):
        with gp.Model() as model:
            x = []
            for _ in range(self.a_dim):
                x.append(model.addVar(lb=-1, ub =1, vtype = gp.GRB.CONTINUOUS))
            obj = gp.QuadExpr()
            for i in range(self.a_dim):
                obj+=(0.5*x[i]-a[i])*x[i]
            model.setObjective(obj, sense = gp.GRB.MINIMIZE)

            self.gp_constraints(model, x, state)
            
            model.optimize()

            x_value = np.array(model.X[0:self.a_dim])

        return x_value

    def enforceConstraintIfNeed(self, state, a):
        if self.isConstraintSatisfied(state, a):
            return a
        else:
            return self.enforceConstraint(state, a)
    
    @abstractmethod
    def numConstraints(self):
        pass
    
    @abstractmethod
    def getL(self, states, centers, v, get_grad:bool = False):
        pass

    @abstractmethod
    def get_center(self, state):
        pass

    def project(self, state, center, a):
        L = self.getL(to_tensors(state), to_tensors(center), to_tensors(a-center)).numpy()[0]
        return center + (a-center) / max(L, 1)

    @abstractmethod
    def cvxpy_constraints(self, x, state = None):
        pass

    @abstractmethod
    def gp_constraints(self, x, state = None):
        pass

    def cvxpy_variables(self):
        q = cp.Parameter(self.a_dim)  # input action

        if self.s_dim > 0:
            s = cp.Parameter(self.s_dim)  # input: state
        else:
            s = None

        x = cp.Variable(self.a_dim)  # output
        obj = cp.Minimize(0.5*cp.sum_squares(x) - q.T @ x)
        cons = self.cvxpy_constraints(x, s)
        prob = cp.Problem(obj, cons)
        return CVXPYVariables(x, q, s, cons, obj, prob)
    
class LinearConstraint(Constraint):
    """
    Abstract class for linear constraints with:
        Ax = b,
        Cx <= d
    When inequality representations are needed, A, b, C, and d are aggregated into Ex <= f
    """
    def __init__(self, a_dim:int, s_dim:int = -1, proj_type: str = "QP"):
        super().__init__(a_dim, s_dim)
        self.proj_type = proj_type
        
    @abstractmethod
    def E(self, state):
        pass
  
    @abstractmethod
    def f(self, state):
        pass

    def C(self, state):
        if isinstance(state, np.ndarray):
            return self.tensor_C(to_tensors(state)).numpy()[0]
        return self.tensor_C(state)
  
    def d(self, state):
        if isinstance(state, np.ndarray):
            return self.tensor_d(to_tensors(state)).numpy()[0]
        return self.tensor_d(state)

    @abstractmethod
    def tensor_C(self, state):
        pass
  
    @abstractmethod
    def tensor_d(self, state):
        pass

    def getL(self, states, centers, v, get_grad:bool = False):
        C = self.tensor_C(states)
        d = self.tensor_d(states)
        div = d-(centers[:,None,:]*C).sum(axis=2)
        maxs = ((v[:,None,:]*C).sum(axis=2) / (div+EPS)).max(axis = 1)
        if not get_grad:
            return maxs[0]
        else:
            indices_2 = maxs[1][:,None].expand(-1,1)
            indices_3 = maxs[1][:,None,None].expand(-1,1,C.shape[2])
            gradL = th.gather(C, 1, indices_3).squeeze(dim=1) / th.gather(div, 1, indices_2).squeeze(dim=1)[:,None]
            return maxs[0], gradL

    def get_center(self, state):
        C = self.C(state)
        d = self.d(state)
        return calc_chebyshev_center(C,d)

    def A(self, state):
        None

    def b(self, state):
        None

    def isConstraintSatisfied(self, state, a, err=1e-1):
        if state.ndim == 2:
            return ((self.E(state) * a[:,None,:]).sum(axis = 2) <= self.f(state) + err).all()
        else:
            return (np.matmul(self.E(state), a) <= self.f(state) + err).all()

    def enforceConstraint(self, state, a):
        if self.proj_type == "QP":
            return super().enforceConstraint(state, a)

        elif self.proj_type == "alpha":
            C = self.C(state)
            d = self.d(state)
            center = calc_chebyshev_center(C,d)
            v = a - center
            return center + v / max((C @ v / (d - C @ center) ).max(), 1)
        elif self.proj_type == "shrinkage":
            C = self.C(state)
            d = self.d(state)
            center = calc_chebyshev_center(C,d)
            v = a - center
            L = (C @ v / (d - C @ center) ).max()
            return center + math.tanh(L) / L * v
        else:
            raise ValueError(self.proj_type)
                
    def constraintViolation(self, state, a, err=1e-2, normalize=False):
        if normalize:
            E = self.E(state)
            f = self.f(state)
            normalized_E, normalized_f = normalizeConstraint(E, f)
            return np.maximum(0.0, np.matmul(normalized_E, a) - normalized_f - err)
        else:
            return np.maximum(0.0, np.matmul(self.E(state), a) - self.f(state) - err)

    def constraintViolationBatch(self, states, actions):
        C = self.tensor_C(states)
        d = self.tensor_d(states)
        return th.maximum(((C*actions[:,None,:]).sum(dim=2)-d)/(C.norm(dim=2)+EPS),th.tensor(0.)).norm(dim=1)

        
if __name__ == "__main__":
    from ..half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint
    cons_pq = HalfCheetahDynamicConstraint("QP")
    cons_al = HalfCheetahDynamicConstraint("alpha")
    cons_sh = HalfCheetahDynamicConstraint("shrinkage")
    from ..nn.opt_layer.opt_layer import OptLayer
    from ..nn.additional_layers.alpha_projection import AlphaProjectionLayer
    from ..nn.additional_layers.radial_shrinkage import ShrinkageLayer
    layer_pq = OptLayer(cons_pq)
    layer_al = AlphaProjectionLayer(cons_pq)
    layer_sh = ShrinkageLayer(cons_pq)
    actions = th.rand(100, 6)
    states = th.rand(100, 17)
    y_pq = layer_pq(actions, states)
    y_al = layer_al(actions, states)
    y_sh = layer_sh(actions, states)
    for i in range(100):
        state = states[i].numpy()
        action = actions[i].numpy()
        projected_pq = cons_pq.enforceConstraint(state, action)
        projected_al = cons_al.enforceConstraint(state, action)
        projected_sh = cons_sh.enforceConstraint(state, action)
        assert cons_pq.isConstraintSatisfied(state, projected_pq) and (not cons_pq.isConstraintSatisfied(state, action) or np.allclose(action, projected_pq, rtol = 1e-3))
        assert cons_al.isConstraintSatisfied(state, projected_al) and (not cons_al.isConstraintSatisfied(state, action) or np.allclose(action, projected_al))
        assert cons_al.isConstraintSatisfied(state, projected_sh)
        assert np.allclose(y_pq[i].numpy(), projected_pq, rtol = 1e-3)
        assert np.allclose(y_al[i].numpy(), projected_al, rtol = 1e-3)
        assert np.allclose(y_sh[i].numpy(), projected_sh, rtol = 1e-3)
    #print(cons.constraintViolation(None, np.array([2.0, 0.0, 3.0, -4.0, 0.0, 5.0])))
    #print(cons.constraintViolation(None, np.array([2.0, 0.0, 3.0, -4.0, 0.0, 5.0]), normalize=True))
