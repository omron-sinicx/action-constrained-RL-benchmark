# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
import torch as th
import numpy as np
import sys

from torch.special import erf

sqrt2 = np.sqrt(2.)
log2 = np.log(2.)
sqrtpi = np.sqrt(np.pi)
sqrt2pi = sqrt2*sqrtpi
lgsqrt2pi = np.log(sqrt2pi)
lg2sqrtpi = np.log(2*sqrtpi)

LOG_STD_MAX = 2
# LOG_PROB_MAX = 1e6

def radial_integral_gaussian(means, log_std, centers, v, epsilon:float = 1e-300):
    # calculate integration of gaussian distribution
    
    dtp=means.dtype
    log_std=log_std.to(th.float64)
    v=v.to(th.float64)
    dim = means.shape[1]
    std = log_std.exp()
    a = v / std / sqrt2
    b = (centers-means).to(th.float64) / std / sqrt2
    A = a.norm(dim=1)
    na = a / A[:,None]
    B = (na*b).sum(dim=1)
    Cross = na[:,:,None]*b[:,None,:]-b[:,:,None]*na[:,None,:]
    C = 1./2. * Cross.square().sum(dim=(1,2)) # C = (b*b).sum(axis=1) - B*B

    log_prob = dim*th.log(v.norm(dim=1)+epsilon) - log_std.sum(dim=1) - dim*lgsqrt2pi - C
    IA = A**-1
    BA = IA*B
    D = IA*th.exp(-(A+B)**2)
    E = sqrtpi * (erf(A+B)-1.)
    if dim == 1:
        log_prob += th.log(-IA*E + epsilon) - log2
    elif dim == 2:
        log_prob += th.log(IA*D + IA*BA * E + epsilon) - log2
    elif dim == 3:
        log_prob += th.log(2 * IA*(IA-BA)*D - IA*(2 * BA ** 2 + IA**2) * E + epsilon) - 2 * log2
    elif dim == 4:
        log_prob += th.log(2 * IA*(IA**2 + 1 - BA + BA**2)*D + IA*BA*(2 * BA ** 2 + 3*IA**2) * E + epsilon) - 2 * log2
    elif dim == 5:
        log_prob += th.log((6 * IA**3 + 4*IA - 10 * IA**3*BA - 4 * IA*BA + 4*IA*BA**2-4*IA*BA**3)*D - (3*IA**5+ 12 * IA**3*BA ** 2 + 4*IA*BA**4) * E + epsilon) - 3 * log2
    elif dim == 6:
        log_prob += th.log(2*IA*(2 + (2+9*IA**2)*BA**2-(2+7*IA**2)*BA+4*IA**2-2*BA**3+2*BA**4+4*IA**4) * D + IA*BA*(4*BA**4+20*IA**2*BA**2+15*IA**4) * E + epsilon) - 3 * log2
    else:
        raise ValueError("We do not implemented for this action space dimension")
     
    return log_prob.to(dtp)

class AlphaGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a alpha projection to ensure bounds.

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int, constraint, epsilon: float = 1e-300):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.cons = constraint
        self.gaussian_actions = None


    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None, deterministic:bool = False, calc_prob:bool = False) -> th.Tensor:
        assert states != None and centers != None
        self.proba_distribution(mean_actions, log_std)
        if deterministic:
            self.gaussian_actions = super().mode()
        else:
            self.gaussian_actions = super().sample()
        self.v = self.gaussian_actions - centers
        if not calc_prob:
            self.L = self.cons.getL(states, centers, self.v)
        else:
            self.L, self.gradL = self.cons.getL(states, centers, self.v, get_grad = True)
        actions = centers + self.v / th.maximum(self.L, th.tensor(1.))[:,None]
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        assert states != None and centers != None
        actions = self.actions_from_params(mean_actions, log_std, states, centers, calc_prob = True)
        inside = th.lt(self.L, 1.)
        inside_log_prob = super().log_prob(self.gaussian_actions) # log_prob for inside of constraints

        # calculate log_prob for outside of constraints

        # calculate measure unit
        v_norm = self.v.norm(dim=1)
        coss = (self.gradL*self.v).sum(axis=1)/self.gradL.norm(dim=1)/v_norm
        darea = v_norm/self.L/coss

        outside_log_prob = radial_integral_gaussian(mean_actions, log_std, centers, self.v/self.L[:,None], self.epsilon) - th.log(darea)

        log_prob = th.where(inside, inside_log_prob, outside_log_prob)
        return actions, log_prob

class AlphaStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a alpha projection to ensure bounds.
    State dependent version

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int, constraint, _epsilon: float = 1e-300, **kargs):
        super().__init__(action_dim, **kargs)
        # Avoid NaN (prevents division by zero or log of zero)
        self._epsilon = _epsilon
        self.cons = constraint
        self.gaussian_actions = None


    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None, deterministic:bool = False, calc_prob:bool = False) -> th.Tensor:
        assert states != None and centers != None
#        mean_actions = th.where(mean_actions.isnan(), centers, mean_actions)
#        log_std = th.where(log_std.isnan(), LOG_STD_MAX, log_std)
        self.proba_distribution(mean_actions, log_std, latent_sde)
        if deterministic:
            self.gaussian_actions = super().mode()
        else:
            self.gaussian_actions = super().sample()
        self.v = self.gaussian_actions - centers
        if not calc_prob:
            self.L = self.cons.getL(states, centers, self.v)
        else:
            self.L, self.gradL = self.cons.getL(states, centers, self.v, get_grad = True)
        actions = centers + self.v / th.maximum(self.L, th.tensor(1.))[:,None]
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        assert states != None and centers != None
        actions = self.actions_from_params(mean_actions, log_std, latent_sde, states, centers, calc_prob = True)
        inside = th.lt(self.L, 1.)
        inside_log_prob = super().log_prob(self.gaussian_actions) # log_prob for inside of constraints

        # calculate log_prob for outside of constraints

        # calculate measure unit
        v_norm = self.v.norm(dim=1)
        coss = (self.gradL*self.v).sum(axis=1)/self.gradL.norm(dim=1)/v_norm
        darea = v_norm/self.L/coss

        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        distribution_std = th.sqrt(variance + self.epsilon)

        outside_log_prob = radial_integral_gaussian(mean_actions, distribution_std, centers, self.v/self.L[:,None], self._epsilon) - th.log(darea)

        log_prob = th.where(inside, inside_log_prob, outside_log_prob)
            
        return actions, log_prob

if __name__ == "__main__":
    from ...half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint
    from ...constraint.power_constraint import PowerConstraint
    from torch import autograd

    cons = PowerConstraint(11, (1,1,1,1,1,1), 10.,17)
    dist = AlphaGaussianDistribution(6, constraint = cons)
    N=1
    mean_actions = th.tensor([[-6.2313e+08,  3.6953e+08,  5.6804e+08,  3.2498e+08, -5.9962e+08,
         3.3148e+08]], dtype = th.float32, requires_grad=True)
    log_std = th.tensor([[  2.,   2.,   2., -20.,   2., -20.]], dtype = th.float32, requires_grad=True)
    states = th.tensor([[  1.1587,  -0.6799,  -0.7305,  -0.2797,   0.6688,  -0.4330,  -0.3629,
          0.6346,  -1.9232,  -2.1278, -10.0000, -10.0000,   7.1536,  10.0000,
        -10.0000,  -9.6341, -10.0000]], dtype = th.float32)
    centers = th.tensor([[ 0.1698, -0.1698, -0.1698,  0.1698,  0.1698,  0.1698]], dtype = th.float32)
    actions, log_prob = dist.log_prob_from_params(mean_actions, log_std, states = states, centers = centers)
    print(actions, log_prob)
    with autograd.detect_anomaly():
        log_prob.backward()
    print(mean_actions.grad, log_std.grad)
    exit()

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    probs = log_prob.exp().numpy()
    colors = [[1,0,0,min(i,10.)/10.] for i in probs]
    plt.scatter(actions[:,0].numpy(), actions[:,1].numpy(), c = colors)
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
