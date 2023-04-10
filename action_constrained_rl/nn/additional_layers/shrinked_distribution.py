# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
import torch as th
import numpy as np

from torch.special import erf

class ShrinkedGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a radial shrinkage to ensure bounds.

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int, constraint, _epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self._epsilon = _epsilon
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
        self.tanhL = th.tanh(self.L)
        actions = centers + (self.tanhL / (self.L+1e-9))[:,None] * self.v
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        assert states != None and centers != None
        device = mean_actions.device
        actions = self.actions_from_params(mean_actions, log_std, states, centers, calc_prob = True)
        log_prob = super().log_prob(self.gaussian_actions)

        # calculate gradient of L
        gradsc = ((self.L*(1-self.tanhL**2)-self.tanhL)/((self.L+1e-9)**2))[:,None] * self.gradL
        jacob = (self.tanhL/(self.L+1e-9))[:,None,None] * th.eye(self.action_dim, device = device)[None,:,:]
        jacob += gradsc[:,None,:] * self.v[:,:,None]
        log_prob -= th.log(th.linalg.det(jacob)+self._epsilon)
        return actions, log_prob

class ShrinkedStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a radial shrinkage to ensure bounds.
    State dependent version
    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int, constraint, _epsilon: float = 1e-6, **kargs):
        super().__init__(action_dim, **kargs)
        # Avoid NaN (prevents division by zero or log of zero)
        self._epsilon = _epsilon
        self.cons = constraint
        self.gaussian_actions = None


    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None, deterministic:bool = False, calc_prob:bool = False) -> th.Tensor:
        assert states != None and centers != None
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
        self.tanhL = th.tanh(self.L)
        actions = centers + (self.tanhL / (self.L+1e-9))[:,None] * self.v
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, states: th.Tensor = None, centers: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        assert states != None and centers != None
        device = mean_actions.device
        actions = self.actions_from_params(mean_actions, log_std, latent_sde, states, centers, calc_prob = True)
        log_prob = super().log_prob(self.gaussian_actions)

        # calculate gradient of L
        gradsc = ((self.L*(1-self.tanhL**2)-self.tanhL)/((self.L+1e-9)**2))[:,None] * self.gradL
        jacob = (self.tanhL/(self.L+1e-9))[:,None,None] * th.eye(self.action_dim, device = device)[None,:,:]
        jacob += gradsc[:,None,:] * self.v[:,:,None]
        log_prob -= th.log(th.linalg.det(jacob)+self._epsilon)
        return actions, log_prob

if __name__ == "__main__":

    from .visualize import TestConstraint

    cons = TestConstraint()
    dist = ShrinkedGaussianDistribution(2, constraint = cons)
    N=1000
    mean_actions = th.tensor([[0.,0.]], dtype = th.float32).repeat(N,1)
    log_std = th.tensor([[-1,-1]], dtype = th.float32).repeat(N,1)
    states = th.zeros((N,2))
    centers = th.zeros((N,2))
    actions, log_prob = dist.log_prob_from_params(mean_actions, log_std, states = states, centers = centers)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    probs = log_prob.exp().numpy()
    colors = [[1,0,0,min(i,10.)/10.] for i in probs]
    plt.scatter(actions[:,0].numpy(), actions[:,1].numpy(), c = colors)
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
