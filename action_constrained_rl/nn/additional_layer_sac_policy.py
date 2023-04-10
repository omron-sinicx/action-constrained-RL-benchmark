# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
import numpy as np
from torch import nn

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)

from .additional_layer_policy import TruncateExtractor

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class AdditionalLayerSACActor(Actor):

    """
    Actor for SAlpha or SRad
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        distribution_class = None, # deformed distribution class
        constraint = None
    ):
        assert distribution_class != None
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        action_dim = get_action_dim(self.action_space)
        self.a_dim = action_dim
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = distribution_class(
                action_dim, constraint, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=False
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = distribution_class(action_dim, constraint)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
        self.flatten = nn.Flatten()

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:

        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        # split obs to features and centers
        features = self.extract_features(obs)
        centers = self.flatten(obs)[:,-self.a_dim:].to(features.dtype)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return mean_actions, self.log_std, dict(states=features, centers=centers, latent_sde=latent_pi)

        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {"states": features, "centers": centers}

        
class AdditionalLayerSACPolicy(SACPolicy):
    """
    Policy for SAlpha or SRad
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = TruncateExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = {},
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        **kwargs
    ):
        self.additional_actor_kwargs = kwargs
        a_dim = get_action_dim(action_space)
        features_extractor_kwargs.update({"a_dim": a_dim})
        super(AdditionalLayerSACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(self.additional_actor_kwargs)
        return AdditionalLayerSACActor(**actor_kwargs).to(self.device)
    

if __name__ == "__main__":
    import numpy as np
    from stable_baselines3 import DDPG
    from ..ddpg.projection_ddpg import ProjectionDDPG
    from ..half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from ..env_wrapper.constraint_wrapper import ConstraintEnvWrapper
    
    cons = HalfCheetahDynamicConstraint()
    env = gym.make("HalfCheetah-v2")
    env = ConstraintEnvWrapper(cons, env)
    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.01 * np.ones(n_actions))
    model = ProjectionDDPG(cons, AdditionalLayerPolicy, env, action_noise=action_noise, verbose=2, batch_size=8, policy_kwargs = {"constraint": cons, "layer_type": None})
    model.learn(total_timesteps=1000)
    
