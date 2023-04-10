# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
import numpy as np
from torch import nn

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, get_action_dim
    
from stable_baselines3.common.type_aliases import Schedule
from .additional_layer_actor import AdditionalLayerActor

class TruncateExtractor(BaseFeaturesExtractor):
    """
    Extractor to take original observations from concatenated observations with centers
    """
    def __init__(self, observation_space: gym.Space, a_dim: int):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space) - a_dim)
        self.flatten = nn.Flatten()
        self.a_dim = a_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)[:,:-self.a_dim]
    
class AdditionalLayerPolicy(TD3Policy):
    """
    policy for DAlpha, DRad
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = TruncateExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = {},
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        **kwargs
    ):
        self.kwargs = kwargs
        a_dim = get_action_dim(action_space)
        features_extractor_kwargs.update({"a_dim": a_dim})
        super(AdditionalLayerPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        self.actor_kwargs.update(self.kwargs)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return AdditionalLayerActor(**actor_kwargs).to(self.device)

    
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
    
