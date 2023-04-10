# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
import numpy as np
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

from .opt_layer import OptLayer

class OptLayerActor(BasePolicy):
    """
    Actor network (policy) for DOpt.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        squash_output = True,
        constraint = None,
    ):
        if constraint is None:
            raise ("constraint should not be None")

        super(OptLayerActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.constraint = constraint

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=squash_output)
        # Deterministic action
        self.first_layers = nn.Sequential(*actor_net)
        self.opt_layer = OptLayer(self.constraint)
        self.tanh = nn.Tanh()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        output = self.first_layers(features)
        output = self.opt_layer(output, features)
        return output

    def forward_before_projection(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        output = self.first_layers(features)
        return output

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation)

    def undeformed_predict(self, obs: np.ndarray) -> th.Tensor:
        # return output before the final layer
        obs, _ = self.obs_to_tensor(obs)
        features = self.extract_features(obs)
        output = self.first_layers(features)
        return output.cpu().detach().numpy()

    def deform_action(self, actions: np.ndarray, obs: np.ndarray) -> th.Tensor:
        # apply the final layer
        obs, _ = self.obs_to_tensor(obs)
        features = self.extract_features(obs)
        actions = th.tensor(actions, device = self.device).float()
        return self.opt_layer(actions, features).cpu().detach().numpy()

if __name__ == "__main__":
    import numpy as np
    from ...half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint
    
    env = gym.make("HalfCheetah-v2")
    cons = HalfCheetahDynamicConstraint()

    actor = OptLayerActor(env.observation_space, env.action_space, [30, 20], FlattenExtractor(env.observation_space), 17, squash_output=False, constraint = cons)
    print(env.observation_space)
    obs = th.ones(1, 17)
    output = actor(obs)
    output.sum().backward()
    for param in actor.parameters():
        print(param)
        print(param.grad)
