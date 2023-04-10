# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
import numpy as np
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.td3.policies import Actor
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


class AdditionalLayerActor(Actor):
    """
    Actor network (policy) for DAlpha, DRad.

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
        squash_output: bool = False,
        constraint = None,
        layer_type = None,
    ):
        if constraint is None:
            raise ("constraint should not be None")

        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        self.net_arch = net_arch
        action_dim = get_action_dim(self.action_space)
        self.a_dim = action_dim
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.constraint = constraint

        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=squash_output)
        # Deterministic action
        self.first_layers = nn.Sequential(*actor_net)
        self.flatten = nn.Flatten()
        self.additional_layer = layer_type(self.constraint)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # split obs to features and centers
        features = self.extract_features(obs)
        centers = self.flatten(obs)[:,-self.a_dim:].to(features.dtype)
        output = self.first_layers(features)
        output = self.additional_layer(output, features, centers)
        return output

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
        centers = self.flatten(obs)[:,-self.a_dim:].to(features.dtype)
        actions = th.tensor(actions, device = self.device).float()
        return self.additional_layer(actions, features, centers).cpu().detach().numpy()
