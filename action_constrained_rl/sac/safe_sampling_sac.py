# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import numpy as np
import math
import gym

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_action_dim

from .logging_gradient import LoggingGradientSAC

class SafeSamplingSAC(LoggingGradientSAC):
    """
    modified SAC to project random sampled actions
    """
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            scaled_action = self.policy.scale_action(unscaled_action)
            action_dim = get_action_dim(self.action_space)
            scaled_action = np.array([self.env.envs[i].constraint.project(self._last_obs[i,:-action_dim], self._last_obs[i,-action_dim:], scaled_action[i]) for i in range(n_envs)])

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
            scaled_action = self.policy.scale_action(unscaled_action)
            buffer_action = scaled_action
            action = unscaled_action

        return action, buffer_action
