# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import numpy as np
import math
import gym
import torch as th

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import TD3

from stable_baselines3.common.preprocessing import get_action_dim
from ..nn.additional_layers.alpha_projection import AlphaProjectionLayer
from torch.nn import functional as F

from stable_baselines3.common.utils import polyak_update

class NoiseInsertionTD3(TD3):
    """
    TD3 to project random samplied actions and to add noise before final layer
    """
    def __init__(self, policy, env, use_center_wrapper:bool = True, **kwargs):
        super(NoiseInsertionTD3, self).__init__(policy, env, **kwargs)
        self.action_dim = get_action_dim(self.action_space)
        self.alpha_layer = AlphaProjectionLayer(env.envs[0].constraint)
        self.use_center_wrapper = use_center_wrapper

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
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            scaled_action = self.policy.scale_action(unscaled_action)
            # project actions
            if self.use_center_wrapper: # use alpha projection
                scaled_action = np.array([self.env.envs[i].constraint.project(self._last_obs[i,:-self.action_dim], self._last_obs[i,-self.action_dim:], scaled_action[i]) for i in range(n_envs)])
            else: # use closest-point projection
                scaled_action = np.array([self.env.envs[i].constraint.enforceConstraintIfNeed(self._last_obs[i], scaled_action[i]) for i in range(n_envs)])
        else:
            scaled_action = self.policy.actor.undeformed_predict(self._last_obs) # output before final layer
            # Add noise to the action (improve exploration)

            if action_noise is not None:
                scaled_action = scaled_action + action_noise()

            # Deform action by final layer
            scaled_action = self.policy.actor.deform_action(scaled_action, self._last_obs)

        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)

        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # instead of clamping, project by alpha projection layer
                next_actions = self.actor_target(replay_data.next_observations) + noise
                next_observations = replay_data.next_observations.to(next_actions.dtype)
                next_actions = self.alpha_layer(next_actions,
                                                next_observations[:, :-self.action_dim],
                                                next_observations[:, -self.action_dim:])

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
