# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
import gym

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import DDPG
from .noise_insertion_ddpg import NoiseInsertionDDPG
from action_constrained_rl.utils.constant_function import ConstantFunction

class DDPGWithPenalty(NoiseInsertionDDPG):
    """
    modified DDPG to add penalty to violation of constraints of outputs before final layer
    This class is used for DOpt+
    """
    def __init__(self, constraint, *args, use_center_wrapper:bool = True, constraint_penalty = ConstantFunction(0), **kwargs):
        super(DDPGWithPenalty, self).__init__(*args, use_center_wrapper = use_center_wrapper, **kwargs)
        self.constraint = constraint
        self.penalty_coeff = constraint_penalty

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
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

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
                outputs = self.actor(replay_data.observations)
                before_projection = (self.actor.forward_before_projection(replay_data.observations))

                outputs.retain_grad()
                actor_loss = -self.critic.q1_forward(replay_data.observations, outputs).mean()
                
                actor_loss += self.penalty_coeff(self.num_timesteps) * self.constraint.constraintViolationBatch(replay_data.observations, outputs).mean()

                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                for tag, value in self.actor.named_parameters():
                    if value.grad is not None:
                        self.logger.record("train/grad/" + tag, np.linalg.norm(value.grad.cpu().numpy()))

                action_grad = outputs.grad.norm(dim=1).sum()
                self.logger.record("train/action_grad", action_grad.item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
