# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
import numpy as np
import math
import gym
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import DDPG

from action_constrained_rl.utils.constant_function import ConstantFunction

class ProjectionDDPG(DDPG):
    """
    class for DPro
    """
    def __init__(self, constraint, *args, constraint_penalty = ConstantFunction(0), **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint
        self.penalty_coeff = constraint_penalty

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
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            obs = self._last_obs[-1]
            #print("unscaled: {}".format(unscaled_action))
            scaled_action = self.policy.scale_action(unscaled_action)
            #print("scaled: {}".format(scaled_action))

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = scaled_action + action_noise()

            scaled_action = np.array([self.env.envs[i].constraint.enforceConstraintIfNeed(self._last_obs[i], scaled_action[i]) for i in range(n_envs)])
            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
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
                outputs.retain_grad()
                actor_loss = -self.critic.q1_forward(replay_data.observations, outputs).mean()

                # calculate penaty
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

if __name__ == "__main__":
    import gym
    import numpy as np
    import os

    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

    video_folder = 'log/videos/'
    video_length = 1000

    env = gym.make("IdealizedPendulum-v0", max_speed=8.0, max_torque=2.0, normalization_factor=100.0, l=1.0, initial_state=np.array([0.25 * np.pi, 1.0]), dt=0.01)

#    env = DummyVecEnv([lambda: env])
#    env = VecVideoRecorder(env, video_folder,
#                       record_video_trigger=lambda x: x == 0, video_length=video_length,
#                       name_prefix=f"ddpg-cgf-pendulum")

# The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    cbf = PendulumCBF(normalization_factor=100.0)
    model = ProjectionDDPG(cbf, 'MlpPolicy', env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=1000)
    env = model.get_env()

    del model # remove to demonstrate saving and loading
