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
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import Schedule

import gurobipy as gp

def FW_update(action, state, grad, constraint, lr):
    ## Solve LP

    a_dim = action.shape[0]
    with gp.Model() as model:
        x = []
        for _ in range(a_dim):
            x.append(model.addVar(lb=-1, ub =1, vtype = gp.GRB.CONTINUOUS))
        obj = gp.LinExpr()
        for i in range(a_dim):
            obj+=grad[i]*x[i]
        model.setObjective(obj, sense = gp.GRB.MAXIMIZE)
        constraint.gp_constraints(model, x, state)
        model.optimize()
        x_value = np.array(model.X[0:a_dim])

    return x_value*lr + action * (1-lr)

class NFWPO(TD3):

    """
    TD3-based implimentation of NFWPO
    """

    def __init__(self, constraint, *args, fw_learning_rate:float = 0.01, actor_learning_rate:float = 1e-3, critic_learning_rate:float = 1e-3, **kargs):

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        super().__init__(*args, **kargs)
        self.constraint = constraint
        self.fw_learning_rate = fw_learning_rate

    def _setup_model(self) -> None:
        super()._setup_model()

        self.policy.actor.optimizer = self.policy.optimizer_class(self.policy.actor.parameters(), lr=self.actor_lr_schedule(1), **self.policy.optimizer_kwargs)
        self.policy.critic.optimizer = self.policy.optimizer_class(self.policy.critic.parameters(), lr=self.critic_lr_schedule(1), **self.policy.optimizer_kwargs)

        
    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.actor_lr_schedule = get_schedule_fn(self.actor_learning_rate)
        self.critic_lr_schedule = get_schedule_fn(self.critic_learning_rate)

        self.lr_schedule = self.actor_lr_schedule ## dummy

    def _update_learning_rate(self, optimizers: List[th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        :param optimizers:
            a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/actor_learning_rate", self.actor_lr_schedule(self._current_progress_remaining))
        self.logger.record("train/critic_learning_rate", self.critic_lr_schedule(self._current_progress_remaining))

        update_learning_rate(optimizers[0], self.actor_lr_schedule(self._current_progress_remaining))
        update_learning_rate(optimizers[1], self.critic_lr_schedule(self._current_progress_remaining))

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
                lr = self.fw_learning_rate

                outputs = self.actor(replay_data.observations)

                # Let actions in constraints
                actions = outputs.cpu().detach().numpy()                
                states = replay_data.observations.cpu().detach().numpy()
                for i in range(batch_size):
                    actions[i] = self.constraint.enforceConstraintIfNeed(states[i], actions[i])
        
                # Compute Q value grad
                action_tensors = th.tensor(actions, device = outputs.device, dtype = outputs.dtype, requires_grad = True)
                q_value = self.critic.q1_forward(replay_data.observations, action_tensors).mean()
                q_value.backward()
                grads = action_tensors.grad.cpu().detach().numpy()

                # Compute optimized action in CPU
                action_table = np.zeros(actions.shape)
                for i in range(batch_size):
                    action_table[i]=FW_update(actions[i], states[i], grads[i], self.constraint, lr)
                action_table = th.tensor(action_table, device = outputs.device, dtype = outputs.dtype)

                outputs.retain_grad()
                actor_loss = F.mse_loss(outputs, action_table)
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                for tag, value in self.actor.named_parameters():
                    if value.grad is not None:
                        self.logger.record("train/grad/" + tag, np.linalg.norm(value.grad.cpu().detach().numpy()))

                action_grad = outputs.grad.norm(dim=1).sum()
                self.logger.record("train/action_grad", action_grad.item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

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
        else:
            unscaled_action, _ = self.policy.predict(self._last_obs)
            scaled_action = self.policy.scale_action(unscaled_action)
            # Add noise to the action (improve exploration)

            if action_noise is not None:
                scaled_action = scaled_action + action_noise()

        scaled_action = np.array([self.env.envs[i].constraint.enforceConstraintIfNeed(self._last_obs[i], scaled_action[i]) for i in range(n_envs)])
        
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)
        return action, buffer_action

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        unscaled_action, states = self.policy.predict(observation, state, episode_start, deterministic)
        scaled_action = self.policy.scale_action(unscaled_action)
        scaled_action = np.array([self.env.envs[i].constraint.enforceConstraintIfNeed(observation[i], scaled_action[i]) for i in range(len(self.env.envs))])
        return self.policy.unscale_action(scaled_action), states

if __name__ == "__main__":
    from ..half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint
    import cvxpy as cp
    cons = HalfCheetahDynamicConstraint()
    action = np.random.rand(6)
    state = 10*np.random.rand(17)
    grad = np.random.rand(6)
    gp.setParam('OutputFlag', 0)
    print(action, state[11:], grad)
    x_value = FW_update(action, state, grad, cons, 1)
    print(x_value)
    x = cp.Variable(6)
    obj = cp.Maximize(grad.T @ x)
    prob = cp.Problem(obj, cons.cvxpy_constraints(x, state))
    prob.solve(solver=cp.GUROBI)
    print(x.value)
