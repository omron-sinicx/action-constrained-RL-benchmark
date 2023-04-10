# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import gym
import numpy as np
from gym import spaces
import math
import time
from stable_baselines3.common.monitor import ResultsWriter
from ..utils.constant_function import ConstantFunction

def quadraticPenalty(x):
    return np.sum(np.square(x))

class ConstraintEnvWrapper(gym.Wrapper):
    """
    wrapper to project actions
    """
    def __init__(self, constraint, env, constraint_penalty=ConstantFunction(0.0), enforce_constraint=False, filename=None, n=1, dual_learning_rate=0.0, normalize=False, infinity_action_space = False):
        super().__init__(env)
        self.env = env
        self.constraint = constraint
        self.prev_obs = None
        self.constraint_penalty = constraint_penalty
        self.num_pre_projection_constraint_violation = 0
        self.enforce_constraint = enforce_constraint
        self.dual_learning_rate = dual_learning_rate
        self.n = n
        self.normalize = normalize

        self.t_start = time.time()
        if filename is not None:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env.spec and env.spec.id},
                extra_keys=("v","c","sum_term","quadratic_sum_term", "quadratic_sum_term_normalized")
            )
        else:
            self.results_writer = None
        self.rewards = None
        self.violations = None
        self.quadratic_sum_violations = None
        self.quadratic_sum_violations_normalized = None
        self.needs_reset = True
        self.total_steps = 0
        self.lagrange_multiplier = np.zeros(self.constraint.numConstraints())
        self.episode =1
        self.infinity_action_space = infinity_action_space
        
    def logConstraintViolation(self, state, action):
        self.violations.append(self.constraint.constraintViolation(state, action))
        self.quadratic_sum_violations.append(quadraticPenalty(self.constraint.constraintViolation(state, action)))
        self.quadratic_sum_violations_normalized.append(quadraticPenalty(self.constraint.constraintViolation(state, action, normalize=True)))

    def step(self, action):
        state = self.prev_obs.squeeze()
        
        penalty = 0.0
        lagrange_term = 0.0

        if self.infinity_action_space:
            action = np.arctanh(np.clip(action,-1+1e-6,1-1e-6))
        self.logConstraintViolation(state, action)
        if self.enforce_constraint and self.constraint.proj_type == "shrinkage":
            action = self.constraint.enforceConstraint(state, action)
        elif not self.constraint.isConstraintSatisfied(state, action):
            self.num_pre_projection_constraint_violation += 1

            if self.enforce_constraint:
                g = self.constraint.constraintViolation(state, action, normalize=self.normalize)
                penalty -= self.constraint_penalty(self.episode // self.n + 1) * quadraticPenalty(g)
                action = self.constraint.enforceConstraint(state, action)
                
                #lagrange_term += np.dot(self.lagrange_multiplier, g)
                

        assert self.constraint.isConstraintSatisfied(state, action), f"constraint violated state={state} action{action} "

        next_state, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            if self.episode % self.n == 0:
                pass
                #self.lagrange_multiplier = np.minimum(0.0, self.lagrange_multiplier -self.dual_learning_rate * sum(self.violations))
                #print("violations: {}".format(sum(self.constraint_violations)))
                #print("multiplier: {}".format(self.lagrange_multiplier))
            self.needs_reset = True
            ep_info = self.getEpInfo()
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
            self.episode += 1
            
        self.total_steps += 1

        self.prev_obs = next_state
        return next_state, reward + lagrange_term + penalty, done, info
    
    def getEpInfo(self):
        ep_rew = sum(self.rewards)
        ep_len = len(self.rewards)
        ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), "v": self.num_pre_projection_constraint_violation, "c": round(self.constraint_penalty(self.episode // self.n + 1), 6), "sum_term": round(sum(sum(self.violations)), 6), "quadratic_sum_term": round(sum(self.quadratic_sum_violations), 6), "quadratic_sum_term_normalized": round(sum(self.quadratic_sum_violations_normalized), 6)}
        return ep_info

    def reset(self, **kwargs):
        self.rewards = []
        self.violations = []
        self.quadratic_sum_violations = []
        self.quadratic_sum_violations_normalized = []
        self.num_pre_projection_constraint_violation = 0
        self.needs_reset = False
        self.prev_obs = self.env.reset()
        return self.prev_obs


if __name__ == "__main__":
    import math
    import time
    import gym
    from ..idealized_pendulum.cbf import PendulumCBF
    
    normalization_factor = 500.0 
    env = ConstraintEnvWrapper(PendulumCBF(normalization_factor=normalization_factor), gym.make("IdealizedPendulum-v0",  normalization_factor=normalization_factor, dt=0.01, max_speed=1000.0))
    observation = env.reset()
    observation = env.setState(np.array([0.25 * np.pi, 1.0]))
    frames = []

    for _ in range(1000):
        action = np.random.uniform(-1.0, 1.0)
        observation, reward, done, info = env.step(np.array([action]))
        print(observation)
        env.render()
        time.sleep(0.03)
#        frames.append(env.render("rgb_array"))

    env.close()
