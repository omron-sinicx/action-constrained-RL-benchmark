# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import gym
import numpy as np
import os
import argparse
import torch
import json
import sys
import pybulletgym

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecMonitor
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure


from action_constrained_rl.env_wrapper import ConstraintEnvWrapper
from action_constrained_rl.env_wrapper import MemorizeCenterEnvWrapper
from action_constrained_rl.ddpg.projection_ddpg import ProjectionDDPG
from action_constrained_rl.td3.projection_td3 import ProjectionTD3
from action_constrained_rl.sac.projection_sac import ProjectionSAC
from action_constrained_rl.ddpg.noise_insertion_ddpg import NoiseInsertionDDPG
from action_constrained_rl.ddpg.logging_gradient import LoggingGradientDDPG
from action_constrained_rl.ddpg.logging_gradient import DDPGWithOutputPenalty
from action_constrained_rl.ddpg.nfwpo import NFWPO
from action_constrained_rl.ddpg.ddpg_with_penalty import DDPGWithPenalty
from action_constrained_rl.td3.td3_with_penalty import TD3WithPenalty
from action_constrained_rl.td3.td3_output_penalty import TD3WithOutputPenalty
from action_constrained_rl.td3.noise_insertion_td3 import NoiseInsertionTD3
from action_constrained_rl.sac.logging_gradient import LoggingGradientSAC
from action_constrained_rl.sac.logging_gradient import SACWithOutputPenalty
from action_constrained_rl.sac.safe_sampling_sac import SafeSamplingSAC
from action_constrained_rl.nn.opt_layer.opt_layer import OptLayer
from action_constrained_rl.nn.opt_layer.opt_layer_policy import OptLayerPolicy
from action_constrained_rl.nn.additional_layers.alpha_projection import AlphaProjectionLayer
from action_constrained_rl.nn.additional_layers.radial_squash import SquashLayer
from action_constrained_rl.nn.additional_layers.alpha_distribution import AlphaGaussianDistribution
from action_constrained_rl.nn.additional_layers.alpha_distribution import AlphaStateDependentNoiseDistribution
from action_constrained_rl.nn.additional_layers.shrinked_distribution import ShrinkedGaussianDistribution
from action_constrained_rl.nn.additional_layers.shrinked_distribution import ShrinkedStateDependentNoiseDistribution
from action_constrained_rl.nn.additional_layer_policy import AdditionalLayerPolicy
from action_constrained_rl.nn.additional_layer_sac_policy import AdditionalLayerSACPolicy
from action_constrained_rl.utils.constant_function import ConstantFunction
from action_constrained_rl.utils.arithmatic_series import ArithmaticSeries
from action_constrained_rl.utils.geometric_series import GeometricSeries
from action_constrained_rl.utils.log_series import LogSeries
from action_constrained_rl.constraint.box_constraint import BoxConstraint
from action_constrained_rl.constraint.power_constraint import PowerConstraint
from action_constrained_rl.constraint.power_constraint import OrthoplexConstraint
from action_constrained_rl.constraint.power_constraint import DecelerationConstraint
from action_constrained_rl.constraint.sphere_constraint import SphericalConstraint
from action_constrained_rl.constraint.tip_constraint import TipConstraint
from action_constrained_rl.constraint.MA_constraint import MAConstraint
from action_constrained_rl.constraint.combined_constraint import CombinedConstraint
from action_constrained_rl.constraint.sin2_constraint import Sin2Constraint

import gurobipy as gp
gp.setParam('OutputFlag', 0)



def nameToConstraint(args):
    name = args.env
    c_name = args.constraint

    if c_name == "Box" or c_name == "Power" or c_name == "Orthoplex" or c_name == "Deceleration" or c_name == "Sphere":
        if name == "Hopper-v3":
            offset = 8
            scale = (1, 1, 1)
            indices = list(range(offset, offset+len(scale)))
            s_dim = 11
        elif name == "ReacherPyBulletEnv-v0":
            offset = 6
            scale = (1, 1)
            indices = [6, 8]
            s_dim = 9
        elif name == 'Ant-v3':
            offset = 19
            scale = (1.,1.,1.,1.,1.,1.,1.,1.)
            indices = list(range(offset, offset+len(scale)))
            s_dim = 27
        elif name == 'HalfCheetah-v3':
            offset = 11
            scale = (1., 1., 1., 1., 1., 1.)
            indices = list(range(offset, offset+len(scale)))
            s_dim = 17
        elif name == 'Swimmer-v3':
            offset = 6
            scale = (1.,1.)
            indices = list(range(offset, offset+len(scale)))
            s_dim = 8
        elif name == 'Walker2d-v3':
            offset = 11
            scale = (1., 1., 1., 1., 1., 1.)
            indices = list(range(offset, offset+len(scale)))
            s_dim = 17

        if c_name == "Box":
            return BoxConstraint(len(scale)) # R+N
        elif c_name == "Power":
            return PowerConstraint(indices, scale, args.max_power, s_dim) # R+M, H+M, W+M
        elif c_name == "Orthoplex":
            return OrthoplexConstraint(indices, scale, args.max_power, s_dim) # R+O03, R+O10, R+O30
        elif c_name == "Deceleration":
            return DecelerationConstraint(indices, scale, args.max_power, s_dim) #unused
        elif c_name == "Sphere":
            return SphericalConstraint(len(scale), args.max_power) #R+L2
        
    elif c_name == 'Tip':
        return TipConstraint(args.max_power)# R+T
    elif c_name == 'MA':
        return MAConstraint(args.max_power) # HC+MA
    elif c_name == 'O+S':
        if name == "Hopper-v3":
            offset_p = 2
            scale = (1, 1, 1)
            indices_p = list(range(offset_p, offset_p+len(scale)))
            offset_v = 8
            indices_v = list(range(offset_v, offset_v+len(scale)))
            s_dim = 11
        elif name == 'Walker2d-v3':
            offset_p = 2
            scale = (1., 1., 1., 1., 1., 1.)
            indices_p = list(range(offset_p, offset_p+len(scale)))
            offset_v = 11
            indices_v = list(range(offset_v, offset_v+len(scale)))
            s_dim = 17
        return CombinedConstraint(OrthoplexConstraint(indices_v, scale, args.max_power[0], s_dim),
                                  Sin2Constraint(indices_p, args.max_power[1], s_dim)) # H+O+S, W+O+S
    else:
        raise

def nameToEnv(name, seed=0):
    env = gym.make(name)
    env.seed(seed)
    return env

# unused
def pickConstraintCoefficient(args):
    if args.d > 0.0:
        constraint_penalty = ArithmaticSeries(args.a0, args.d)
    elif args.r > 0.0:
        constraint_penalty = GeometricSeries(args.a0, args.r)
    elif args.use_log_series:
        constraint_penalty = LogSeries(args.a0)
    else:
        constraint_penalty = ConstantFunction(args.c)
    return constraint_penalty

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", action="store", default="tmp")
parser.add_argument("--prob_id", action="store", default = "");
parser.add_argument("--algo_id", action="store", default = "");
parser.add_argument("--env", action="store", default="HalfCheetahDynamic")
parser.add_argument("--constraint", action="store", default="Normal")
parser.add_argument("--max_power", action="store", default=1., type=float)
parser.add_argument("--solver", action="store", default="TD3", choices=["DDPG", "SAC", "TD3"])
parser.add_argument("--num_time_steps", action="store", default=1e6, type=int)
parser.add_argument("--batch_size", action="store", default=100, type=int)
parser.add_argument("--use_env_wrapper", action="store_true", default=False, help="use projection inside environments")
parser.add_argument("--use_action_restriction", action="store_true", default=False)
parser.add_argument("--sigma", action="store", default=0.01, type=float, help="stddev for Gaussian Action Noise")
parser.add_argument("--a0", action="store", default=0.001, type=float)
parser.add_argument("--dual_learning_rate", action="store", default=0.0, type=float)
parser.add_argument("--learning_rate", action="store", default=1e-3, type=float)
parser.add_argument("--init_ent_coef", action="store", default=2.0, type=float)
parser.add_argument("--n", action="store", default=1, type=int, help="Update the penalty coefficient c every n episodes")
parser.add_argument("--verbose", action="store", default=1, type=int)
parser.add_argument("--seed", action="store", default=0, type=int)
parser.add_argument("--eval_freq", action="store", default=5000, type=int, help="run evaluation episodes every eval_freq time steps")
parser.add_argument("--n_eval_episodes", action="store", default=5, type=int)
parser.add_argument("--normalize_constraint", action="store_true", default=False)
parser.add_argument("--device", action="store", default='auto')
parser.add_argument("--squash_output", action="store_true", default=False)
parser.add_argument("--use_my_mlppolicy", action="store_true", default=False)
parser.add_argument("--infinity_action_space", action="store_true", default=False)
parser.add_argument("--use_NFWPO", action="store_true", default=False)
parser.add_argument("--fw_learning_rate", action="store", default=0.01, type=float)
parser.add_argument("--logging_gradient", action="store", default=True, type=bool)
parser.add_argument("--output_stdout", action="store_true", default=False)

group = parser.add_mutually_exclusive_group()
group.add_argument("--c", action="store", default=0.0, type=float, help="Constant penalty coefficient")
group.add_argument("--d", action="store", default=-1.0, type=float, help="add d to the penalty coefficient")
group.add_argument("--r", action="store", default=-1.0, type=float)
group.add_argument("--use_log_series", action="store_true", default=False)

group = parser.add_mutually_exclusive_group()
group.add_argument("--use_static_constraint_net", action="store_true", default=False)
group.add_argument("--use_opt_layer", action="store_true", default=False)
group.add_argument("--use_alpha_projection_layer", action="store_true", default=False)
group.add_argument("--use_squash_layer", action="store_true", default=False)

parser.add_argument("--proj_type", action="store", default="QP", choices=["QP", "alpha", "squash"])
args = parser.parse_args()


# from problem id, set problem arguments
if args.prob_id != "":
    if args.prob_id == "R-N":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Box"
    elif args.prob_id == "R-L2":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Sphere"
        args.max_power = 0.05
    elif args.prob_id == "R-O03":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Orthoplex"
        args.max_power = 0.3
    elif args.prob_id == "R-O10":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Orthoplex"
        args.max_power = 1.0
    elif args.prob_id == "R-O30":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Orthoplex"
        args.max_power = 3.0
    elif args.prob_id == "R-M":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Power"
        args.max_power = 1.0 
    elif args.prob_id == "R-T":
        args.env = "ReacherPyBulletEnv-v0"
        args.constraint = "Tip"
        args.max_power = 0.05 
    elif args.prob_id == "HC+O" or args.prob_id == "HC+O-16":
        args.env = "HalfCheetah-v3"
        args.constraint = "Orthoplex"
        args.max_power = 20.
    elif args.prob_id == "H+M" or args.prob_id == "H+M-16":
        args.env = "Hopper-v3"
        args.constraint = "Power"
        args.max_power = 10.
    elif args.prob_id == "W+M" or args.prob_id == "W+M-16":
        args.env = "Walker2d-v3"
        args.constraint = "Power"
        args.max_power = 10.
    elif args.prob_id == "HC+MA":
        args.env = "HalfCheetah-v3"
        args.constraint = "MA"
        args.max_power = 5.
    elif args.prob_id == "H+O+S":
        args.env = "Hopper-v3"
        args.constraint = "O+S"
        args.max_power = (10., 0.1)
    elif args.prob_id == "W+O+S":
        args.env = "Walker2d-v3"
        args.constraint = "O+S"
        args.max_power = (10., 0.1)
    else: raise ValueError("unknown problem id")

# from algorithm id, set algorithm arguments
if args.algo_id != "":
    if args.algo_id == "DPro":
        args.use_action_restriction = True
    elif args.algo_id == "DPro+":
        args.use_action_restriction = True
        args.c = 1.
    elif args.algo_id == "DPre":
        args.use_env_wrapper = True
    elif args.algo_id == "DPre+":
        args.use_env_wrapper = True
        args.c = 1.
    elif args.algo_id == "DOpt":
        args.use_opt_layer = True
        args.squash_output = True
    elif args.algo_id == "DOpt+":
        args.use_opt_layer = True
        args.squash_output = True
        args.c = 1.
    elif args.algo_id == "NFW":
        args.use_NFWPO = True
    elif args.algo_id == "DAlpha":
        args.use_alpha_projection_layer = True
    elif args.algo_id == "DRad":
        args.use_squash_layer = True
    elif args.algo_id == "SPre":
        args.use_env_wrapper = True
        args.solver = "SAC"
    elif args.algo_id == "SPre+":
        args.use_env_wrapper = True
        args.solver = "SAC"
        args.c = 1.
    elif args.algo_id == "SAlpha":
        args.use_alpha_projection_layer = True
        args.solver = "SAC"
    elif args.algo_id == "SRad":
        args.use_squash_layer = True
        args.solver = "SAC"
    else:
        raise ValueError("unknown algo id")

if args.proj_type == "squash":
    assert args.infinity_action_space
if args.use_squash_layer:
    assert not args.squash_output
if args.use_opt_layer:
    assert args.squash_output

log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
if not args.output_stdout:
    sys.stdout = open(log_dir+"/log.txt", "w")
    sys.stderr = open(log_dir+"/error_log.txt", "w")
print(args)
with open(f'{log_dir}/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

env = nameToEnv(args.env, args.seed)
constraint = nameToConstraint(args)
constraint.proj_type = args.proj_type
constraint_penalty = pickConstraintCoefficient(args) # penalty coefficient function for output penalty

if args.use_alpha_projection_layer or args.use_squash_layer: # wrapper to memorize the centers
    EnvWrapper = MemorizeCenterEnvWrapper
    env = EnvWrapper(constraint, env, n=args.n, dual_learning_rate=args.dual_learning_rate)
    env = VecMonitor(DummyVecEnv([lambda: env]), filename=log_dir + "/monitor.csv")
    eval_env = EnvWrapper(constraint, nameToEnv(args.env, args.seed), n=args.n, dual_learning_rate=args.dual_learning_rate)
    eval_env = VecMonitor(DummyVecEnv([lambda: eval_env]), filename=None)
else:  # wrapper to project actions. We do not use reward penalty
    env = ConstraintEnvWrapper(constraint, env, constraint_penalty=ConstantFunction(0), enforce_constraint=args.use_env_wrapper or args.use_action_restriction, filename=log_dir + "/monitor.csv", n=args.n, dual_learning_rate=args.dual_learning_rate, normalize=args.normalize_constraint, infinity_action_space = args.infinity_action_space)
    env = VecMonitor(DummyVecEnv([lambda: env]), filename=log_dir + "/vec_monitor.csv")
    eval_env = ConstraintEnvWrapper(constraint, nameToEnv(args.env, args.seed), constraint_penalty=ConstantFunction(0), enforce_constraint=args.use_env_wrapper or args.use_action_restriction, filename=None, n=args.n, infinity_action_space = args.infinity_action_space)
    eval_env = VecMonitor(DummyVecEnv([lambda: eval_env]), filename=None)
    
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=args.eval_freq, n_eval_episodes = args.n_eval_episodes,
                         deterministic=True, render=False)

# set rl-zoo hyperparameters
n_actions = env.action_space.shape[-1]
if args.env == "ReacherPyBulletEnv-v0":
    if args.solver == "TD3" or args.use_NFWPO:
        n_timesteps = 3e5
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))
        kargs = {"gamma": 0.98, "buffer_size": 200000, "learning_starts": 10000,
                 "action_noise": action_noise, "gradient_steps": -1, "train_freq": (1, "episode"),
                 "learning_rate": 1e-3, "policy_kwargs": {"net_arch":[400, 300]}}
    elif args.solver == "SAC":
        n_timesteps = 3e5
        kargs = {"learning_rate": 7.3e-4, "buffer_size": 300000, "batch_size": 256,
                 "ent_coef": 'auto', "gamma": 0.98, "tau": 0.02, "train_freq": 8,
                 "gradient_steps": 8, "learning_starts": 10000,
                 "use_sde": True, "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300])}
elif args.env == "HalfCheetah-v3":
    if args.solver == "TD3" or args.use_NFWPO:
        n_timesteps = 1e6
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))
        kargs = {"learning_starts": 10000, "action_noise": action_noise}
    elif args.solver == "SAC":
        n_timesteps = 1e6
        kargs = {"learning_starts": 10000}
elif args.env == "Hopper-v3":
    if args.solver == "TD3" or args.use_NFWPO:
        n_timesteps = 1e6
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))
        kargs = {"learning_starts": 10000, "action_noise": action_noise, "train_freq": 1,
                 "gradient_steps": 1, "learning_rate": 3e-4, "batch_size": 256}
    elif args.solver == "SAC":
        n_timesteps = 1e6
        kargs = {"learning_starts": 10000}
elif args.env == "Walker2d-v3":
    if args.solver == "TD3" or args.use_NFWPO:
        n_timesteps = 1e6
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))
        kargs = {"learning_starts": 10000, "action_noise": action_noise}
    elif args.solver == "SAC":
        n_timesteps = 1e6
        kargs = {"learning_starts": 10000}
else:
    raise
kargs["verbose"]=args.verbose
if not "policy_kwargs" in kargs:
    kargs["policy_kwargs"]={}
if args.prob_id[-3:] == "-16":
    print("batch_size: 16")
    kargs.update({"batch_size": 16})

def pickModel(constraint):
    # select model according to arguments
    seed = args.seed
    
    if args.use_NFWPO: #NFW
        if args.prob_id[:2] == "R-":
            fw_learning_rate = 0.05
        else:
            fw_learning_rate = 0.01
        model = NFWPO(constraint, "MlpPolicy", env, fw_learning_rate = fw_learning_rate,
                      device = args.device, seed = seed, **kargs)

    elif args.use_action_restriction: #DPro, DPro+
        if args.solver == "DDPG":
            algo = ProjectionDDPG
        elif args.solver == "TD3":
            algo = ProjectionTD3
        elif args.solver == "SAC":
            algo = ProjectionSAC
        model = algo(constraint, "MlpPolicy", env, constraint_penalty = constraint_penalty, device = args.device, seed = seed, **kargs)

    elif args.use_alpha_projection_layer or args.use_squash_layer: # DAlpha, DRad, SAlpha, SRad
        kargs["policy_kwargs"].update({"constraint": constraint})
        if args.solver == "DDPG" or args.solver == "TD3":
            if args.solver == "DDPG":
                algo = NoiseInsertionDDPG
            else:
                algo = NoiseInsertionTD3
            policy = AdditionalLayerPolicy
            if args.use_alpha_projection_layer:
                layer_type = AlphaProjectionLayer
            elif args.use_squash_layer:
                layer_type = SquashLayer
            kargs["policy_kwargs"].update({"layer_type": layer_type, "squash_output": args.squash_output})
        else:
            algo = SafeSamplingSAC
            action_noise = None
            policy = AdditionalLayerSACPolicy
            if args.use_alpha_projection_layer:
                if "use_sde" in kargs and kargs["use_sde"]:
                    distribution_class = AlphaStateDependentNoiseDistribution
                else:
                    distribution_class = AlphaGaussianDistribution
            if args.use_squash_layer:
                if "use_sde" in kargs and kargs["use_sde"]:
                    distribution_class = ShrinkedStateDependentNoiseDistribution
                else:
                    distribution_class = ShrinkedGaussianDistribution
            kargs["policy_kwargs"].update({"distribution_class": distribution_class})
        model = algo(policy, env, device = args.device, seed = seed, **kargs)
    elif args.use_opt_layer: # DOpt, DOpt+
        if args.solver == "DDPG" or args.solver == "TD3":
            if args.solver == "DDPG":
                #algo = NoiseInsertionDDPG
                algo = DDPGWithPenalty
            else:
                algo = TD3WithPenalty
        else:
            algo = SafeSamplingSAC
            action_noise = None
        kargs["policy_kwargs"].update({"constraint": constraint, "squash_output": args.squash_output})
        model = algo(constraint, OptLayerPolicy, env, use_center_wrapper = False, constraint_penalty = constraint_penalty, device = args.device, seed = seed, **kargs)
    else:
        if args.solver == "DDPG":
            algo = DDPGWithOutputPenalty
        elif args.solver == "TD3":
            algo = TD3WithOutputPenalty # DPre, DPre+
        elif args.solver == "SAC":
            algo = SACWithOutputPenalty # SPre, SPre+
        model = algo(constraint, "MlpPolicy", env, constraint_penalty = constraint_penalty,  device = args.device, seed = seed, **kargs)
    return model

model = pickModel(constraint)
if args.logging_gradient:
    logger = configure(args.log_dir)
    model.set_logger(logger)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)

del model # remove to demonstrate saving and loading
