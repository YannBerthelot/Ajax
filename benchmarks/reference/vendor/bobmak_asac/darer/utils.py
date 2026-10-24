import os
import random
import yaml
from gymnasium.core import ActionWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import time

import torch
import sys
try:
    import wandb
except ImportError:
    print("Unable to import wandb, proceeding without.")

from gymnasium.wrappers import TimeLimit
sys.path.append("../tabular")
sys.path.append("tabular")
from gymnasium import spaces
# from tabular_utils import get_dynamics_and_rewards, solve_unconstrained
from wrappers import FrameStack
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers import RecordVideo

def safe_open(path):
    with open(path) as f:
        yaml_contents = yaml.load(f, yaml.FullLoader)
    return yaml_contents

def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        # another run may be creating a folder:
        random_wait_time = random.uniform(0, 3)
        time.sleep(random_wait_time)
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            # another run may be creating a folder:
            random_wait_time = random.uniform(0, 3)
            time.sleep(random_wait_time)
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"
            # try:
            #     os.makedirs(tmp_path, exist_ok=False)
            # except FileExistsError:
            #     # try again with an incremented number:
            # pass
        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        # , "csv", "tensorboard"])
        logger = configure(format_strings=["stdout"])

    return logger

class PermuteAtariObs(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = self.env
        new_shape = (self.observation_space.shape[-1], *self.observation_space.shape[:-1])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.transpose([2,0,1]),
            high=self.observation_space.high.transpose([2,0,1]),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )
        self.action_space = self.action_space

    def step(self, *args, **kwargs):
        res = self.env.step(*args, **kwargs)
        newres = (np.transpose(res[0], [2,1,0]), *res[1:])
        del res
        return newres

    def reset(self, *args, **kwargs):
        res, info = self.env.reset(*args, **kwargs)
        res = np.transpose(res, [2,1,0])
        return res, info

def env_id_to_envs(env_id, render, is_atari=False, permute_dims=False, max_steps=None, render_mode=None):
    # first split off any prefix for discrete action wrapper:
    if isinstance(env_id, str):
        if 'DiscA:' in env_id:
            env_id = env_id.split(":")[1]
            # Use the get_environment function to get the environment:
            env = get_environment(env_id, nbins=3)
            # Make a new copy for the eval env:
            import copy
            eval_env = copy.deepcopy(env)
            return env, eval_env
        
        elif 'DiscA15:' in env_id:
            env_id = env_id.split(":")[1]
            # Use the get_environment function to get the environment:
            env = get_environment(env_id, nbins=15)
            # Make a new copy for the eval env:
            import copy
            eval_env = copy.deepcopy(env)
            return env, eval_env
        elif 'Disc2000A:' in env_id:
            env_id = env_id.split(":")[1]
            # Use the get_environment function to get the environment:
            env = get_environment(env_id, nbins=3, max_episode_steps=10000)
            # Make a new copy for the eval env:
            import copy
            eval_env = copy.deepcopy(env)
            return env, eval_env
        
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    if is_atari:
        return atari_env_id_to_envs(env_id, render, n_envs=1, frameskip=4, framestack_k=4, permute_dims=permute_dims, render_mode=render_mode)
    else:
        env = gym.make(env_id)
        if max_steps is not None:
            eval_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
        else:
            eval_env = gym.make(env_id, render_mode='rgb_array')

        return env, eval_env


def atari_env_id_to_envs(env_id, render, n_envs, frameskip=1, framestack_k=None, grayscale_obs=True, permute_dims=False, render_mode=None):
    if isinstance(env_id, str):
        # Don't vectorize if there is only one env
        if n_envs==1:
            env = gym.make(env_id, frameskip=frameskip, render_mode='human' if render else None)
            env = AtariPreprocessing(env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                env = FrameStack(env, framestack_k)
            # permute dims for nature CNN in sb3
            if permute_dims:
                env = PermuteAtariObs(env)
            # env = AtariAdapter(env)
            # make another instance for evaluation purposes only:
            eval_env = gym.make(env_id, render_mode='human' if render else None, frameskip=frameskip)
            eval_env = AtariPreprocessing(eval_env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                eval_env = FrameStack(eval_env, framestack_k)
            if permute_dims:
                eval_env = PermuteAtariObs(eval_env)
            # eval_env = AtariAdapter(eval_env)
            if render:
                eval_env = RecordVideo(eval_env, video_folder='videos')
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
                eval_env = FireResetEnv(eval_env)
        else:
            env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

            eval_env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

    elif isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
    else:
        env = env_id

        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        # raise ValueError(
            # "env_id must be a string or gym.Env instance.")

    return env, eval_env

# Fire on reset env wrapper:
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, {}


def log_class_vars(self, logger, params, use_wandb=False):
    # logger = self.logger
    for key, value in params.items():
        value = self.__dict__[value]
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(key, value)
        if use_wandb:
            wandb.log({key: value})

def get_eigvec_values(fa, save_name=None, logu=False):
    env = fa.env
    nS = env.observation_space.n
    nA = fa.nA
    eigvec = np.zeros((nS, nA))
    for i in range(nS):
        if logu:
            eig_val = np.mean([logu.forward(i).cpu().detach().numpy() for logu in fa.model.nets], axis=0)
            eig_val = np.exp(eig_val)
        else:
            eig_val = np.mean([logu.forward(i).cpu().detach().numpy() for logu in fa.model.nets], axis=0)
        eigvec[i, :] = eig_val

    if save_name is not None:
        np.save(f'{save_name}.npy', eigvec)

    # normalize:
    eigvec /= np.linalg.norm(eigvec)
    if save_name is not None:
        np.save(f'{save_name}.npy', eigvec)
    return eigvec

def get_true_eigvec(fa, beta):
    dynamics, rewards = get_dynamics_and_rewards(fa.env.unwrapped)
    # uniform prior:
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    solution = solve_unconstrained(
        beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution
    
    # normalize:
    u_true /= np.linalg.norm(u_true)
    return u_true

def is_tabular(env):
    return isinstance(env.observation_space, gym.spaces.Discrete) and isinstance(env.action_space, gym.spaces.Discrete)


def sample_wandb_hyperparams(params, int_hparams=None):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] in {'uniform', 'q_uniform'} or v['distribution'] in {'q_uniform_values', 'uniform_values'}:
                val = random.uniform(v['min'], v['max'])
                if v['distribution'].startswith("q_"):
                    val = int(val)
                sampled[k] = val
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] in {'log_uniform_values', 'q_log_uniform_values'}:
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                if v['distribution'].startswith("q_"):
                    sample = int(sample)
                sampled[k] = sample
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError # f"Value {v} not recognized."
        if k in int_hparams:
            sampled[k] = int(sampled[k])

        assert k in sampled, f"Hparam {k} not successfully sampled."
    return sampled


def find_torch_modules(module, modules=None, prefix=None):
    """
    Recursively find all torch.nn.Modules within a given module.
    Args:
        module (nn.Module): The module to inspect.
        modules (dict, optional): A dictionary to collect module names and their instances.
        prefix (str, optional): A prefix for the module names to handle nested structures.
    Returns:
        dict: A dictionary with module names as keys and module instances as values.
    """
    if modules is None:
        modules = {}
    # Check if the current module itself is an instance of nn.Module
    submodules = None
    if isinstance(module, torch.nn.Module):
        modules[prefix] = module.state_dict()
        submodules = module.named_children
    elif hasattr(module, '__dict__'):
        submodules = module.__dict__.items
    # Recursively find all submodules if the current module is a container
    if submodules:
        for name, sub_module in submodules():
            if prefix:
                mod_name = f"{prefix}.{name}"
            else:
                mod_name = name
            find_torch_modules(sub_module, modules, mod_name)

    return modules


def get_max_grad(model):
    grad_norms = []
    with torch.no_grad():
        # Iterate over the parameters of the model
        for param in model.parameters():
            # Check if the parameter has a gradient (i.e., it's trainable)
            if param.grad is not None:
                # Calculate and store the gradient norm
                grad_norms.append(torch.norm(param.grad).item())

        # Check if any gradients were found
        if grad_norms:
            # Compute the maximum gradient norm
            max_grad_norm = max(grad_norms)
        else:
            # No gradients found, set max_grad_norm to 0
            max_grad_norm = 0.0
    return max_grad_norm


class DiscretizeAction(ActionWrapper):
    def __init__(self, env: gym.Env, nbins: int) -> None:
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box)
        assert len(env.action_space.shape) == 1

        self.ndim_actions, = env.action_space.shape
        self.powers = [nbins ** (i-1) for i in range(self.ndim_actions, 0, -1)]

        low = env.action_space.low
        high = env.action_space.high
        self.action_mapping = np.linspace(low, high, nbins)
        self.action_space = spaces.Discrete(nbins ** self.ndim_actions)
    
    def action(self, action):
        
        a = action
        unwrapped_action = np.zeros((self.ndim_actions,), dtype=float)

        for i, p in enumerate(self.powers):
            idx, a = a // p, a % p
            unwrapped_action[i] = self.action_mapping[idx, i]

        return unwrapped_action

def get_environment(env_name, nbins=3, max_episode_steps=0):

    if env_name == 'Pendulum-v1':
        env = gym.make('Pendulum-v1', max_episode_steps=max_episode_steps)
        env = DiscretizeAction(env, nbins=nbins)
    else:
        raise ValueError(f'wrong environment name {env_name}')

    return env