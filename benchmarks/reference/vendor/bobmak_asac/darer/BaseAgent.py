import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from typing import Optional, Union, List, Tuple, Dict, Any, get_type_hints
from typeguard import typechecked
try:
    import wandb
except:
    pass
from utils import env_id_to_envs, get_true_eigvec, is_tabular, log_class_vars, get_eigvec_values, find_torch_modules
from gymnasium.wrappers import RecordVideo
try:
    from gymnasium.experimental.wrappers import RecordVideoV0 as RecordVideo
except:
    print("RecordVideoV0 not available")
HPARAM_ATTRS = {
    'beta': 'beta',
    'learning_rate': 'learning_rate',
    'batch_size': 'batch_size',
    'buffer_size': 'buffer_size',
    'target_update_interval': 'target_update_interval',
    'tau': 'tau',
    'hidden_dim': 'hidden_dim',
    'num_nets': 'num_nets',
    'tau_theta': 'tau_theta',
    'gradient_steps': 'gradient_steps',
    'train_freq': 'train_freq',
    'max_grad_norm': 'max_grad_norm',
    'learning_starts': 'learning_starts',
}

LOG_PARAMS = {
    'time/env. steps': 'env_steps',
    'eval/avg_reward': 'avg_eval_rwd',
    'eval/auc': 'eval_auc',
    'time/num. episodes': 'num_episodes',
    'time/fps': 'fps',
    'time/num. updates': '_n_updates',
    'rollout/beta': 'beta',
    'train/lr': 'lr',
}

int_args = ['batch_size',
            'buffer_size',
            'target_update_interval',
            'hidden_dim',
            'num_nets',
            'gradient_steps',
            'train_freq',
            'max_grad_norm',
            'learning_starts']


str_to_aggregator = {'min': lambda x, dim: torch.min(x, dim=dim, keepdim=True)[0],
                     'max': lambda x, dim: torch.max(x, dim=dim, keepdim=True)[0],
                     'mean': lambda x, dim: (torch.mean(x, dim=dim, keepdim=True))}

# use get_type_hints to throw errors if the user passes in an invalid type:


class BaseAgent:
    @typechecked
    def __init__(self,
                 env_id: Union[str, gym.Env],
                 learning_rate: float = 1e-3,
                 beta: Union[float, str] = 0.1,
                 beta_schedule: str = 'none',
                 batch_size: int = 64,
                 buffer_size: int = 100_000,
                 target_update_interval: int = 10_000,
                 tau: float = 1.0,
                 prior_update_interval: int = 1_000,
                 prior_tau: float = 0.9,
                 hidden_dim: int = 64,
                 num_nets: int = 2,
                 tau_theta: float = 0.005,
                 gradient_steps: int = 1,
                 train_freq: Union[int, Tuple[int, str]] = 1,
                 max_grad_norm = None,
                 learning_starts = 5_000,
                 aggregator: str = 'min',
                 # F.HuberLoss(),
                 loss_fn: torch.nn.modules.loss = F.mse_loss,
                 device: Union[torch.device, str] = "auto",
                 render: bool = False,
                 tensorboard_log: Optional[str] = None,
                 log_interval: int = 1_000,
                 save_checkpoints: bool = False,
                 use_wandb: bool = False,
                 scheduler_str: str = 'none',
                 beta_end: Optional[float] = None,
                 max_eval_steps=None,
                 name_suffix: Optional[str] = '',
                 seed: Optional[int] = None,
                 save_best: Optional[bool] = False,
                 save_path: Optional[str] = 'best_models',
                 render_mode: Optional[str] = None,
                 ) -> None:
        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('TypeCheckMemo', None)
        self.kwargs.pop('memo', None)
        self.kwargs.pop('check_argument_types', None)
        # first check if Atari env or not:
        if isinstance(env_id, str):
            is_atari = 'NoFrameskip' in env_id or 'ALE' in env_id
        else:
            is_atari = False
        self.env, self.eval_env = env_id_to_envs(
            env_id, render, is_atari=is_atari, max_steps=max_eval_steps,
            render_mode=render_mode)

        if hasattr(self.env.unwrapped.spec, 'id'):
            self.env_str = self.env.unwrapped.spec.id
        elif hasattr(self.env.unwrapped, 'id'):
            self.env_str = self.env.unwrapped.id
        else:
            self.env_str = str(self.env.unwrapped)

        self.is_tabular = is_tabular(self.env)
        if self.is_tabular:
            # calculate the eigenvector exactly:
            self.true_eigvec = get_true_eigvec(self, beta).A.flatten()

        self.learning_rate = learning_rate
        self.beta = float(beta) if isinstance(beta, int) else beta
        self.beta_schedule = beta_schedule
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.prior_update_interval = prior_update_interval
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.train_freq = train_freq
        if isinstance(train_freq, tuple):
            raise NotImplementedError("train_freq as a tuple is not supported yet.\
                                       \nEnter int corresponding to env_steps")
        self.max_grad_norm = max_grad_norm if max_grad_norm != 0 else None
        self.num_nets = num_nets
        self.prior = None
        self.learning_starts = learning_starts
        self.use_wandb = use_wandb
        self.aggregator = aggregator
        self.tensorboard_log = tensorboard_log
        self.aggregator_fn = str_to_aggregator[aggregator]
        self.avg_eval_rwd = -np.inf
        self.fps = None
        self.beta_end = beta_end
        self.scheduler_str = scheduler_str
        self.train_this_step = False
        self.name_suffix = name_suffix
        # Track the rewards over time:
        self.step_to_avg_eval_rwd = {}
        # save the parameters for model loading
        self.save_best = save_best
        self.save_path = save_path

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=True,
                                          device=device)
        # assert isinstance(self.env.action_space, gym.spaces.Discrete), \
        #     "Only discrete action spaces are supported."
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.nA = self.env.action_space.n

        self.theta = torch.Tensor([0.0]).to(self.device)
        self.min_reward = torch.inf
        self.max_reward = -torch.inf
        self.eval_auc = 0
        self.num_episodes = 0

        self._n_updates = 0
        self.env_steps = 0
        # self._initialize_networks()
        self.loss_fn = loss_fn

    def log_hparams(self, logger):
        # Log the hparams:
        log_class_vars(self, logger, HPARAM_ATTRS)
        logger.dump()

    def _initialize_networks(self):
        raise NotImplementedError

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        """
        Sample an action from the policy of choice
        """
        raise NotImplementedError

    def gradient_descent(self, batch, grad_step: int):
        """
        Do a gradient descent step
        """
        raise NotImplementedError

    def _train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        
        # Increase update counter
        self._n_updates += gradient_steps
        # average self.theta over multiple gradient steps
        for grad_step in range(gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.replay_buffer.sample(batch_size)
            self.gradient_descent(batch, grad_step)
            # Update target net:
            if grad_step % self.target_update_interval == 0:
                self._update_target()


        # Can't use env_steps b/c we are inside the learn function which is called only
        # # every train_freq steps:
        # self.theta = self.tau_theta * self.theta + \
        #     (1 - self.tau_theta) * new_theta
        
        # # Log info from this training cycle:
        # self.logger.record("train/avg logu", curr_logu.mean().item())
        # self.logger.record("train/min logu", curr_logu.min().item())
        # self.logger.record("train/max logu", curr_logu.max().item())

        # Log the max gradient:
        # total_norm = torch.max(torch.stack(
        #             [px.grad.detach().abs().max()
        #                 for p in self.online_logus.parameters() for px in p]
        #             ))
        # self.logger.record("train/max_grad", total_norm.item())

    def learn(self, total_timesteps: int, early_stop: dict = {}) -> bool:
        """
        Train the agent for total_timesteps
        """
        stop_steps = early_stop.get('steps', 0)
        if stop_steps > 0:
            assert stop_steps % self.log_interval == 0, \
                "early_stop['steps'] must be a multiple of log_interval, or will never be checked"
        stop_reward = early_stop.get('reward', -np.inf)
        self.betas = self._beta_scheduler(self.beta_schedule, total_timesteps)

        # Start a timer to log fps:
        self.initial_time = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action_freqs = torch.zeros(self.nA)

            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            avg_ep_len = 0
            entropy = 0
            while not done and self.env_steps < total_timesteps:
                # take a random action:
                # if self.env_steps < self.learning_starts:
                #     action = self.env.action_space.sample()
                # else:
                (action, buffer_action), kl = self.exploration_policy(state)
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs[action] += 1
                # Add KL divergence bw the current policy and the prior:
                entropy += float(kl)
                # action = self.online_logus.greedy_action(state)
                # action = self.env.action_space.sample()
                action = action.reshape(self.env.action_space.shape)
                next_state, reward, terminated, truncated, infos = self.env.step(action)
                # update min and max reward:
                self.min_reward = min(self.min_reward, reward)
                self.max_reward = max(self.max_reward, reward)
                avg_ep_len += 1
                self._on_step()
                done = terminated or truncated
                self.rollout_reward += reward

                self.train_this_step = (self.train_freq == -1 and terminated) or \
                    (self.train_freq != -1 and self.env_steps %
                     self.train_freq == 0)

                # Add the transition to the replay buffer:
                sarsa = (state, next_state, buffer_action, reward, terminated)

                self.replay_buffer.add(*sarsa, [infos])
                state = next_state

            if terminated:
                # self.rollout_reward += 0
                avg_ep_len += 1
            if done:
                self.rollout_reward
                self.logger.record("rollout/ep_reward", self.rollout_reward)
                free_energy = (self.rollout_reward + 1/self.beta * entropy)
                try:
                    free_energy = free_energy.item()
                except:
                    pass
                # entropy = 0
                self.logger.record("rollout/neg_free_energy",
                                   free_energy / avg_ep_len)
                self.logger.record("rollout/avg_entropy", entropy / avg_ep_len)
                self.logger.record("rollout/avg_episode_length", avg_ep_len)
                self.logger.record("rollout/avg_reward_rate",
                                   self.rollout_reward / avg_ep_len)
                if self.use_wandb:
                    wandb.log({f'rollout/{self.env_str}/reward': self.rollout_reward})
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs /= action_freqs.sum()
                    for i, freq in enumerate(action_freqs):
                        # As percentage:
                        self.logger.record(
                            f'rollout/action {i} (%)', freq.item() * 100)

        return False

    def _on_step(self):
        """
        This method is called after every step in the environment
        """
        self.beta = self.betas[self.env_steps]
        self.logger.record("rollout/beta", self.beta)
        self.env_steps += 1
        if self.env_steps % self.log_interval == 0:
            self._log_stats()

        if self.train_this_step:
            if self.env_steps > self.learning_starts:
                self._train(self.gradient_steps, self.batch_size)

        # if self.env_steps % self.target_update_interval == 0:
        #     self._update_target()

        if self.env_steps % self.prior_update_interval == 0:
            self._update_prior()

        if self.env_steps % self.log_interval == 0:
            # Log info from this training cycle:
            self.logger.record("train/theta", self.theta.item())

    def _log_stats(self):
        # end timer:
        t_final = time.thread_time_ns()
        # fps averaged over log_interval steps:
        self.fps = self.log_interval / \
            ((t_final - self.initial_time + 1e-16) / 1e9)

        if self.env_steps > 0:
            self.avg_eval_rwd = self.evaluate()
            self.eval_auc += self.avg_eval_rwd
        if self.save_checkpoints:
            raise NotImplementedError
            torch.save(self.online_logu.state_dict(),
                       'sql-policy.para')
        # Get the current learning rate from the optimizer:
        self.lr = 0#self.optimzers.get_lr()
        log_class_vars(self, self.logger, LOG_PARAMS, use_wandb=self.use_wandb)

        if self.is_tabular:
            # Record the error in the eigenvector:
            if self.algo_name == 'LogU':
                log = True
            elif self.algo_name == 'U':
                log = False
            else:
                raise ValueError(
                    f"Unknown agent name: {self.name}. Use U/LogU (defaults).")
            fa_eigvec = get_eigvec_values(self, logu=log).flatten()
            err = np.abs(self.true_eigvec - fa_eigvec).max()
            self.logger.record('train/eigvec_err', err.item())

        if self.use_wandb:
            wandb.log({'env_steps': self.env_steps,
                       f'eval/{self.env_str}/avg_reward': self.avg_eval_rwd})
        self.logger.dump(step=self.env_steps)
        self.initial_time = time.thread_time_ns()

    def evaluate(self, n_episodes=10, render=False, verbose=False) -> float:
        # run the current policy and return the average reward
        if render:
            self.eval_env = RecordVideo(self.eval_env, video_folder=f'videos/{self.env_str}/{self.env_steps}')#, video_length=1000)

        self.initial_time = time.process_time_ns()
        avg_reward = 0.
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # log the action frequencies:
            action_freqs = torch.zeros(self.nA)
        n_steps = 0
        # if render:
        #     self.eval_env.render_mode = 'human'
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.evaluation_policy(state)
                # action = self.online_logus.choose_action(state)
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs[action] += 1
                # action = action.item()
                # action = self.online_logus.choose_action(state)
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated
                if render:
                    self.eval_env.render()
                if verbose:
                    print(info)
        # self.eval_env.close()

        avg_reward /= n_episodes
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # log the action frequencies:
            action_freqs /= n_episodes
            action_freqs /= action_freqs.sum()
            for i, freq in enumerate(action_freqs):
                # As percentage:
                self.logger.record(f'eval/action {i} (%)', freq.item() * 100)
        self.logger.record('eval/avg_episode_length', n_steps / n_episodes)
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time + 1e-12) / 1e9
        eval_fps = n_steps / eval_time
        self.logger.record('eval/time', eval_time)
        self.logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        # find the current best saved model
        if os.path.exists(os.path.join(self.save_path, f"{str(self)}_score.txt")):
            with open(os.path.join(self.save_path, f"{str(self)}_score.txt"), 'r') as f:
                best_saved_avg_rwd = float(f.read())
        else:
            best_saved_avg_rwd = -np.inf
            # Make the folder here:
            os.makedirs(self.save_path, exist_ok=True)

        if self.save_best and avg_reward > best_saved_avg_rwd:
            with open(os.path.join(self.save_path, f"{str(self)}_score.txt"), 'w') as f:
                f.write(str(avg_reward))
            self.save(os.path.join(self.save_path, str(self)))
            print("new best model saved")
        self.avg_eval_rwd = avg_reward
        self.step_to_avg_eval_rwd[self.env_steps] = avg_reward
        return avg_reward

    def _beta_scheduler(self, beta_schedule, total_timesteps):
        # setup beta scheduling
        if beta_schedule == 'exp':
            self.betas = torch.exp(torch.linspace(np.log(self.beta), np.log(
                self.beta_end), total_timesteps)).to(self.device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(
                self.beta, self.beta_end, total_timesteps).to(self.device)
        elif beta_schedule == 'none':
            if self.beta == 'auto':
                self.betas = torch.tensor([1.0] * total_timesteps).to(self.device)
            else:
                self.betas = torch.tensor([self.beta] * total_timesteps).to(self.device)
        else:
            raise NotImplementedError(
                f"Unknown beta schedule: {beta_schedule}")
        return self.betas

    def _update_prior(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}_{self.env_str}"

    def save(self, path=None):
        kwargs = self.kwargs
        kwargs.pop('env_id', None)
        kwargs.pop('env', None)
        kwargs.pop('eval_env', None)
        if path is None:
            path = str(self)
        total_state = {
            "kwargs": kwargs,
            "state_dicts": find_torch_modules(self),
            "class": self.__class__.__name__
        }
        # if the path is a directory, make :
        if '/' in path:
            bp = path.split('/')
            base_path = os.path.join(*bp[:-1])
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        torch.save(total_state, path)

    @staticmethod
    def load(path, args=None, **new_kwargs):
        state = torch.load(path, weights_only=False)
        cls = BaseAgent
        for cls_ in BaseAgent.__subclasses__():
            if cls_.__name__ == state['class']:
                cls = cls_
        if args is None:
            args = state['kwargs'].get('args', [])
        print(args)
        kwargs = state['kwargs']
        kwargs.update(new_kwargs)
        agent = cls(*args, **kwargs)
        for k, v in state['state_dicts'].items():
            attrs = k.split('.')
            module = agent
            for attr in attrs:
                module = getattr(module, attr)
            module.load_state_dict(v)
        return agent
