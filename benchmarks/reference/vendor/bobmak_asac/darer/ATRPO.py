import time
from typing import Callable, Generator, NamedTuple, Optional
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import gymnasium as gym
from sb3_contrib import TRPO
import copy
from functools import partial
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv


from gymnasium import spaces
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad

from BaseAgent import LOG_PARAMS
from utils import log_class_vars, logger_at_folder

class ATRPO(TRPO):
    def __init__(self,
                 env_id=None,
                 policy='MlpPolicy',
                 log_interval=500, 
                 hidden_dim=64, 
                 tensorboard_log=None,
                 name_suffix='', 
                 max_eval_steps=1000,
                 critic_l2=3e-3,
                 save_best=False,
                 **kwargs):

        # Raise a warning if save best, not implemented yet
        if save_best:
            print("Warning: save_best not implemented yet")
        policy_kwargs = {'net_arch': [hidden_dim, hidden_dim], 'log_std_init': -0.5}
        kwargs.pop('env', None)
        kwargs.pop('render', None)
        super().__init__(policy=policy, env=env_id, verbose=4, gamma=1, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, **kwargs)

        self.log_interval = log_interval
        self.eval_auc = 0
        self.eval_time = 0
        self.initial_time = time.thread_time_ns()
        self.env_id = env_id
        self.tensorboard_log = tensorboard_log
        self.name_suffix = name_suffix
        if isinstance(self.env_id, gym.Env):
            # just make a direct copy for the evaluation environment
            import copy
            self.eval_env = copy.deepcopy(self.env_id)
        else:
            self.eval_env = gym.make(env_id, max_episode_steps=max_eval_steps) if env_id else None
        self.our_logger = logger_at_folder(
            tensorboard_log,
            algo_name='ATRPO' + str(self.ent_coef) + name_suffix
        ) if tensorboard_log else None


        self.reward_rate = th.tensor(0.0, device=self.device)
        self.critic_l2 = critic_l2

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # returns, not used here
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            self.reward_rate = np.mean(self.rollout_buffer.rewards)

            advantages -= self.reward_rate
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)           

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()

            # KL divergence
            kl_div = kl_divergence(distribution, old_distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)  # type: ignore[assignment, arg-type]

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    # Applying the scaled step direction
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()

                    # New KL-divergence
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        # log the reward rate:
        self.logger.record("train/reward_rate", self.reward_rate)
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                values_pred = self.policy.predict_values(rollout_data.observations) - self.reward_rate
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten()) 
                 # weight decay
                for param in self.policy.value_net.parameters():
                    value_loss += param.pow(2).sum() * self.critic_l2
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()



        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


    def _log_stats(self):
        # end timer:
        t_final = time.thread_time_ns()
        # fps averaged over log_interval steps:
        self.fps = self.log_interval / \
            ((t_final - self.initial_time + 1e-16) / 1e9)

        if self.num_timesteps > 0:
            self.avg_eval_rwd = self.evaluate()
            self.eval_auc += self.avg_eval_rwd
        
        self.lr = 0#self.optimzers.get_lr()
        self.beta = 1
        log_class_vars(self, self.our_logger, LOG_PARAMS, use_wandb=False)

        
        self.our_logger.dump(step=self.env_steps)
        self.initial_time = time.thread_time_ns()

    def evaluate(self, n_episodes=10) -> float:
        # run the current policy and return the average reward
        self.initial_time = time.process_time_ns()
        avg_reward = 0.
        n_steps = 0
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.evaluation_policy(state)
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        
        self.our_logger.record('eval/avg_episode_length', n_steps / n_episodes)
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time + 1e-12) / 1e9
        eval_fps = n_steps / eval_time
        self.our_logger.record('eval/time', eval_time)
        self.our_logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        # self.step_to_avg_eval_rwd[self.env_steps] = avg_reward
        return avg_reward
    
    def evaluation_policy(self, state):
        return self.predict(state, deterministic=True)[0]


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            self._on_step()
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                    # subtract the termination cost:
                    rewards[idx] -= 100.0

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    # overwrite the on_step method to log stats proper intervals:
    def _on_step(self) -> None:
        self.env_steps = self.num_timesteps
        self.num_episodes = self._episode_num
        
        if self.num_timesteps % self.log_interval == 0:
            self._log_stats()

    def __str__(self):
        return f"{self.__class__.__name__}_{self.env_id}"

    def save(self, path, **kwargs):
        # remove unpickleable logger:
        del self.our_logger
        super().save(path, include=['tensorboard_log'], **kwargs)

    def load(self, path, **kwargs):
        agent = super().load(path, **kwargs)
        agent.our_logger = logger_at_folder(
            agent.tensorboard_log,
            algo_name='SAC' + str(agent.gamma) + str(agent.ent_coef) + self.name_suffix
        )
        return agent


def main():
    from sb3_contrib.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, ActorCriticPolicy

    env = 'Pendulum-v1'
    env = 'HalfCheetah-v4'
    kwargs = {
        # 'learning_starts': 10_000,
        'n_steps': 5000,
        'n_critic_updates': 1,
        'batch_size': 100,
        # 'learning_rate': 0.0003,
        'log_interval': 2500,
        'cg_max_steps': 10,
        'cg_damping': 0.01, # "Damping coeff."
        # 'buffer_size': 100_000,
        # 'ent_coef': '0.2',
        # Use MLP with tanh activations:
        # 'net_arch': [64, 64],
        # 'policy_kwargs': {'activation_fn': th.nn.Tanh},
    }
    # learning rate schedule: goes to zero after total_timesteps:
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return initial_value
            return progress_remaining * initial_value

        return func
    kwargs['learning_rate'] = linear_schedule(initial_value=0.0003)

    agent = ATRPO(policy='MlpPolicy', env_id=env, device='auto', **kwargs, tensorboard_log=f'ft_logs/tests/{env}')
    # agent.save('sac')
    # agent.load('sac')
    agent.learn(total_timesteps=1000_000)

if __name__ == '__main__':
    main()