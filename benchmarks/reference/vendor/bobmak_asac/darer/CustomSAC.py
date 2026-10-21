import os
import time
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import gymnasium as gym
from stable_baselines3 import SAC

from BaseAgent import LOG_PARAMS
from utils import log_class_vars, logger_at_folder


class CustomSAC(SAC):
    def __init__(self,
                 env_id=None,
                 policy='MlpPolicy',
                 log_interval=500, 
                 hidden_dim=64, 
                 tensorboard_log=None,
                 name_suffix='', 
                 max_eval_steps=1000,
                 save_best=False,
                 save_path='best_models',
                 **kwargs):
        policy_kwargs = {'net_arch': [hidden_dim, hidden_dim]}
        kwargs.pop('env', None)
        kwargs.pop('render', None)
        kwargs.pop('render_mode', None)
        super().__init__(policy=policy, env=env_id, verbose=4, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, **kwargs)
        
        self.log_interval = log_interval
        self.eval_auc = 0
        self.eval_time = 0
        self.initial_time = time.thread_time_ns()
        self.env_id = env_id
        self.tensorboard_log = tensorboard_log
        self.name_suffix = name_suffix
        self.save_best = save_best
        self.save_path = save_path
        if isinstance(self.env_id, gym.Env):
            # just make a direct copy for the evaluation environment
            import copy
            self.eval_env = copy.deepcopy(self.env_id)
        else:
            self.eval_env = gym.make(env_id, max_episode_steps=max_eval_steps) if env_id else None
        
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # sample 20 actions for each next state
                do_sampling = False
                if do_sampling:
                    next_actions, next_log_prob = [self.actor.predict(replay_data.next_observations, 
                                                                  deterministic=False, 
                                                                  deterministic_with_noise=False)
                                                  for _ in range(20)]
                    next_actions = th.cat(next_actions, dim=0)
                    # mean over 20 actions
                
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_v_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/mean_q", next_q_values.mean().item())

        self._logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._logger.record("train/ent_coef", np.mean(ent_coefs))
        self._logger.record("train/actor_loss", np.mean(actor_losses))
        self._logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self._logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

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
        log_class_vars(self, self._logger, LOG_PARAMS, use_wandb=False)

        
        self._logger.dump(step=self.env_steps)
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
        
        self._logger.record('eval/avg_episode_length', n_steps / n_episodes)
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time + 1e-12) / 1e9
        eval_fps = n_steps / eval_time
        self._logger.record('eval/time', eval_time)
        self._logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        # self.step_to_avg_eval_rwd[self.env_steps] = avg_reward
        return avg_reward
    
    def evaluation_policy(self, state):
        return self.predict(state, deterministic=True)[0]

    # overwrite the on_step method to log stats proper intervals:
    def _on_step(self) -> None:
        super()._on_step()
        self.env_steps = self.num_timesteps
        self.num_episodes = self._episode_num
        if isinstance(self.ent_coef, str):
            self.beta = 1
        elif isinstance(self.ent_coef, th.Tensor):
            self.beta = 1 / self.ent_coef.detach().item() 
        elif isinstance(self.ent_coef, float):
            self.beta = 1 / self.ent_coef
        if self.num_timesteps % self.log_interval == 0:
            self._log_stats()

    def __str__(self):
        return f"{self.__class__.__name__}_{self.env_id}"

    def save(self, path, **kwargs):
        # remove unpickleable logger:
        #del self._logger
        super().save(path, include=['tensorboard_log'], **kwargs)

    def load(self, path, **kwargs):
        agent = super().load(path, **kwargs)
        return agent


if __name__ == '__main__':
    env = 'Pendulum-v1'
    env = 'HalfCheetah-v4'
    kwargs = {
        'learning_starts': 10_000,
        'batch_size': 256,
        'buffer_size': 100_000,
        'ent_coef': '0.2',
    }
    agent = CustomSAC(policy='MlpPolicy', env_id=env, device='auto', **kwargs, tensorboard_log='ft_logs/EVAL')
    agent.save('sac')
    agent.load('sac')

