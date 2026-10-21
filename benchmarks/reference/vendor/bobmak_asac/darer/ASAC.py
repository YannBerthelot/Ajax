from typing import Optional
import yaml
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
import numpy as np
import torch
from torch.nn import functional as F
import gymnasium as gym
# import wandb
import sys
sys.path.append('darer')
from Models import OnlineQNets, Qsa, Optimizers, TargetNets, polyak_update
from BaseAgent import BaseAgent
from utils import get_max_grad, logger_at_folder
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
from stable_baselines3.sac.policies import Actor
import torch as th


# torch.backends.cudnn.benchmark = True
# raise warning level for debugger:
# import warnings
# warnings.filterwarnings("error")
class ASAC(BaseAgent):
    def __init__(self,
                 *args,
                 policy: str = 'MlpPolicy',
                 actor_learning_rate: Optional[float] = None,
                 use_ppi: bool = False,
                 use_dones: bool = True,
                 name_suffix: str = '',
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.kwargs.update(locals())
        self.kwargs.pop('self')
        self.kwargs.pop('args')
        self.kwargs.pop('kwargs')
        self.kwargs.pop('__class__')
        self.algo_name = 'ASAC' + '-no'*(not use_dones) + '-auto'*(self.beta == 'auto') + name_suffix
        self.use_dones = use_dones
        self.use_ppi = use_ppi
        self.baseline = torch.tensor(0.0, device=self.device)
        self.penalty = torch.tensor(0.0, device=self.device)
        self.actor_learning_rate = self.learning_rate if actor_learning_rate is None else actor_learning_rate
        self.nA = get_action_dim(self.env.action_space)        
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.new_theta = torch.tensor(0.0, device=self.device)
        self.theta = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self.logpi0 = th.log(th.tensor(1/self.nA, device=self.device))
        # self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        if self.beta != 'auto':
            self.ent_coef = self.beta**(-1)
        else:
            self.ent_coef = 'auto'
        self._initialize_networks()
        self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))


    def _initialize_networks(self):
        self.online_critics = OnlineQNets([Qsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator_fn=self.aggregator_fn)
        self.target_critics = TargetNets([Qsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.model = self.online_critics
        self.target_critics.load_state_dicts(
            [q.state_dict() for q in self.online_critics])
        
        self.actor = Actor(self.env.observation_space, self.env.action_space,
                    [self.hidden_dim, self.hidden_dim],
                    FlattenExtractor(self.env.observation_space),
                    self.nS,
                    )
            
        # send the actor to device:
        self.actor.to(self.device)
        # TODO: Try a fixed covariance network (no/ignored output)
        opts = [torch.optim.Adam(q.parameters(), lr=self.learning_rate)
                for q in self.online_critics]
        
        self.q_optimizers = Optimizers(opts)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                 lr=self.actor_learning_rate)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=1e-4)#, lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
            self.ent_coef_optimizer = None
        
    def exploration_policy(self, state):
        self.actor.set_training_mode(False)
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # Get a stochastic action from the actor:
        # action, _ = self.actor.predict(state)
        action, buffer_action = self._sample_action(state)
        return (action, buffer_action), 0
    
    def evaluation_policy(self, state):
        self.actor.set_training_mode(False)
        # Get a deterministic action from the actor:
        # state = torch.tensor(state, dtype=torch.float32)#.to(self.device)
        action, _ = self.actor.predict(state, deterministic=True)
        return action


    def gradient_descent(self, batch, grad_step):
        states, actions, next_states, dones, rewards = batch
        # ent_coef = self.beta ** (-1)

        optimizers = [self.actor_optimizer, self.q_optimizers]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        # self._update_learning_rate(optimizers)

        self.actor.set_training_mode(True)

        # We need to sample because `log_std` may have changed between two gradient steps
        # if self.use_sde:
        #     self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            # ent_coef_loss = -((log_prob + self.target_entropy).detach()).mean()
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

            # ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        # ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.online_critics(states, actions)

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(next_states)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.target_critics(next_states, next_actions), dim=1)
            next_q_values = self.aggregator_fn(next_q_values, dim=1)
            # add entropy term
            next_v_values = next_q_values - ent_coef * (next_log_prob.reshape(-1, 1) - self.logpi0)
            # td error + entropy term
            # target_q_values = rewards +  * self.gamma * next_q_values
            if self.use_dones:
                # make the penalty same as mean of non-terminating rewards:
                penalty = 20 * th.max(th.gather(rewards, dim=0, index=dones.long()))
                self.penalty = penalty * self.tau_theta + (1 - self.tau_theta) * self.penalty
                self.logger.record("train/penalty", self.penalty.item())
                next_v_values = next_v_values * (1 - dones) - self.penalty * dones # penalty of 100 for resetting
            new_theta = th.mean(rewards - ent_coef * (log_prob.reshape(-1, 1) - self.logpi0))
            # always subtract the mean of target q values to try and keep it centered (in terms of its span):
            self.baseline = torch.mean(torch.cat(
                self.target_critics(
                torch.zeros(self.nS),
                torch.zeros(self.nA)
                )
            ))
            # self.baseline += torch.mean(rewards - self.theta)
            next_v_values = next_v_values - self.baseline

            # log the baseline:
            self.logger.record("train/baseline", self.baseline.item())

            target_q_values = rewards - self.theta + next_v_values

            self.logger.record(f"train/next_logprob", next_log_prob.mean().item())
            self.logger.record(f"train/new_theta", new_theta.item())
            self.logger.record(f"train/mean_reward", rewards.mean().item())

        # Reduce tau_theta:
        self.logger.record("train/tau_theta", self.tau_theta)
        self.theta = self.theta * (1 - self.tau_theta) + self.tau_theta * new_theta 

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        self.logger.record("train/critic_loss", critic_loss.item())

        # self.logger.record("train/td-scaling", scaling.item())
        # Log the mean q values:
        self.logger.record("train/mean_q", next_q_values.mean().item())

        self.lr = self.q_optimizers.get_lr()
        # Optimize the critic
        self.q_optimizers.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            self.online_critics.clip_grad_norm(self.max_grad_norm)
        critic_max_grad_norm = max(get_max_grad(critic) for critic in self.online_critics)

        # Log the maximum gradient norm
        self.logger.record("train/max_grad_norm", critic_max_grad_norm)
        # log ent coef:
        self.logger.record("train/ent_coef", ent_coef.item())
        self.q_optimizers.step()

        # Compute actor loss
        # Min over all critic networks
        q_values_pi = th.cat(self.online_critics(states, actions_pi), dim=1)
        min_qf_pi = self.aggregator_fn(q_values_pi, dim=1)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        self.logger.record("train/actor_loss", actor_loss.item())

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # clip the actor gradient:
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        actor_max_grad = get_max_grad(self.actor)
        self.logger.record("train/actor_max_grad", actor_max_grad)
        self.actor_optimizer.step()

        # log newest temperature:
        self.logger.record("train/temp", ent_coef.item())
  
    def _update_target(self):
        # TODO: Make sure we use gradient steps to track target updates:
        # if gradient_step % self.target_update_interval == 0:
        for net in range(self.num_nets):
            polyak_update(self.online_critics.nets[net].parameters(), self.target_critics.nets[net].parameters(), self.tau)

    def _sample_action(self, state, n_envs=1):
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
        if self.env_steps < self.learning_starts:
            # Warmup phase
            unscaled_action = np.array([self.env.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            # assert se is not None, "self._last_obs was not set"
            unscaled_action, _ = self.actor.predict(state, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.env.action_space, gym.spaces.Box):
            scaled_action = self.actor.scale_action(unscaled_action)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.actor.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

def main():
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v3'
    # env_id = 'CartPole-v1'
    env_id = 'InvertedPendulum-v5'
    # env_id = 'Hopper-v4'
    # env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    with open(f'hparams/{env_id}/asac.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # from simple_env import SimpleEnv
    agent = ASAC(env_id, **params, device='cuda',
        tensorboard_log=f'local-asac-{env_id}',
        render=False, log_interval=200,
        save_best=True
    )
                      
    agent.learn(total_timesteps=500_000)


if __name__ == '__main__':
    main()
