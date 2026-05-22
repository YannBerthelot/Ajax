from functools import partial
from typing import Callable, Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.DQN.state import DQNConfig
from ajax.agents.DQN.train_DQN import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.types import EnvType


class DQN(ActorCritic):
    """Deep Q-Network (Mnih et al., 2015) for discrete action spaces.

    Value-based, off-policy. A single Q-network is trained by TD against a
    periodically refreshed target network; the policy is greedy over Q,
    with epsilon-greedy exploration during collection.
    """

    name: str = "DQN"

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        learning_rate: float = 1e-3,
        architecture=("128", "relu", "128", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 10.0,
        buffer_size: int = int(1e5),
        batch_size: int = 64,
        learning_starts: int = 1000,
        tau: float = 1.0,
        target_update_interval: int = 500,
        reward_scale: float = 1.0,
        n_gradient_steps: int = 1,
        # Linear epsilon-greedy schedule: epsilon goes from epsilon_start
        # to epsilon_end over the first epsilon_decay_frac of training.
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frac: float = 0.5,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        # --- Composable hook overrides ---
        action_pipeline: Optional[Callable] = None,
        eval_action_transform: Optional[Callable] = None,
        # TD target: None -> vanilla DQN. Pass compute_double_dqn_td_target
        # (from train_DQN) for Double DQN.
        td_target_fn: Optional[Callable] = None,
        # TD loss: None -> MSE. Pass make_huber_td_loss(delta) for Huber.
        td_loss_fn: Optional[Callable] = None,
        # Q-network module: None -> QNetwork. Pass DuelingQNetwork
        # (from DQN.networks) for the dueling architecture.
        q_network_cls: Optional[type] = None,
    ) -> None:
        self.config = {**locals()}
        self.config.update({"algo_name": "DQN"})

        # DQN has a single Q-network; forward `architecture`/`learning_rate`
        # to both actor/critic slots of the base class. The Q-network is
        # built from the critic slots and stored in `actor_state`.
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            actor_architecture=architecture,
            critic_architecture=architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
        )

        if check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("DQN only supports discrete action spaces.")

        self.agent_config = DQNConfig(
            gamma=gamma,
            tau=tau,
            target_update_interval=target_update_interval,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            n_gradient_steps=n_gradient_steps,
        )
        self.buffer = get_buffer(
            buffer_size=buffer_size, batch_size=batch_size, n_envs=n_envs
        )
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frac = epsilon_decay_frac
        self.action_pipeline = action_pipeline
        self.eval_action_transform = eval_action_transform
        self.td_target_fn = td_target_fn
        self.td_loss_fn = td_loss_fn
        self.q_network_cls = q_network_cls

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            buffer=self.buffer,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_frac=self.epsilon_decay_frac,
            action_pipeline=self.action_pipeline,
            eval_action_transform=self.eval_action_transform,
            td_target_fn=self.td_target_fn,
            td_loss_fn=self.td_loss_fn,
            q_network_cls=self.q_network_cls,
        )
