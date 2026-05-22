from functools import partial
from typing import Callable, Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.PQN.state import PQNConfig
from ajax.agents.PQN.train_PQN import make_train
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.types import EnvType


class PQN(ActorCritic):
    """Parallelised Q-Network (Gallici et al., 2024) for discrete action spaces.

    On-policy value-based RL: vectorised rollouts, Q(lambda) regression
    targets, no replay buffer and no target network -- training stability
    instead comes from LayerNorm in the Q-network. The policy is greedy
    over Q, with epsilon-greedy exploration during collection.
    """

    name: str = "PQN"

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 8,
        learning_rate: float = 2.5e-4,
        architecture=("128", "relu", "128", "relu"),
        gamma: float = 0.99,
        q_lambda: float = 0.65,
        n_steps: int = 128,
        n_epochs: int = 4,
        num_minibatches: int = 4,
        reward_scale: float = 1.0,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 10.0,
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
        # TD loss: None -> MSE. Pass make_huber_td_loss(delta) for Huber.
        td_loss_fn: Optional[Callable] = None,
        # Extra eval metrics: (agent_state, key) -> dict, logged each eval.
        extra_eval_metrics: Optional[Callable] = None,
    ) -> None:
        self.config = {**locals()}
        self.config.update({"algo_name": "PQN"})

        # PQN has a single Q-network; forward `architecture`/`learning_rate`
        # to both actor/critic slots of the base class.
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
            raise ValueError("PQN only supports discrete action spaces.")
        if n_steps % num_minibatches != 0:
            raise ValueError(
                f"n_steps ({n_steps}) must be divisible by num_minibatches "
                f"({num_minibatches})."
            )

        self.agent_config = PQNConfig(
            gamma=gamma,
            q_lambda=q_lambda,
            n_steps=n_steps,
            n_epochs=n_epochs,
            num_minibatches=num_minibatches,
            reward_scale=reward_scale,
        )
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frac = epsilon_decay_frac
        self.action_pipeline = action_pipeline
        self.eval_action_transform = eval_action_transform
        self.td_loss_fn = td_loss_fn
        self.extra_eval_metrics = extra_eval_metrics

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_frac=self.epsilon_decay_frac,
            action_pipeline=self.action_pipeline,
            eval_action_transform=self.eval_action_transform,
            td_loss_fn=self.td_loss_fn,
            extra_eval_metrics=self.extra_eval_metrics,
        )
