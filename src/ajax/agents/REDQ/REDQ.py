from collections.abc import Sequence
from functools import partial
from typing import Callable, Literal, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.cloning import warn_actor_cloning_lr
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.REDQ.state import REDQConfig
from ajax.agents.REDQ.train_REDQ import make_train
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.extensions.pretrain import CloningPretrain
from ajax.networks.memory import MemoryConfig
from ajax.state import AlphaConfig
from ajax.types import EnvType

# MBPO's per-task target entropies (Janner et al., 2019), each the whole
# action's: the default of REDQ's official code (watchernyu/REDQ@7b5d1bf,
# ``redq/algos/core.py:10-13``, read at ``redq/algos/redq_sac.py:57-58``),
# here under the ids of brax's versions of those gym MuJoCo tasks.
MBPO_TARGET_ENTROPY = {
    "hopper": -1.0,
    "halfcheetah": -3.0,
    "walker2d": -3.0,
    "ant": -4.0,
    "humanoid": -2.0,
}


def resolve_target_entropy(
    per_dim: Union[float, Literal["mbpo"]], action_dim: int, env_id: object
) -> float:
    """``per_dim`` per action dimension, or with ``"mbpo"`` MBPO's target
    entropy for ``env_id``."""
    if not isinstance(per_dim, str):
        return per_dim * action_dim
    if per_dim != "mbpo":
        raise ValueError(
            f"target_entropy_per_dim takes a number or 'mbpo', not {per_dim!r}."
        )
    if not isinstance(env_id, str) or env_id not in MBPO_TARGET_ENTROPY:
        raise ValueError(
            f"target_entropy_per_dim='mbpo': MBPO's table has no {env_id!r}; it"
            f" has {sorted(MBPO_TARGET_ENTROPY)}."
        )
    return MBPO_TARGET_ENTROPY[env_id]


class REDQ(ActorCritic):
    """Randomized Ensembled Double Q-learning (Chen et al., 2021) for continuous
    action spaces."""

    name: str = "REDQ"
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    supports_memory: bool = True

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,
        target_entropy_per_dim: Union[float, Literal["mbpo"]] = -1.0,
        num_critic_updates: int = 20,
        num_critics: int = 10,
        subset_size: int = 2,
        # Pluggable memory block (see ajax.networks.memory); trains on
        # replayed sequences with R2D2-style burn-in (see ajax.agents.recurrent).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        burn_in: int = 8,
        sequence_length: int = 16,
        # R2D2 stored-state replay: initialize replayed sequences from the
        # actor carries recorded at collection time instead of zero+burn-in
        # (Kapturowski et al. 2019 show this mitigates state staleness best).
        # Stores flatten_carry(actor carry) per step in the buffer.
        stored_state: bool = False,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        actor_cloning_epochs: int = 10,
        actor_cloning_lr: Optional[float] = None,  # deprecated, ignored
        actor_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        # Expert for the cloning pre-training and the eval expert-bias
        # metric; online BC is the ImitationLoss extension.
        expert_policy: Optional[Callable] = None,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Initialize the REDQ agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
            actor_learning_rate (float): Learning rate for the actor's optimizer.
            critic_learning_rate (float): Learning rate for the critic's optimizer.
            alpha_learning_rate (float): Learning rate for the temperature's
                optimizer.
            actor_architecture (tuple): Architecture of the actor network.
            critic_architecture (tuple): Architecture of the critic network.
            gamma (float): Discount factor for rewards.
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_starts (int): Timesteps before training starts.
            tau (float): Soft update coefficient for target networks.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): Initial value for the temperature parameter.
                1.0 as in REDQ's official code, which learns ``log_alpha``
                from 0 (watchernyu/REDQ@7b5d1bf,
                ``redq/algos/redq_sac.py:59-61``). Its ``self.alpha = alpha``
                at ``:80`` then sets the temperature to the unused
                ``alpha=0.2`` default until its first temperature step
                (``:214-236``), on the last of the 20 critic updates
                (``utd_ratio``) of the first env step that trains: those
                updates' targets and the first actor loss read 0.2, a
                transient not reproduced.
            target_entropy_per_dim (float | "mbpo"): Target entropy per
                action dimension, or ``"mbpo"`` for MBPO's per-task target
                entropy (:data:`MBPO_TARGET_ENTROPY`), the default of REDQ's
                official code; an env id outside that table raises.
        """
        self.config = {**locals()}
        self.config.update({"algo_name": "REDQ"})

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            memory=memory,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            squash=True,
            extensions=extensions,
        )

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("REDQ only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        target_entropy = resolve_target_entropy(
            target_entropy_per_dim, action_dim, env_id
        )
        self.agent_config = REDQConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
            num_critic_updates=num_critic_updates,
            num_critics=num_critics,
            subset_size=subset_size,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )

        warn_actor_cloning_lr(actor_cloning_lr)
        self.cloning = CloningPretrain.from_deprecated_kwargs(
            self.expert_policy,
            self.extension_stack.extensions,
            pre_train_n_steps=pre_train_n_steps,
            actor_cloning_epochs=actor_cloning_epochs,
            actor_cloning_batch_size=actor_cloning_batch_size,
        )

    def get_make_train(self) -> Callable:
        """
        Create a training function for the REDQ agent.

        Returns:
            Callable: A function that trains the REDQ agent.
        """
        return partial(
            make_train,
            buffer=self.buffer,
            alpha_args=self.alpha_args,
            cloning=self.cloning,
            expert_policy=self.expert_policy,
            extensions=tuple(self.extension_stack.extensions),
        )
