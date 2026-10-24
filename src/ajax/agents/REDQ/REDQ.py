from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES, REPLAY_BATCH_KEYS
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.REDQ.state import REDQConfig
from ajax.agents.REDQ.train_REDQ import make_train
from ajax.controller import policy_action_dim
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
)
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDPolicy
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
    capabilities: frozenset = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
        }
    )
    supported_extension_phases: frozenset = LOOP_PHASES.union(
        {
            "on_target",
            "critic_loss",
            "actor_loss",
            "transform_action",
            "actuate",
        }
    )
    phase_batch_keys: Mapping[str, frozenset] = REPLAY_BATCH_KEYS
    pretrain_context: frozenset = frozenset({"buffer", "env_args", "update"})
    extension_context: frozenset = frozenset(
        {"actor_optimizer_args", "buffer", "env_args", "expert_policy"}
    )
    supports_memory: bool = True
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs", "return_to_go"})
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
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
        # Expert for the CloningPretrain extension and the eval expert-bias
        # metric; online BC is the ImitationLoss extension. Its actions,
        # from its state along each trajectory, are stored for the
        # extensions that read them.
        expert_policy: Optional[Callable] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        """
        Initialize the REDQ agent.

        Args:
            alpha_init (float): Initial value for the temperature parameter.
                1.0 as in REDQ's official code, which learns ``log_alpha``
                from 0 (watchernyu/REDQ@7b5d1bf,
                ``redq/algos/redq_sac.py:59-61``). Its ``self.alpha = alpha``
                at ``:80`` then sets the temperature to the unused
                ``alpha=0.2`` default until its first temperature step
                (``:214-236``), on the last of the 20 critic updates
                (``utd_ratio``) of the first env step that trains: those
                updates' targets and the first actor loss read 0.2, and
                so do Ajax's (``train_REDQ.FIRST_ALPHA``).
            target_entropy_per_dim (float | "mbpo"): Target entropy per
                action dimension, or ``"mbpo"`` for MBPO's per-task target
                entropy (:data:`MBPO_TARGET_ENTROPY`), the default of REDQ's
                official code; an env id outside that table raises.
        """
        self.config = run_config(locals(), self.name)

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        # The initialisation of REDQ's reference code: watchernyu/REDQ
        # @7b5d1bf applies weights_init_ (redq/algos/core.py:18-22,
        # xavier_uniform with gain 1, bias 0) to every Linear layer of
        # its policy and Q networks, the mean and log-std layers
        # included (:96-98, :194-198), whatever their comments say.
        # Recurrent REDQ (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its recurrent reference (pomdp-baselines), is
        # audited: at REDQ's, its memory cells' actors read 24-29/32.
        reference_init: dict[str, Any] = {}
        if memory is None:
            reference_init = {
                "encoder_kernel_init": "xavier_uniform",
                "value_kernel_init": "xavier_uniform",
            }
            # The Gaussian head's settings are the MLP actor's: a PID actor in
            # its place keeps its own log-std start (PID_LOG_STD_INIT).
            if not isinstance(actor_architecture, PIDPolicy):
                reference_init.update(
                    {
                        "mean_kernel_init": "xavier_uniform",
                        "log_std_kernel_init": "xavier_uniform",
                        "log_std_init": 0.0,
                    }
                )

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            memory=memory,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            squash=True,
            **reference_init,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        # Adam's eps is torch's default, 1e-8, which the reference's Adams
        # keep (watchernyu/REDQ @7b5d1bf redq/algos/redq_sac.py:48, :51,
        # torch 1.3.1); recurrent REDQ's reference (Ni et al.,
        # pomdp-baselines e7c19c3 policy_rnn.py:75, :93) keeps it too.
        # Ajax's shared default is 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)
        # REDQ's log-probability subtracts log(1 - tanh^2 + 1e-6)
        # (redq/algos/core.py:232).
        self.network_args = self.network_args.replace(squash_log_det_eps=1e-6)

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("REDQ only supports continuous action spaces.")
        # The policy's actions: gain mode's gains.
        action_dim = policy_action_dim(self.env_args)
        target_entropy = resolve_target_entropy(
            target_entropy_per_dim, action_dim, env_id
        )
        self.agent_config: REDQConfig = REDQConfig(
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

        self.train_kwargs = {
            "buffer": self.buffer,
            "alpha_args": self.alpha_args,
            "expert_policy": self.expert_policy,
        }
