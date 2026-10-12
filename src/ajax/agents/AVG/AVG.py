from collections.abc import Sequence
from typing import Callable, Optional

from gymnax import EnvParams

from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import make_train
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.deprecation import warn_inert
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.state import AlphaConfig
from ajax.types import EnvType

#: Why AVG's temperature keywords have no effect.
INERT_TEMPERATURE = (
    "AVG keeps its entropy coefficient at alpha_init, as AVG's reference code"
    " does (gauthamvasan/avg, avg.py), so nothing tunes it."
)


class AVG(ActorCritic):
    """Action Value Gradient (AVG) from Vasan et al. 2024. See https://arxiv.org/abs/2411.15370

    AVG has no memory option (``supports_memory`` stays False): its
    fully-incremental single-transition updates give length-1 BPTT, so
    memory weights could not learn temporal structure without eligibility
    traces / RTRL.
    """

    name: str = "AVG"
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 6.3e-3,
        critic_learning_rate: float = 8.7e-3,
        # Inert, warned, to be removed (INERT_TEMPERATURE).
        alpha_learning_rate: Optional[float] = None,
        actor_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        critic_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        learning_starts: int = 0,
        reward_scale: float = 1.0,
        alpha_init: float = 0.07,
        target_entropy_per_dim: Optional[float] = None,
        beta_1: float = 0,
        beta_2: float = 0.999,
        num_critics: int = 1,
        # Expert for the eval expert-bias metric only.
        expert_policy: Optional[Callable] = None,
        # Expose the most recent ``(T=1, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        """
        Initialize the AVG agent.

        Args:
            alpha_init (float): The (fixed) entropy coefficient.
            beta_1 (float): Adam's first-moment decay (AVG uses 0).
        """
        self.config = run_config(locals(), self.name)
        warn_inert(
            "AVG",
            INERT_TEMPERATURE,
            alpha_learning_rate=alpha_learning_rate,
            target_entropy_per_dim=target_entropy_per_dim,
        )

        # AVG normalises its observations (never its rewards) and squashes
        # its actions.
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
            normalize_observations=True,
            squash=True,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("AVG only supports continuous action spaces.")

        # AVG's penultimate normalisation and Adam betas.
        self.network_args = self.network_args.replace(penultimate_normalization=True)
        self.actor_optimizer_args = self.actor_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2
        )
        self.critic_optimizer_args = self.critic_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2
        )
        # The temperature's optimizer never steps: its rate is inert.
        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate or 0.0, alpha_init=alpha_init
        )
        self.agent_config: AVGConfig = AVGConfig(
            gamma=gamma,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            num_critics=num_critics,
            expose_recent_rollout=expose_recent_rollout,
        )
        self.train_kwargs = {
            "alpha_args": self.alpha_args,
            "expert_policy": self.expert_policy,
        }
