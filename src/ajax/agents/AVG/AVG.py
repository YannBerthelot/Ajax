from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from gymnax import EnvParams

from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import make_train
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import ACTOR_STEP_KEYS, CRITIC_LOSS_KEYS, LOOP_PHASES, TARGET_KEYS
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDPolicy
from ajax.state import AlphaConfig
from ajax.types import EnvType


class AVG(ActorCritic):
    """Action Value Gradient (AVG) from Vasan et al. 2024. See https://arxiv.org/abs/2411.15370

    AVG has no memory option (``supports_memory`` stays False): its
    fully-incremental single-transition updates give length-1 BPTT, so
    memory weights could not learn temporal structure without eligibility
    traces / RTRL.
    """

    name: str = "AVG"
    # No replay, so no warmup: AVG learns from each step as it comes, and
    # steps taken before learning_starts feed no update.
    capabilities: frozenset = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
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
    phase_batch_keys: Mapping[str, frozenset] = {
        "on_target": TARGET_KEYS,
        "critic_loss": CRITIC_LOSS_KEYS,
        "actor_loss": ACTOR_STEP_KEYS.union(
            {"observations", "raw_observations", "a_expert", "critic_state"}
        ),
    }
    pretrain_context: frozenset = frozenset({"env_args", "update"})
    # The step's transition carries what a replay batch holds: the
    # expert's stored actions and, for a reader, the raw next observation.
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs"})
    extension_context: frozenset = frozenset(
        {"actor_optimizer_args", "env_args", "expert_policy"}
    )
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 6.3e-3,
        critic_learning_rate: float = 8.7e-3,
        actor_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        critic_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        learning_starts: int = 0,
        reward_scale: float = 1.0,
        alpha_init: float = 0.07,
        beta_1: float = 0,
        beta_2: float = 0.999,
        num_critics: int = 1,
        # Expert for CloningPretrain and the eval expert-bias metric; its
        # actions, from its state along each trajectory, reach the
        # extensions that read them (ImitationLoss).
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

        # AVG normalises its observations (never its rewards) and squashes
        # its actions.
        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        # The Gaussian head's settings are the MLP actor's: a PID actor in its
        # place keeps its own log-std start (PID_LOG_STD_INIT).
        head_init: dict[str, Any] = (
            {}
            if isinstance(actor_architecture, PIDPolicy)
            else {
                "mean_kernel_init": "orthogonal(1.0)",
                "log_std_kernel_init": "orthogonal(1.0)",
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
            normalize_observations=True,
            squash=True,
            # The initialisation of AVG's reference code (gauthamvasan/avg
            # @6b6822c, avg.py:16-20, :59-62, :97-98): every Linear layer,
            # the mean and log-std heads included, starts orthogonal with
            # gain 1 and a zero bias.
            encoder_kernel_init="orthogonal(1.0)",
            value_kernel_init="orthogonal(1.0)",
            **head_init,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("AVG only supports continuous action spaces.")

        # AVG's penultimate normalisation and Adam betas. Adam's eps is
        # torch's default, 1e-8, which the reference's Adams keep
        # (gauthamvasan/avg @6b6822c avg.py:116-117, torch >= 2.7.0);
        # Ajax's shared default is 1e-5.
        # Its MultivariateNormal takes exp(log_std) as the covariance's
        # diagonal (avg.py:70-72): the head is read as a log-variance.
        self.network_args = self.network_args.replace(
            penultimate_normalization=True, log_variance_head=True
        )
        self.actor_optimizer_args = self.actor_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2, eps=1e-8
        )
        self.critic_optimizer_args = self.critic_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2, eps=1e-8
        )
        # AVG keeps its entropy coefficient at alpha_init, as its reference
        # code does (gauthamvasan/avg, avg.py): the optimizer never steps.
        self.alpha_args = AlphaConfig(learning_rate=0.0, alpha_init=alpha_init)
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
