from collections.abc import Mapping, Sequence
from typing import Callable, Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.CrossQ.state import CrossQConfig
from ajax.agents.CrossQ.train_CrossQ import make_train
from ajax.agents.loop import REPLAY_BATCH_KEYS
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.state import AlphaConfig
from ajax.types import EnvType


def batch_renorm_mlp(width: int) -> tuple[str, ...]:
    """The authors' MLP: Batch Renorm on the input and after each of two
    ReLU layers of ``width`` units (``sbx/sac/policies.py:199-231``)."""
    layer = (str(width), "relu", "batchrenorm")
    return ("batchrenorm", *layer, *layer)


class CrossQ(ActorCritic):
    """CrossQ (Bhatt et al., ICLR 2024): SAC without target networks, its
    actor and critics batch-renormalised, its critics trained on one pass
    over the rows and their next states, for continuous action spaces."""

    name: str = "CrossQ"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble discounted obs_networks "
        "off_policy policy_density q_critic replay".split()
    )
    supported_extension_phases: frozenset = frozenset(
        "pretrain post_update eval_metrics on_target critic_loss actor_loss".split()
    )
    phase_batch_keys: Mapping[str, frozenset] = REPLAY_BATCH_KEYS
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )
    trains_batch_norm: bool = True
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs", "return_to_go"})
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        learning_rate: float = 1e-3,
        alpha_learning_rate: Optional[float] = None,
        actor_width: int = 256,
        critic_width: int = 2048,
        num_critics: int = 2,
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = 5000,
        policy_delay: int = 3,
        adam_beta_1: float = 0.5,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,
        target_entropy_per_dim: float = -1.0,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # Expert for CloningPretrain and the eval expert-bias metric; its
        # actions, from its state along each trajectory, are stored for the
        # extensions that read them.
        expert_policy: Optional[Callable] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        """
        Initialize the CrossQ agent, by default as the authors' ``train.py
        -algo crossq`` (adityab/CrossQ@a318a26, ``train.py:126-136``,
        ``:171-200``).

        Args:
            learning_rate (float): The actor's and the critics' (and the
                temperature's, unless ``alpha_learning_rate`` is given).
            actor_width (int): The units of the actor's two hidden layers.
            critic_width (int): The units of the critics' two hidden layers;
                the paper's are 2048 wide (SBX's 1024).
            adam_beta_1 (float): The actor's and the critics' Adam beta1;
                the temperature's Adam keeps 0.9, as the authors'.
        """
        self.config = run_config(locals(), self.name)
        self.expert_policy = expert_policy
        # The authors' layers are flax's defaults: lecun-normal kernels and
        # zero biases, the mean and log-std heads included.
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            actor_architecture=batch_renorm_mlp(actor_width),
            critic_architecture=batch_renorm_mlp(critic_width),
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            squash=True,
            encoder_kernel_init="lecun_normal",
            value_kernel_init="lecun_normal",
            mean_kernel_init="lecun_normal",
            log_std_kernel_init="lecun_normal",
            log_std_init=0.0,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        # optax.adam's eps, 1e-8, which the authors' Adams keep.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(
            eps=1e-8, beta_1=adam_beta_1
        )
        self.critic_optimizer_args = self.critic_optimizer_args.replace(
            eps=1e-8, beta_1=adam_beta_1
        )
        self.alpha_args = AlphaConfig(
            learning_rate=learning_rate
            if alpha_learning_rate is None
            else alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("CrossQ only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        self.agent_config: CrossQConfig = CrossQConfig(
            gamma=gamma,
            target_entropy=target_entropy_per_dim * action_dim,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            policy_delay=policy_delay,
        )
        self.buffer = get_buffer(buffer_size, batch_size, n_envs)
        self.train_kwargs = {
            "buffer": self.buffer,
            "alpha_args": self.alpha_args,
            "num_critics": num_critics,
            "expert_policy": self.expert_policy,
        }
