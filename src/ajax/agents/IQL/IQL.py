from collections.abc import Mapping, Sequence
from typing import Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.IQL.state import IQLConfig
from ajax.agents.IQL.train_IQL import make_train
from ajax.agents.loop import ACTOR_STEP_KEYS, CRITIC_LOSS_KEYS, LOOP_PHASES, TARGET_KEYS
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.state import OptimizerConfig
from ajax.types import EnvType, FloatOrCallable

#: The reference's initialisation (common.py ``default_init``).
ORTHOGONAL_SQRT2 = "orthogonal(np.sqrt(2))"


class IQL(ActorCritic):
    """Implicit Q-Learning (Kostrikov, Nair and Levine, ICLR 2022): offline
    RL with a state value fitted as an expectile of the critic over the
    data's actions, and an actor fitted by advantage-weighted regression
    (:mod:`ajax.agents.IQL.train_IQL`). Train it on a dataset with
    ``train_offline``, then fine-tune online with ``train``.

    Defaults are the reference code's (ikostrikov/implicit_q_learning
    @09d7002, ``configs/mujoco_config.py``, ``learner.py``, ``policy.py``):
    expectile 0.7 and temperature 3 (locomotion; antmaze takes 0.9 and 10);
    256-256 ReLU networks, orthogonal(sqrt 2) initialisation; Adam at 3e-4
    with optax's eps 1e-8 for the actor, the critics and the value; the
    actor a Gaussian on a tanh mean with one learned log-std per action
    dimension, clipped to [-5, 2]; batch 256, gamma 0.99, tau 0.005. The
    reference decays the actor's offline learning rate to 0 by a cosine
    over its updates: pass ``actor_learning_rate=optax.cosine_decay_schedule(
    3e-4, n_updates)`` for that. Its locomotion rewards are scaled by 1000
    over the dataset's range of returns: ``reward_scale``.
    """

    name: str = "IQL"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble discounted obs_networks off_policy "
        "policy_density q_critic replay target_critic v_critic reward_learning".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    phase_batch_keys: Mapping[str, frozenset] = {
        "on_target": TARGET_KEYS,
        "critic_loss": CRITIC_LOSS_KEYS,
        "actor_loss": ACTOR_STEP_KEYS
        | frozenset("observations raw_observations actions critic_state".split()),
    }
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args".split()
    )
    replays_transitions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs", "return_to_go"})
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: FloatOrCallable = 3e-4,
        critic_learning_rate: FloatOrCallable = 3e-4,
        value_learning_rate: FloatOrCallable = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        expectile: float = 0.7,
        temperature: float = 3.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = 0,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        normalize_observations: bool = False,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)
        if not 0.0 < expectile < 1.0:
            raise ValueError(f"IQL's expectile is in (0, 1), got {expectile}.")
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
            normalize_observations=normalize_observations,
            encoder_kernel_init=ORTHOGONAL_SQRT2,
            mean_kernel_init=ORTHOGONAL_SQRT2,
            value_kernel_init=ORTHOGONAL_SQRT2,
            log_std_state_independent=True,
            log_std_init=0.0,
            log_std_bounds=(-5.0, 2.0),
            tanh_mean=True,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("IQL only supports continuous action spaces.")
        # optax's Adam eps, the reference's (Ajax's shared default is 1e-5).
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)
        value_optimizer_args: OptimizerConfig = self.critic_optimizer_args.replace(
            learning_rate=value_learning_rate
        )
        self.agent_config: IQLConfig = IQLConfig(
            expectile=expectile,
            temperature=temperature,
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
        )
        self.buffer = get_buffer(buffer_size, batch_size, n_envs)
        self.train_kwargs = {
            "buffer": self.buffer,
            "value_optimizer_args": value_optimizer_args,
        }
