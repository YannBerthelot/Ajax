from collections.abc import Sequence
from typing import Any, Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.DAgger.state import DAggerConfig
from ajax.agents.DAgger.train_DAgger import make_train
from ajax.agents.loop import LOOP_PHASES
from ajax.buffers.utils import get_buffer
from ajax.extensions.base import Extension
from ajax.types import EnvType


class DAgger(ActorCritic):
    """DAgger (Ross, Gordon and Bagnell, AISTATS 2011): imitation of an
    expert by dataset aggregation, with a supervised loss only.

    At iteration ``i`` the expert acts with probability ``beta_i =
    beta_decay ** (i - 1)`` at each step, the actor otherwise; every visited
    state joins the replay with the expert's action there, and the actor
    fits those actions (:mod:`ajax.agents.DAgger.train_DAgger`). It needs no
    reward and has no critic. The expert follows the conventions of the
    expert guide: it may be stateful; a discrete expert returns its action
    index as an integer ``(n, 1)`` column.

    Defaults: the paper's parameter-free ``beta_i = I(i = 1)``
    (``beta_decay=0``, its Sec. 3: "often performs best in practice"); the
    rest is the reference code's, HumanCompatibleAI/imitation@e5ef188:
    500-step iterations (``rollout_round_min_timesteps``) and its behaviour
    cloning's batch of 32, 32-32 tanh network and torch Adam (learning rate
    1e-3, eps 1e-8). Its cloning fits a Gaussian by likelihood with an
    entropy bonus; this actor is deterministic, as the paper's policy, and
    fits by squared error (cross-entropy for discrete actions).
    """

    name: str = "DAgger"
    capabilities: frozenset = frozenset("actor obs_networks off_policy replay".split())
    supported_extension_phases: frozenset = LOOP_PHASES | {"actor_loss"}
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )
    # Its replay is not handed to pretraining: it stores the expert's next
    # action, which a transition dataset does not hold (PriorData).
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        expert_policy: Any,
        n_envs: int = 1,
        beta_decay: float = 0.0,
        iteration_steps: int = 500,
        actor_learning_rate: float = 1e-3,
        actor_architecture=("32", "tanh", "32", "tanh"),
        buffer_size: int = int(1e6),
        batch_size: int = 32,
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
        if not 0.0 <= beta_decay <= 1.0:
            raise ValueError(f"DAgger's beta_decay is in [0, 1], got {beta_decay}.")
        if expert_policy is None:
            raise ValueError("DAgger imitates an expert: give it an expert_policy.")
        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=actor_learning_rate,  # the critic is unused
            actor_architecture=actor_architecture,
            critic_architecture=actor_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            normalize_observations=normalize_observations,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        # torch's Adam eps, the reference's (Ajax's shared default is 1e-5).
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)

        self.agent_config: DAggerConfig = DAggerConfig(
            beta_decay=beta_decay, iteration_steps=iteration_steps
        )
        self.buffer = get_buffer(buffer_size, batch_size, n_envs)
        self.train_kwargs = {
            "buffer": self.buffer,
            "batch_size": batch_size,
            "expert_policy": self.expert_policy,
        }
