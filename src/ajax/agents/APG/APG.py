"""Analytic Policy Gradient (APG): train a deterministic controller by
back-propagating the closed-loop return through a differentiable simulator.

The general agent trains any :class:`~ajax.agents.APG.networks.Controller`
(MLP, or memory-augmented via ``memory=MemoryConfig(...)``, optionally
closed by a learnable PID layer) on a class of systems
(:mod:`ajax.environments.system_class`) by BPTT over a fixed horizon. No
critic, no replay buffer, no exploration noise: the environment must
expose transition gradients (gymnax ``with_transition_gradients``).

:meth:`APG.contextual_controller` is the configuration of Busetto,
Breschi, Forgione, Piga and Formentin, "One controller to rule them all"
(arXiv:2411.06482, 2024): a decoder-only transformer over the tracking
history with a PID output layer, trained on the model-reference matching
cost over a system class, with AdamW and a warmup-cosine schedule. Pair
it with :class:`~ajax.environments.model_reference.ModelReferenceWrapper`
for the task and :func:`~ajax.agents.APG.curriculum.train_curriculum`
for the paper's staged training.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.APG.networks import PIDHeadConfig
from ajax.agents.APG.state import APGConfig
from ajax.agents.APG.train_APG import make_train
from ajax.agents.base import ActorCritic
from ajax.environments.differentiable import with_transition_gradients
from ajax.environments.system_class import SystemClass
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.state import OptimizerConfig
from ajax.types import EnvType

_DEFAULT_PID = PIDHeadConfig()


class APG(ActorCritic):
    name: str = "APG"
    supports_memory: bool = True

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 8,
        horizon: int = 100,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        lr_schedule: Optional[str] = None,
        warmup_steps: int = 0,
        lr_end_fraction: float = 0.1,
        max_grad_norm: Optional[float] = None,
        actor_architecture: Sequence[str] = ("128", "relu", "128", "relu"),
        memory: Optional[Union[MemoryConfig, dict]] = None,
        pid: Optional[PIDHeadConfig] = None,
        squash: bool = True,
        system_class: Optional[SystemClass] = None,
        env_params: Optional[EnvParams] = None,
        episode_length: Optional[int] = None,
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Args:
            env_id: gymnax env id or prebuilt gymnax env (e.g. a
                ``ModelReferenceWrapper``). Must support transition
                gradients.
            n_envs: systems sampled (and simulated in parallel) per update.
            horizon: closed-loop rollout length differentiated through.
            learning_rate / weight_decay / beta_1 / beta_2: AdamW settings
                (``weight_decay=0`` is plain Adam).
            lr_schedule: ``None`` (constant) or ``"warmup_cosine"``
                (linear warmup over ``warmup_steps`` updates, cosine decay
                to ``learning_rate * lr_end_fraction`` at the last update).
            max_grad_norm: optional global-norm gradient clipping.
            actor_architecture: MLP encoder in front of the memory/output;
                ``()`` for a linear embedding only (the paper's choice).
            memory: :class:`MemoryConfig` (or dict) for a memory-augmented
                controller; ``None`` for a memoryless one.
            pid: :class:`PIDHeadConfig` to close the controller with a
                learnable PID layer; ``None`` for a plain output.
            squash: tanh-bound actions to ``[-1, 1]`` (Ajax convention).
            system_class: distribution over ``EnvParams`` to train across;
                ``None`` trains on the env's own params only.
            extensions: composable research features (see
                :mod:`ajax.extensions`).
        """
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.config.update({"algo_name": "APG"})

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,  # no critic
            actor_architecture=tuple(actor_architecture),
            critic_architecture=tuple(actor_architecture),
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            memory=memory,
            episode_length=episode_length,
            extensions=extensions,
        )
        if not check_env_is_gymnax(self.env_args.env):
            raise ValueError("APG needs a gymnax env (differentiable transitions).")
        self.env_args = self.env_args.replace(
            env=with_transition_gradients(self.env_args.env)
        )
        self.actor_optimizer_args = OptimizerConfig(
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
        )
        self.agent_config = APGConfig(horizon=horizon)
        self.pid = pid
        self.squash = squash
        self.system_class = system_class
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.lr_end_fraction = lr_end_fraction

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            system_class=self.system_class,
            pid=self.pid,
            squash=self.squash,
            lr_schedule=self.lr_schedule,
            warmup_steps=self.warmup_steps,
            lr_end_fraction=self.lr_end_fraction,
            extensions=tuple(self.extension_stack.extensions),
        )

    @classmethod
    def contextual_controller(
        cls,
        env_id: str | EnvType,
        system_class: Optional[SystemClass] = None,
        n_envs: int = 1,
        horizon: int = 100,
        n_layers: int = 8,
        n_heads: int = 4,
        d_model: int = 128,
        context: int = 100,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 5000,
        pid: Optional[PIDHeadConfig] = _DEFAULT_PID,
        **kwargs: Any,
    ) -> "APG":
        """The in-context controller of Busetto et al. 2024.

        Defaults follow the paper (Table 2: 8 layers, 4 heads, context 100,
        width 128, ``b=1`` system per iteration, ``N=100`` horizon) and the
        reference code (AdamW betas (0.9, 0.95), 5k-step warmup then
        cosine decay to ``lr/10``). Extra kwargs go to :class:`APG`.
        """
        return cls(
            env_id,
            n_envs=n_envs,
            horizon=horizon,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_2=kwargs.pop("beta_2", 0.95),
            lr_schedule=kwargs.pop("lr_schedule", "warmup_cosine"),
            warmup_steps=warmup_steps,
            actor_architecture=kwargs.pop("actor_architecture", ()),
            memory=MemoryConfig(
                kind="transformer",
                hidden_size=d_model,
                num_layers=n_layers,
                num_heads=n_heads,
                window=context,
            ),
            pid=pid,
            system_class=system_class,
            **kwargs,
        )
