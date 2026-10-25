"""Analytic Policy Gradient (APG): a deterministic controller trained by
back-propagating the closed-loop return through a differentiable simulator.

One iteration on the own-iteration family's loop (OwnIterationAgent), each
line a method or function below::

    sample n_envs systems from the system class          (SystemClass.sample)
    roll the controller out in closed loop for horizon steps, gradients
        flowing through policy, memory carry and environment
                                   compute_rollout_returns (closed_loop_rollout)
    one AdamW step of the controller down -mean_i sum_k r_k       APG.iterate

On a model-reference tracking task (ajax.environments.model_reference) that
objective is the closed-loop matching cost of Busetto et al. 2024, eq. (9),
and sqrt(loss / horizon) is their M-RMSE. No critic, no replay, no
exploration noise: the environment must expose transition gradients (gymnax
with_transition_gradients). The controller is any Controller
(ajax.agents.APG.networks): an MLP, memory-augmented with memory=..., closed
by a learnable PID layer with pid=..., or a downstream module
(controller_factory).

Each evaluation draws num_episode_test systems (and, through the env's
reset, references and initial conditions) from the evaluation key, fresh at
each evaluation and apart from the training draws, and reports the
deterministic closed-loop return and M-RMSE, the paper's validation metric.

APG.contextual_controller is the configuration of Busetto, Breschi,
Forgione, Piga and Formentin, "One controller to rule them all"
(arXiv:2411.06482, 2024): a decoder-only transformer over the tracking
history with a PID output layer, trained on the model-reference matching
cost over a system class, with AdamW and a warmup-cosine schedule. Pair it
with ModelReferenceWrapper (ajax.environments.model_reference) for the task
and train_curriculum (ajax.agents.APG.curriculum) for the paper's staged
training.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.serialization import to_state_dict

from ajax.agents.APG.networks import Controller, PIDHeadConfig
from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    OwnIterationAgent,
    TrainContext,
)
from ajax.agents.update_steps import apply_post_update
from ajax.controller import ActingState, policy_action_dim
from ajax.environments.differentiable import (
    Rollout,
    closed_loop_rollout,
    with_transition_gradients,
)
from ajax.environments.interaction import act_through_expert, init_collector_state
from ajax.environments.system_class import FixedSystem, SystemClass
from ajax.environments.utils import (
    agent_action_to_env,
    check_env_is_gymnax,
    get_state_action_shapes,
)
from ajax.extensions.base import Extension
from ajax.extensions.capabilities import NotWired
from ajax.modules.pid_actor import PIDPolicy
from ajax.networks.memory import (
    MemoryConfig,
    parse_memory_config,
    with_default_init,
    zeros_carry_like,
)
from ajax.networks.utils import get_adam_tx
from ajax.schedule import warmup_cosine_schedule
from ajax.state import (
    BaseAgentState,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import EnvType

# Identity at initialisation (K_P=1, K_I=K_D=0). The reference code
# initialises all three coefficients to 1.0 on an unbounded output; under
# Ajax's tanh-bounded action that init trained markedly worse (pendulum
# tracking, fresh-system M-RMSE 0.33 vs 0.25 for the identity init). These
# runs, and the deviations of contextual_controller, predate the map of
# actions onto the env's bounds: the torque was capped at +-1 of +-2.
_DEFAULT_PID = PIDHeadConfig()

# A transformer memory starts as the contextual controller's reference code
# starts its GPT: normal(0.02) kernels, the residual projections at
# 0.02/sqrt(2 n_layer), zero biases (buswayne/in-context-controller @7eb094c,
# transformer_onestep.py:162-166, :189-195). Its other kinds have no
# reference and keep flax's.
APG_MEMORY_INITS = {"transformer": "nanogpt"}

#: A controller factory builds the policy network from the agent's context:
#: a flax module honouring the Controller contract (ajax.agents.APG.networks),
#: a static stateful attribute, initialize_carry(key, batch_size) when
#: stateful, and __call__ returning a distrax distribution (and the new carry
#: when stateful).
ControllerFactory = Callable[..., Any]


@dataclass(kw_only=True, eq=False, repr=False)
class APG(GradientClipping, ExpertGuided, OwnIterationAgent):
    """Analytic Policy Gradient through a differentiable simulator (the
    module's program).

    Args:
        horizon: The closed-loop rollout length differentiated through.
        learning_rate: AdamW's step size.
        beta_1: AdamW's first-moment decay.
        beta_2: AdamW's second-moment decay.
        lr_schedule: None (constant) or "warmup_cosine": linear warmup over
            warmup_steps updates, then cosine decay to learning_rate *
            lr_end_fraction at the call's last update, for the parameters of
            two or more dimensions; the others keep learning_rate, as in the
            reference (see contextual_controller). When train keeps the
            optimiser (reset_optimizer_on_resume=False), a resumed call runs
            the schedule of the call that built it, the run's first or a
            checkpoint's skeleton: given that flag, that call decays to the
            run's last update, at its total_timesteps.
        warmup_steps: The schedule's warmup, in updates.
        lr_end_fraction: The schedule's final rate, a fraction of the peak.
        max_grad_norm: Global-norm gradient clipping, on by default: BPTT
            over 100 steps of a nonlinear plant across a system class
            diverged without it in our runs (the loss rose 1.2 -> 1.7 M-RMSE
            within a stage) and was stable with it. None disables, as
            contextual_controller does by default (the reference code does
            not clip).
        actor_architecture: The MLP encoder in front of the memory and the
            output; () for a linear embedding only (the paper's choice).
        memory: A MemoryConfig (or its dict form) for a memory-augmented
            controller; None for a memoryless one.
        pid: A PIDHeadConfig to close the controller with a learnable PID
            layer; None for a plain output.
        squash: tanh-bound actions to [-1, 1] (Ajax's convention). Squashed
            or not, [-1, 1] is mapped onto the env's bounds (an unsquashed
            action is clipped there first).
        system_class: The distribution of env params to train across; None
            trains on the env's own params only.
        controller_factory: A builder of a custom policy module, called with
            the env_args, network_args, action_dim, pid and squash keywords;
            it must honour the Controller contract (ajax.agents.APG.networks).
            Downstream projects plug their own architectures into the same
            trainer without touching Ajax.

    An expert_policy runs in the rollout alongside the controller: a
    residual policy or gain mode acts through it, its gradient included.
    """

    n_envs: int = 8
    horizon: int = 100
    learning_rate: float = 5e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    lr_schedule: Optional[str] = None
    warmup_steps: int = 0
    lr_end_fraction: float = 0.1
    max_grad_norm: Optional[float] = 1.0
    actor_architecture: Sequence[str] = ("128", "relu", "128", "relu")
    memory: Optional[Union[MemoryConfig, dict]] = None
    pid: Optional[PIDHeadConfig] = None
    squash: bool = True
    system_class: Optional[SystemClass] = None
    controller_factory: Optional[ControllerFactory] = None

    def __post_init__(self) -> None:
        # Resolved before the run config records them: the log keeps what runs.
        self.memory = with_default_init(
            parse_memory_config(self.memory), APG_MEMORY_INITS
        )
        self.actor_architecture = tuple(self.actor_architecture)
        super().__post_init__()

    def check(self) -> None:
        # Refused at construction, before a train call starts anything (its
        # logging worker).
        if self.lr_schedule not in (None, "warmup_cosine"):
            raise ValueError(
                f"Unknown lr_schedule {self.lr_schedule!r}; use 'warmup_cosine'"
            )
        if self.lr_schedule is not None and not isinstance(
            self.learning_rate, (int, float)
        ):
            raise ValueError("lr_schedule='warmup_cosine' needs a float learning_rate")
        # Refused, as gain mode's unwired modules are, until APG's evaluation
        # logs an expert's return.
        if self.eval_expert_policy is not None:
            reason = "APG's evaluation does not roll out an expert yet"
            raise NotWired(f"{reason}.", {"eval_expert_policy": reason})

    def actor_critic_args(self) -> dict[str, Any]:
        return {"actor_learning_rate": self.learning_rate}  # the only network

    def optimizer_settings(self) -> dict[str, Any]:
        # Adam's eps is torch's default, 1e-8, which the reference's AdamW
        # keeps (buswayne/in-context-controller at 7eb094ce17:
        # transformer_onestep_CL.py:86); Ajax's shared default is 1e-5.
        return {"beta_1": self.beta_1, "beta_2": self.beta_2, "eps": 1e-8}

    def setup(self) -> None:
        if not check_env_is_gymnax(self.env_args.env):
            raise ValueError("APG needs a gymnax env (differentiable transitions).")
        self.env_args = self.env_args.replace(
            env=with_transition_gradients(self.env_args.env)
        )

    def check_recurrent_options(self, extensions: Sequence[Extension]) -> None:
        # Its rollout carries the controller's memory through the expert and
        # the extensions.
        del extensions

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
        final_norm: bool = True,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 5000,
        pid: Optional[PIDHeadConfig] = _DEFAULT_PID,
        beta_2: float = 0.95,
        lr_schedule: Optional[str] = "warmup_cosine",
        max_grad_norm: Optional[float] = None,
        actor_architecture: Sequence[str] = (),
        **kwargs: Any,
    ) -> APG:
        """The in-context controller of Busetto et al. 2024.

        Defaults follow the paper (Table 2: 8 layers, 4 heads, context 100,
        width 128, b=1 system per iteration, N=100 horizon) and the
        reference code (AdamW betas (0.9, 0.95), eps 1e-8, 5k-step warmup
        then cosine decay to lr/10, no gradient clipping). Deviations,
        each backed by runs on the pendulum tracking test bed: actions are
        tanh-bounded (Ajax convention; the reference leaves u unbounded),
        and the PID head starts at the identity rather than the reference's
        all-ones coefficients (see _DEFAULT_PID). As in the reference
        (buswayne/in-context-controller at 7eb094ce17), which sets its
        scheduled rate on its first parameter group only
        (train_onestep_lin_CL.py:241), the schedule steps the matrices
        alone: the biases, LayerNorm scales and PID gains keep the peak
        rate. Not reproduced: the reference decays only tensors of two or
        more dimensions (transformer_onestep_CL.py:72-77; moot at
        weight_decay=0). Not
        backed by runs: the LayerNorms, ln_f included, keep a bias and
        flax's epsilon 1e-6, where the reference's have no bias and epsilon
        1e-5 (buswayne/in-context-controller at 7eb094ce17:
        transformer_onestep.py:33 calls F.layer_norm with 1e-5, :156 builds
        ln_f = LayerNorm(n_embd, bias=config.bias), and
        train_onestep_lin_CL.py:103 defaults --bias to False). The
        transformer starts as the reference's GPT does (APG_MEMORY_INITS).
        final_norm=False rebuilds the transformer without its final
        LayerNorm, only to restore a checkpoint saved before it (see
        MemoryConfig); an expand-then-contract shim (CLAUDE.md, refactoring
        rule 10), it is removed together with MemoryConfig.final_norm. Extra
        kwargs go to APG.
        """
        return cls(
            env_id,
            n_envs=n_envs,
            horizon=horizon,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_2=beta_2,
            lr_schedule=lr_schedule,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            actor_architecture=actor_architecture,
            memory=MemoryConfig(
                kind="transformer",
                hidden_size=d_model,
                num_layers=n_layers,
                num_heads=n_heads,
                window=context,
                final_norm=final_norm,
            ),
            pid=pid,
            system_class=system_class,
            **kwargs,
        )

    def count_iteration_steps(self, context: TrainContext) -> tuple[int, str]:
        return context.env_args.n_envs * self.horizon, "n_envs x horizon"

    def configure_call(
        self,
        context: TrainContext,
        n_iterations: int,
        reset_optimizer_on_resume: bool = True,
    ) -> tuple[TrainContext, Optional[Callable[[Any], Any]]]:
        """The call's optimiser, its learning-rate schedule over the call's
        updates, and its rebuild on resume (reset_optimizer_on_resume: a
        call is a curriculum stage).

        The schedule counts the optimiser's steps and lives in its update
        rule (tx), a static field of the state: the call that builds the
        optimiser sets it. By default a resume rebuilds it (reset_optimizer),
        so it spans each call's updates. Kept, the resumed call runs the tx
        of the state it is given, built by the run's fresh call or a
        checkpoint's 0-step skeleton; there it spans the run's updates, from
        0 to the horizon, and a resumed call's value goes unused.
        """
        per_update = self.count_iteration_steps(context)[0]
        schedule_updates = (
            n_iterations
            if reset_optimizer_on_resume
            else max(context.total_timesteps // per_update, 1)
        )
        if self.lr_schedule == "warmup_cosine":
            peak = context.actor_optimizer_args.learning_rate
            assert isinstance(peak, (int, float))  # check refuses others
            schedule = warmup_cosine_schedule(
                float(peak),
                # keep at least one decay step for very short runs
                warmup_steps=min(self.warmup_steps, schedule_updates - 1),
                total_steps=schedule_updates,
                end_value_fraction=self.lr_end_fraction,
            )
            # The 1-D parameters keep the peak rate, as in the reference (see
            # contextual_controller).
            optimizer = context.actor_optimizer_args.replace(
                learning_rate=schedule, vector_learning_rate=float(peak)
            )
            context = dataclasses.replace(context, actor_optimizer_args=optimizer)
        if not reset_optimizer_on_resume:
            return context, None
        return context, partial(
            reset_optimizer, optimizer_args=context.actor_optimizer_args
        )

    def init(self, key: jax.Array, context: TrainContext) -> APGState:
        return make_agent_state(
            key,
            context.env_args,
            context.actor_optimizer_args,
            context.network_args,
            self.pid,
            self.squash,
            controller_factory=self.controller_factory,
        )

    def iterate(
        self, agent_state: APGState, context: TrainContext
    ) -> tuple[APGState, APGAuxiliaries]:
        """One update on a closed-loop rollout through the simulator, then
        the extensions' post_update."""
        training_key, rollout_key, extensions_key = jax.random.split(agent_state.rng, 3)
        agent_state = agent_state.replace(rng=training_key)
        env_args = context.env_args
        horizon = self.horizon
        timestep = agent_state.collector_state.timestep

        def compute_loss(params: Any) -> tuple[jax.Array, tuple[jax.Array, dict]]:
            returns, rollout = compute_rollout_returns(
                agent_state,
                params,
                rollout_key,
                env_args,
                self.system_class,
                horizon,
                env_args.n_envs,
                expert_policy=self.expert,
            )
            matching_loss = -returns.mean()
            # Along the rollout: what the controller saw (APG normalises no
            # observation, so they are the env's own) and its actions.
            actor_batch = {
                "rollout": rollout,
                "returns": returns,
                "actor_params": params,
                "actor_state": agent_state.actor_state,
                "observations": rollout.obs,
                "raw_observations": rollout.obs,
                "pi_mean": rollout.policy_action,
            }
            extra, metrics = context.extensions.compute_extensions_actor_loss(
                agent_state,
                actor_batch,
                timestep,
                extensions_key,
                context.total_timesteps,
            )
            return matching_loss + extra, (matching_loss, metrics)

        (loss, (matching_loss, metrics)), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(agent_state.actor_state.params)
        actor_state = agent_state.actor_state.apply_gradients(grads=grads)
        new_timestep = timestep + env_args.n_envs * horizon
        agent_state = agent_state.replace(
            actor_state=actor_state,
            collector_state=agent_state.collector_state.replace(timestep=new_timestep),
            n_updates=agent_state.n_updates + 1,
        )
        agent_state = apply_post_update(
            context.extensions,
            agent_state,
            agent_state.collector_state.timestep,
            context.total_timesteps,
        )
        return agent_state, APGAuxiliaries(
            loss=loss,
            matching_loss=matching_loss,
            m_rmse=jnp.sqrt(jnp.maximum(matching_loss, 0.0) / horizon),
            grad_norm=optax.global_norm(grads),
            timestep=new_timestep,
            extensions=metrics,
        )

    def report(self, agent_state: APGState, aux: APGAuxiliaries) -> dict:
        del agent_state
        metrics = {
            "timestep": aux.timestep,
            "Train/loss": aux.loss,
            "Train/matching_loss": aux.matching_loss,
            "Train/m_rmse": aux.m_rmse,
            "Train/grad_norm": aux.grad_norm,
        }
        metrics.update(aux.extensions)
        return metrics

    def evaluate(
        self,
        agent_state: APGState,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """The deterministic closed-loop return and M-RMSE on
        num_episode_test fresh systems, over the training horizon."""
        returns, _ = compute_rollout_returns(
            agent_state,
            agent_state.actor_state.params,
            key,
            context.env_args,
            self.system_class,
            self.horizon,
            num_episode_test,
            expert_policy=self.expert,
        )
        return {
            "Eval/episodic mean reward": returns.mean(),
            "Eval/m_rmse": jnp.sqrt(jnp.maximum(-returns.mean(), 0.0) / self.horizon),
        }

    def make_iteration_output(
        self, aux: APGAuxiliaries, evaluations: Optional[dict]
    ) -> APGAuxiliaries:
        """Every update's metrics, as train returns them, with what it
        evaluated and logged (NaN on the updates that do not evaluate; None
        without a logging config)."""
        return dataclasses.replace(aux, evaluations=evaluations)

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "APG"
    # Its env must expose transition gradients (setup), which refuses a
    # discrete one with that reason.
    action_space = "any"
    capabilities = frozenset({"actor", "obs_networks", "reward_learning"})
    # It acts through the run's transform itself (transform_action, actuate).
    supported_extension_phases = OwnIterationAgent.supported_extension_phases.union(
        {"actor_loss", "transform_action", "actuate"}
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "actor_loss": frozenset(
            {
                "actor_state",
                "observations",
                "raw_observations",
                "pi_mean",
                "rollout",
                "returns",
            }
        ),
    }
    # Its rollout traces the env params (a system class's samples).
    supports_batched_env_params = True
    supports_memory = True
    returns_iterations = True


# -- APG's controller, state and rollout, in the order an update uses them ----
def make_controller(
    env_args: EnvironmentConfig,
    network_args: NetworkConfig,
    pid: Optional[PIDHeadConfig],
    squash: bool,
    factory: Optional[ControllerFactory] = None,
) -> nn.Module:
    """The controller: factory's module, or a Controller of the network's
    hidden layers and memory."""
    action_dim = policy_action_dim(env_args)  # gain mode's gains
    if factory is not None:
        return factory(
            env_args=env_args,
            network_args=network_args,
            action_dim=action_dim,
            pid=pid,
            squash=squash,
        )
    architecture = network_args.actor_architecture
    if isinstance(architecture, PIDPolicy):
        raise TypeError(
            "APG's controller takes hidden layers, not a PIDPolicy: its PID is"
            " the output head, pid=PIDHeadConfig(...)."
        )
    return Controller(
        input_architecture=tuple(architecture),
        action_dim=action_dim,
        memory=network_args.memory,
        pid=pid,
        squash=squash,
    )


def make_optimizer(optimizer_args: OptimizerConfig) -> optax.GradientTransformation:
    """The controller's AdamW, its learning rate possibly scheduled."""
    return get_adam_tx(**to_state_dict(optimizer_args))


def make_agent_state(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    pid: Optional[PIDHeadConfig],
    squash: bool,
    controller_factory: Optional[ControllerFactory] = None,
) -> APGState:
    """A fresh state: the controller, its carry for a stateful one, its
    optimiser and a collector without an expert (the rollout carries one)."""
    training_key, params_key, carry_key, collector_key, evaluation_key = (
        jax.random.split(key, 5)
    )
    controller = make_controller(
        env_args, network_args, pid, squash, factory=controller_factory
    )
    obs_shape, _ = get_state_action_shapes(env_args.env)
    if controller.stateful:
        carry = controller.initialize_carry(carry_key, env_args.n_envs)
        params = controller.init(
            params_key,
            jnp.zeros((1, env_args.n_envs, *obs_shape)),
            hidden_state=carry,
            done=jnp.zeros((1, env_args.n_envs), bool),
        )
    else:
        carry = None
        params = controller.init(params_key, jnp.zeros((1, *obs_shape)))
    actor_state = LoadedTrainState.create(
        params=params,
        tx=make_optimizer(actor_optimizer_args),
        apply_fn=controller.apply,
        hidden_state=carry,
    )
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode="gymnax"
    )
    return APGState(
        rng=training_key,
        eval_rng=evaluation_key,
        actor_state=actor_state,
        critic_state=None,
        collector_state=collector_state,
    )


def reset_optimizer(agent_state: APGState, optimizer_args: OptimizerConfig) -> APGState:
    """A fresh optimiser (its config, zeroed moments, step 0) on the same
    parameters: between curriculum stages each stage runs its own
    learning-rate schedule from the top instead of inheriting the previous
    stage's step count and decayed rate."""
    optimizer = make_optimizer(optimizer_args)
    actor_state = agent_state.actor_state
    actor_state = actor_state.replace(
        tx=optimizer, opt_state=optimizer.init(actor_state.params), step=0
    )
    return agent_state.replace(actor_state=actor_state)


def make_policy_step(
    actor_state: LoadedTrainState, params: Any, stateful: bool
) -> Callable:
    """(carry, obs, resets) -> (action, carry) for closed_loop_rollout.

    stateful is the static (trace-time) counterpart of actor_state.recurrent,
    which is a pytree leaf inside jit.
    """
    if stateful:

        def policy_step(carry, obs, resets):
            pi, carry = actor_state.apply(
                params, obs[None], hidden_state=carry, done=resets[None]
            )
            return pi.mean()[0], carry

    else:

        def policy_step(carry, obs, resets):
            del resets
            return actor_state.apply(params, obs).mean(), carry

    return policy_step


def make_step_through_expert(
    policy_step: Callable, transform: Any, expert_policy: Any, n_systems: int
) -> Callable:
    """policy_step acting through the run's transform (residual RL, gain
    mode) and the tracked expert, whose state rides in the carry, (actor
    carry, expert state). The return's gradient runs through the expert too:
    it is part of the loop the controller acts in."""

    def step(carry, obs, resets):
        actor_carry, expert_state = carry
        action, actor_carry = policy_step(actor_carry, obs, resets)
        acting = ActingState(
            obs=obs, done=resets, timestep=None, expert_state=expert_state
        )
        executed, _, expert_state = act_through_expert(
            transform,
            expert_policy,
            acting,
            obs,
            action,
            n_systems,
            stop_gradient=False,
        )
        return executed, (actor_carry, expert_state)

    return step


def compute_rollout_returns(
    agent_state: APGState,
    params: Any,
    key: jax.Array,
    env_args: EnvironmentConfig,
    system_class: Optional[SystemClass],
    horizon: int,
    n_systems: int,
    stateful: Optional[bool] = None,
    expert_policy: Any = None,
) -> tuple[jax.Array, Rollout]:
    """The controller's closed-loop returns (n_systems,) on n_systems fresh
    systems (the env's own system without a system_class), and the rollout.
    The controller acts in [-1, 1], mapped onto each system's action bounds
    (agent_action_to_env, differentiable): Rollout.action is what the env
    received, Rollout.policy_action the controller's, through the run's
    transform and expert_policy where it has one.

    stateful (a static flag) defaults to whether the actor carries a hidden
    state, which make_agent_state sets exactly for a stateful controller.
    """
    if stateful is None:
        stateful = agent_state.actor_state.hidden_state is not None
    system_key, rollout_key = jax.random.split(key)
    if system_class is None:
        system_class = FixedSystem(env_args.env_params)
    env_params = system_class.sample(system_key, n_systems)
    carry = (
        zeros_carry_like(agent_state.actor_state.hidden_state, n_systems)
        if stateful
        else None
    )
    policy_step = make_policy_step(agent_state.actor_state, params, stateful)
    transform = next(iter(env_args.transforms), None)
    if transform is not None:
        expert_state = None
        if hasattr(expert_policy, "init_state"):
            expert_state = expert_policy.init_state(n_systems)
        carry = (carry, expert_state)
        policy_step = make_step_through_expert(
            policy_step, transform, expert_policy, n_systems
        )
    rollout, _ = closed_loop_rollout(
        policy_step,
        carry,
        rollout_key,
        env_args.env,
        env_params,
        horizon,
        action_map=lambda action: agent_action_to_env(action, env_args.env, env_params),
    )
    return rollout.reward.sum(0), rollout


# -- State and logged metrics -------------------------------------------------
@struct.dataclass(kw_only=True)
class APGState(BaseAgentState):
    """APG carries no value function: critic_state is None. Keyword-only so
    the narrowed default can sit before the base class's required fields."""

    critic_state: Optional[LoadedTrainState] = None  # type: ignore[assignment]


@struct.dataclass
class APGAuxiliaries:
    loss: jax.Array  # objective actually minimised (incl. extension terms)
    matching_loss: jax.Array  # -mean_i sum_k r_k, the closed-loop cost
    m_rmse: jax.Array  # sqrt(matching_loss / horizon)
    grad_norm: jax.Array  # the gradient's global norm, before clipping
    timestep: Union[int, jax.Array]
    extensions: dict  # the extensions' training metrics
    # With a logging config, what the update evaluated and logged (NaN on
    # the updates that do not evaluate), returned whatever the backend;
    # None without one.
    evaluations: Optional[dict] = None
