"""Proximal Policy Optimization (Schulman, Wolski, Dhariwal, Radford and
Klimov, 2017), as its reference code, OpenAI Baselines' ppo2 (@ea25b9e),
runs it.

One iteration of the paper's Algorithm 1. The on-policy family runs the
policy for ``n_steps`` in each of ``n_envs`` envs
(:class:`~ajax.agents.families.OnPolicyAgent`); each line after that is a
function below::

    A_t = GAE(gamma, lambda) of V; V_targ = A_t + V(s_t)         _compute_gae
    for each of n_epochs, for each minibatch of the rollout:      PPO.update
        r_t = pi(a_t | s_t) / pi_old(a_t | s_t)
        L_clip = -mean min(r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t)
                 - ent_coef mean H(pi(. | s_t))             policy_loss_function
        L_V = vf_coef mean (V(s_t) - V_targ)^2 / 2           value_loss_function
        one Adam step of each network down its own loss, the two
        gradients clipped together to max_grad_norm         PPO.minibatch_step

How the rollout is cut into minibatches (``choose_rollout_split``): ppo2 shuffles
single steps and computes GAE once, on the rollout; with ``unroll_length``
(brax's PPO) the minibatches are shuffled fragments of that many steps, each
in time order, and GAE is computed again in each minibatch from the current
critic; a recurrent policy (``memory``) trains on sequences of
``bptt_length`` steps in time order, each replayed from the memory it
started with, GAE once. Each network takes the gradient of its own loss,
which is that of ppo2's summed loss (their parameters are apart).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Optional, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict

from ajax.agents.diagnostics import approximate_kl, explained_variance, mean_policy_std
from ajax.agents.families import (
    ExpertGuided,
    ImageObservations,
    ObservationNormalization,
    OnPolicyAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.minibatch import (
    RolloutSplit,
    check_rollout_split,
    choose_rollout_split,
    get_actions_and_log_probs,
    run_epochs,
)
from ajax.agents.recurrent import rollout_sequence_values
from ajax.agents.steps import extension_key, joint_step
from ajax.environments.interaction import get_pi, get_pi_sequence
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.networks.distributions import policy_entropy, recompute_log_prob
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import predict_value, predict_value_sequence
from ajax.schedule import Schedule, at
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture, FloatOrCallable, InitializationFunction

#: The hidden layers of PPO's reference code: OpenAI Baselines' ppo2 @ea25b9e
#: builds its MLP with ortho_init(sqrt(2)) layers (baselines/common/models.py:96),
#: as CleanRL @fe8d8a0 (ppo.py:94) and PureJaxRL @31756b1 (ppo.py:25-42) do;
#: its output layers (orthogonal 0.01 for the policy, 1.0 for the value) are
#: the networks' defaults. ppo1, the paper's earlier code, used normc
#: instead; the owner chose ppo2 (2026-10-17). Recurrent PPO's references
#: start the layers before the memory the same way: PureJaxRL @31756b1
#: (ppo_rnn.py:53-55) and popjaxrl @12e5d42 (ppo_gru.py:45-47), whose GRU
#: keeps flax's defaults (ppo_rnn.py:35, ppo_gru.py:30), as Ajax's does.
PPO_HIDDEN_INIT = "orthogonal(np.sqrt(2))"


@dataclass(kw_only=True, eq=False, repr=False)
class PPO(
    ObservationNormalization,
    RewardNormalization,
    ExpertGuided,
    ImageObservations,
    OnPolicyAgent,
):
    """Proximal Policy Optimization, for discrete and continuous actions.

    Defaults are ppo2's (``ppo2/ppo2.py:22``, ``model.py:75``, ``:91``,
    ``:100-107``): Adam at 3e-4 with eps 1e-5, its gradient clipped to 0.5
    over both networks at once, ``vf_coef`` 0.5 on half the squared value
    error, clip 0.2, gamma 0.99, lambda 0.95, the advantages normalised per
    minibatch; the Gaussian's log-std one learned parameter per action
    dimension starting at 0 (std 1), never clipped (the Actor in
    networks.py cites it). The 128-128 tanh layers are Ajax's.
    """

    n_envs: int = 4
    actor_learning_rate: FloatOrCallable = 3e-4
    critic_learning_rate: FloatOrCallable = 3e-4
    actor_architecture: Architecture = ("128", "tanh", "128", "tanh")
    critic_architecture: Architecture = ("128", "tanh", "128", "tanh")
    gamma: float = 0.99
    #: ppo2's clip: one global norm over both networks' gradients
    #: (ppo2/model.py:105-107); None leaves them unclipped.
    max_grad_norm: Optional[float] = 0.5
    ent_coef: float = 0
    clip_range: Schedule = 0.2  # or a function of the timestep
    n_steps: int = 2048
    #: The samples of a minibatch, unless num_minibatches is set.
    batch_size: int = 64
    n_epochs: int = 10
    #: When > 0, the minibatches of each epoch (brax); at 0, Ajax's legacy
    #: max(batch_size, n_steps) // min(batch_size, n_steps).
    num_minibatches: int = 0
    adam_eps: float = 1e-5
    #: The critic's loss is vf_coef * 0.5 * MSE: ppo2's vf_coef 0.5 on its
    #: 0.5 * MSE (ppo2/ppo2.py:22, model.py:75, :91).
    vf_coef: float = 0.5
    #: Opts into brax's minibatches: the (n_steps, n_envs) rollout is cut
    #: into fragments of unroll_length steps, each in time order, shuffled
    #: into the minibatches, and GAE, recomputed in each minibatch,
    #: bootstraps at every fragment's end (brax's (T=unroll_length,
    #: B=batch_size) loss); None: GAE once on the rollout, its single steps
    #: shuffled.
    unroll_length: Optional[int] = None
    #: brax's num_resets_per_eval: every env reset with a fresh key this
    #: many times per evaluation, to cycle through new initial conditions;
    #: 0, only the episodes' own resets. Playground envs start every episode
    #: from a fresh state by default; this adds diversity only with
    #: build_env_from_id(..., fresh_reset=False).
    num_resets_per_eval: int = 0
    num_evals: int = 1
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    #: A memory block, MemoryConfig("gru", 64) or {"kind": "lstm",
    #: "hidden_size": 64}: PPO then trains with BPTT on sequences of
    #: bptt_length steps of the rollout (docs/guide/memory.md).
    memory: Optional[Union[MemoryConfig, dict]] = None
    #: A recurrent policy's sequence length; None derives it from the
    #: minibatches (minibatch.default_bptt_length): whole rollouts when
    #: n_envs is a multiple of them. An explicit one must divide that one.
    #: Each sequence starts from the memory recomputed with the current
    #: parameters, never a zero one mid-episode.
    bptt_length: Optional[int] = None
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    #: The hidden layers' initialiser, with or without memory; None is the
    #: reference's, PPO_HIDDEN_INIT.
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = PPO_HIDDEN_INIT
    #: False and log_std_init -1.0 build the state-dependent head PPO had
    #: before (SAC's), which log_std_bounds=LOG_STD_BOUNDS clips as then.
    log_std_state_independent: bool = True
    log_std_init: float = 0.0
    log_std_bounds: Optional[Tuple[float, float]] = None
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_layer_norm: bool = False
    #: Squashed or not, the action is read in [-1, 1] and mapped onto the
    #: env's bounds (agent_action_to_env), clipped there first.
    squash: bool = False

    def actor_critic_args(self) -> dict[str, Any]:
        if self.encoder_kernel_init is None:
            return {"encoder_kernel_init": PPO_HIDDEN_INIT}
        return {}

    def optimizer_settings(self) -> dict[str, Any]:
        # ppo2's Adam (model.py:100-107) with eps 1e-5; its one clip over both
        # networks is minibatch_step's (steps.joint_step), so neither
        # network's own optimiser clips.
        return {"eps": self.adam_eps, "max_grad_norm": None}

    def setup(self) -> None:
        # Checked here, not in check(): n_envs and the memory are known once built.
        check_rollout_split(
            self, self.env_args.n_envs, self.network_args.memory is not None
        )

    def init(self, key: jax.Array, ctx: TrainContext) -> PPOState:
        """The initial actor, critic and collector (holding a stateful
        expert's initial state)."""
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        env_args = ctx.env_args
        continuous = check_if_environment_has_continuous_actions(
            env_args.env, env_params=env_args.env_params
        )
        actor_state, critic_state = ctx.actor_critic(
            init_key,
            continuous=continuous,
            action_value=False,
            squash=ctx.network_args.squash,
            num_critics=1,
        )
        return PPOState(
            rng=rng,
            eval_rng=eval_rng,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=ctx.collector(collector_key),
            n_updates=0,
        )

    def update(
        self, state: PPOState, rollout: Transition, start: PPOState, ctx: TrainContext
    ) -> tuple[PPOState, AuxiliaryLogs]:
        """One PPO update on the ``(n_steps, n_envs)`` ``rollout`` collected
        from ``start``: ``n_epochs`` passes over its minibatches."""
        n_steps, n_envs = rollout.obs.shape[:2]
        # Cut the rollout into minibatches, in one of three ways the hyperparameters choose:
        # - shuffled single steps (ppo2): advantages computed once, on the whole rollout;
        # - shuffled pieces of unroll_length consecutive steps (brax): advantages
        #   recomputed in each minibatch, with the current critic;
        # - pieces of bptt_length consecutive steps (a recurrent policy), each replayed
        #   from the memory it started with: advantages computed once.
        split = choose_rollout_split(self, n_steps, n_envs, ctx.recurrent)
        actions, log_probs = get_actions_and_log_probs(rollout)
        extensions, total_timesteps = ctx.extensions, ctx.total_timesteps
        # The update's key: it shuffles the rollout, and the extensions' reward
        # and on_target draw on keys derived from it.
        state, shuffle_key = state.draw_keys()
        reward_key, target_key = jax.random.split(extension_key(shuffle_key))
        rewards = rollout.reward
        reward_metrics: dict[str, Any] = {}
        if "reward" in extensions.implemented_phases():
            # The extensions' rewards replace the env's (GAIL's discriminator).
            state, rewards, reward_metrics = extensions.apply_reward(
                state,
                {
                    "observations": rollout.obs,
                    "raw_observations": rollout.raw_obs,
                    "actions": actions,
                },
                rewards,
                state.collector_state.timestep,
                reward_key,
                total_timesteps,
            )
        batch = {
            "obs": rollout.obs,
            "actions": actions,
            "log_probs": log_probs,
            "raw_actions": rollout.raw_action,
            # The environment's own observations, which only an expert reads
            # (ImitationLoss, in actor_loss); jit drops them where none does.
            "raw_obs": rollout.raw_obs,
        }
        # What GAE reads of each step beside its observation, on the rollout
        # or in each minibatch.
        outcomes = {
            "next_obs": rollout.next_obs,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "rewards": rewards,
        }
        rollout_metrics: dict = {}  # the extensions' on the rollout's targets
        carries = None  # a recurrent split's sequences' start memories

        if split.advantages_per_minibatch:
            # GAE in each minibatch, with the critic of that step.
            batch |= outcomes
        else:
            # GAE once, on the rollout, with the critic that collected it.
            if ctx.recurrent:
                state, values, next_values, resets, carries = rollout_sequence_values(
                    state, rollout, start, split.length
                )
                batch["resets"] = resets
            else:
                values = predict_value(
                    critic_state=state.critic_state,
                    critic_params=state.critic_state.params,
                    x=rollout.obs,
                ).squeeze(0)
                next_values = predict_value(
                    critic_state=state.critic_state,
                    critic_params=state.critic_state.params,
                    x=rollout.next_obs,
                ).squeeze(0)
            gae, value_targets, rollout_metrics = self.advantages(
                state, batch | outcomes, values, next_values, target_key, ctx
            )
            batch |= {"targets": value_targets, "gae": gae}

        def minibatches(key: jax.Array) -> dict:
            """One epoch's partition of the rollout."""
            return split.minibatches(batch, key, carries)

        step = partial(self.minibatch_step, split=split, ctx=ctx)
        state, aux = run_epochs(state, shuffle_key, minibatches, step, self.n_epochs)
        # on_target ran once, over the rollout, unless the advantages are per minibatch.
        aux = aux.replace(
            extensions=merge_metrics(reward_metrics, rollout_metrics, aux.extensions)
        )
        state = state.replace(n_updates=state.n_updates + 1)
        return state, aux

    def advantages(
        self,
        state: PPOState,
        steps: dict,
        values: jax.Array,
        next_values: jax.Array,
        key: jax.Array,
        ctx: TrainContext,
        hold: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """GAE and the value targets ``A + V`` on ``steps`` (the rollout's or
        a minibatch's obs, next_obs, rewards, terminated and truncated),
        held constant with ``hold`` (brax's); the extensions' on_target
        reshapes the targets on ``key``. Returns them and its metrics."""
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=steps["rewards"],
            terminateds=steps["terminated"],
            truncateds=steps["truncated"],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if hold:
            gae = jax.lax.stop_gradient(gae)
            value_targets = jax.lax.stop_gradient(value_targets)
        if not ctx.extensions:
            return gae, value_targets, {}
        target_batch = {
            "observations": steps["obs"],
            "next_observations": steps["next_obs"],
            "rewards": steps["rewards"],
            "terminated": steps["terminated"],
            "truncated": steps["truncated"],
            "gae": gae,
            "values": values,
            "next_values": next_values,
            "gamma": self.gamma,
        }
        timestep, horizon = state.collector_state.timestep, ctx.total_timesteps
        value_targets, metrics = ctx.extensions.apply_on_target(
            state, target_batch, value_targets, timestep, key, horizon
        )
        return gae, value_targets, metrics

    def minibatch_step(
        self, state: PPOState, minibatch: dict, split: RolloutSplit, ctx: TrainContext
    ) -> tuple[PPOState, AuxiliaryLogs]:
        """A critic and an actor step, both from the state before either."""
        state, ent_rng = state.draw_keys()
        target_metrics: dict = {}
        critic, actor = state.critic_state, state.actor_state
        extensions, total_timesteps = ctx.extensions, ctx.total_timesteps
        # The extensions' target and loss terms, each on a key of its own.
        target_key, critic_key, actor_key = jax.random.split(extension_key(ent_rng), 3)
        critic_extra = _stack_critic_loss(
            extensions, state, critic_key, total_timesteps
        )
        actor_extra = _stack_actor_loss(
            extensions, state, actor_key, total_timesteps, minibatch["raw_obs"]
        )

        if split.advantages_per_minibatch:
            # brax's compute_ppo_loss: GAE from the current critic, held
            # constant (no gradient reaches the critic through it).
            values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic, critic_params=critic.params, x=minibatch["obs"]
                ).squeeze(0)
            )
            next_values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic,
                    critic_params=critic.params,
                    x=minibatch["next_obs"],
                ).squeeze(0)
            )
            gae, value_targets, target_metrics = self.advantages(
                state, minibatch, values, next_values, target_key, ctx, hold=True
            )
        else:
            gae, value_targets = minibatch["gae"], minibatch["targets"]

        # Only a recurrent split's minibatches carry resets and start memories.
        resets = minibatch["resets"] if ctx.recurrent else None
        (_, (v_aux, v_metrics)), v_grads = jax.value_and_grad(
            value_loss_function, has_aux=True
        )(
            critic.params,
            critic,
            minibatch["obs"],
            value_targets,
            ctx.recurrent,
            vf_coef=self.vf_coef,
            extra_loss_fn=critic_extra,
            resets=resets,
            initial_hidden=minibatch["critic_hidden"] if ctx.recurrent else None,
        )
        (_, (p_aux, p_metrics)), p_grads = jax.value_and_grad(
            policy_loss_function, has_aux=True
        )(
            actor.params,
            actor,
            minibatch["obs"],
            minibatch["actions"],
            minibatch["log_probs"],
            gae,
            ctx.recurrent,
            at(self.clip_range, state.collector_state.timestep),
            self.ent_coef,
            self.normalize_advantage,
            extra_loss_fn=actor_extra,
            raw_actions=minibatch["raw_actions"],
            entropy_rng=ent_rng,
            resets=resets,
            initial_hidden=minibatch["actor_hidden"] if ctx.recurrent else None,
        )
        actor, critic = joint_step(actor, critic, p_grads, v_grads, self.max_grad_norm)
        state = state.replace(critic_state=critic, actor_state=actor)
        # Each network's norm before the joint clip.
        p_aux = p_aux.replace(grad_norm=optax.global_norm(p_grads))
        v_aux = v_aux.replace(grad_norm=optax.global_norm(v_grads))
        metrics = merge_metrics(target_metrics, v_metrics, p_metrics)
        return state, AuxiliaryLogs(policy=p_aux, value=v_aux, extensions=metrics)

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "PPO"
    action_space = "any"
    capabilities = frozenset(
        "actor discounted obs_networks policy_density v_critic reward_learning".split()
    )
    supported_extension_phases = LOOP_PHASES | frozenset(
        "reward on_target critic_loss actor_loss transform_action actuate".split()
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "reward": frozenset("observations raw_observations actions".split()),
        "on_target": frozenset("terminated truncated gae values".split()),
        "critic_loss": frozenset("critic_params critic_state".split()),
        "actor_loss": frozenset(
            "actor_state observations raw_observations pi_mean".split()
        ),
    }
    supports_memory = True


# -- PPO's equations, in the order the update calls them ---------------------
@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE (Schulman et al., 2016) over a time-major rollout, in reverse:
    ``delta_t = r_t + gamma V(s_{t+1}) (1 - terminated_t) - V(s_t)`` and
    ``A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}``: a truncation cuts
    the trace but still bootstraps. Returns ``(A, A + V)``."""

    def _get_advantages(
        gae: jax.Array,
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        done = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * done * gae

        return gae, gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=jnp.zeros_like(values[-1]),
        xs=(rewards, values, next_values, terminateds, truncateds),
        reverse=True,
        unroll=1,
    )
    returns = advantages + values
    return advantages, returns


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    extra_loss_fn: Optional[Callable] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[ValueAuxiliaries, dict]]:
    """``vf_coef * 0.5 * MSE(V(obs), targets)`` plus ``extra_loss_fn``'s
    term, with the metrics and ``extra_loss_fn``'s (``{}`` without one). A
    recurrent critic replays the sequence from ``initial_hidden``,
    ``resets`` flagging the observations that start an episode."""
    if recurrent:
        assert resets is not None and initial_hidden is not None
        v_preds, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        v_preds = predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
        )
    v_preds = v_preds.squeeze(0)  # the single critic keeps the ensemble axis

    loss = vf_coef * 0.5 * jnp.mean((v_preds - value_targets) ** 2)
    metrics: dict = {}
    if extra_loss_fn is not None:
        extra, metrics = extra_loss_fn(
            critic_params, critic_states, observations, value_targets
        )
        loss = loss + extra

    aux = ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
        explained_variance=explained_variance(v_preds, value_targets),
    )
    return loss, (aux, metrics)


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, dict]]:
    """The clipped surrogate minus ``ent_coef`` times the entropy bonus,
    plus ``extra_loss_fn``'s term; with ``advantage_normalization`` the
    advantages are normalised per minibatch. A recurrent actor replays the
    sequence from ``initial_hidden`` (the live carry on ``actor_state`` has
    advanced past this rollout), ``resets`` flagging episode starts."""
    if recurrent:
        assert resets is not None
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        pi, _ = get_pi(actor_state, actor_params, observations)

    new_log_probs = recompute_log_prob(pi, actions, raw_actions)
    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    surrogate, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    loss_actor = surrogate.mean()
    entropy = policy_entropy(pi, entropy_rng).mean()

    total_loss = loss_actor - ent_coef * entropy
    metrics: dict = {}
    if extra_loss_fn is not None:
        # A discrete policy has no mean action.
        pi_mean = None if isinstance(pi, distrax.Categorical) else pi.mean()
        extra, metrics = extra_loss_fn(actor_params, actor_state, observations, pi_mean)
        total_loss = total_loss + extra

    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
        approx_kl=approximate_kl(new_log_probs - log_probs),
        policy_std=mean_policy_std(pi),
    )
    return total_loss, (aux, metrics)


def clipped_surrogate(
    ratio: jax.Array, advantages: jax.Array, clip_coef: float
) -> tuple[jax.Array, jax.Array]:
    """The per-sample clipped surrogate loss ``-min(r A, clip(r) A)`` and
    the fraction of samples whose ratio is clipped."""
    loss = -jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages,
    )
    return loss, (jnp.abs(ratio - 1) > clip_coef).mean()


# -- The extensions' loss terms, as the losses' extra_loss_fn ----------------
def _stack_critic_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
) -> Optional[Callable]:
    """The extensions' additive critic-loss term on ``key`` as a
    ``value_loss_function`` ``extra_loss_fn``: ``(critic_params,
    critic_states, observations, value_targets) -> (term, metrics)``;
    ``None`` without extensions. The extensions read their state from
    ``agent_state``.
    """
    if not extension_stack:
        return None

    def critic_loss(critic_params, critic_states, observations, value_targets):
        _batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": critic_params,
            "critic_state": critic_states,
        }
        return extension_stack.apply_critic_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return critic_loss


def _stack_actor_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
    raw_observations: Optional[jax.Array] = None,
) -> Optional[Callable]:
    """The extensions' additive actor-loss term on ``key`` as a
    ``policy_loss_function`` ``extra_loss_fn``: ``(actor_params,
    actor_state, observations, pi_mean) -> (term, metrics)``, the minibatch's
    ``raw_observations`` beside them; ``None`` without extensions."""
    if not extension_stack:
        return None

    def actor_loss(actor_params, actor_state, observations, pi_mean):
        _batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        return extension_stack.apply_actor_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return actor_loss


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass
class PPOState(BaseAgentState):
    """PPO's state: the base state's fields."""

    #: Its networks' gradients are clipped together (steps.joint_step): a
    #: checkpoint from when each clipped its own restores (ajax.checkpoint).
    #: A ClassVar: a flag on the class, shared by every PPO state, not stored in its arrays.
    joint_gradient_clip: ClassVar[bool] = True


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_probs: jax.Array
    old_log_probs: jax.Array
    clip_fraction: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    policy_std: jax.Array
    # The loss builds these metrics before the gradient exists; the update
    # fills it in after the step.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    predictions: jax.Array
    targets: jax.Array
    explained_variance: jax.Array
    grad_norm: jax.Array | float = jnp.nan  # filled in after the step, as the policy's


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


#: ``agent.agent_config``'s class: a view of the fields until 0.5.0.
PPOConfig = PPO.declare_config(
    "gamma ent_coef clip_range n_steps batch_size n_epochs gae_lambda"
    " normalize_advantage num_minibatches vf_coef max_grad_norm unroll_length"
    " bptt_length num_resets_per_eval num_evals expose_recent_rollout"
)
