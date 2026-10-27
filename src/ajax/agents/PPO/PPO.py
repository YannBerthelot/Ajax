"""Proximal Policy Optimization (Schulman, Wolski, Dhariwal, Radford and
Klimov, 2017), as its reference code, OpenAI Baselines' ppo2 (@ea25b9e),
runs it.

One iteration of the paper's Algorithm 1. The on-policy family runs the
policy for n_steps in each of n_envs envs
(OnPolicyAgent); each line after that is a
function below::

    A_t = GAE(gamma, lambda) of V; V_targ = A_t + V(s_t)         _compute_gae
    for each of n_epochs, for each minibatch of the rollout:      PPO.update
        r_t = pi(a_t | s_t) / pi_old(a_t | s_t)
        L_clip = -mean min(r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t)
                 - ent_coef mean H(pi(. | s_t))              compute_policy_loss
        L_V = vf_coef mean (V(s_t) - V_targ)^2 / 2            compute_value_loss
        one Adam step of each network down its own loss, the two
        gradients clipped together to max_grad_norm         PPO.minibatch_step

How the rollout is cut into minibatches (choose_rollout_split): ppo2 shuffles
single steps and computes GAE once, on the rollout; with unroll_length
(brax's PPO) the minibatches are shuffled fragments of that many steps, each
in time order, and GAE is computed again in each minibatch from the current
critic; a recurrent policy (memory) trains on sequences of
bptt_length steps in time order, each replayed from the memory it
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
from ajax.agents.recurrent import (
    compute_rollout_values,
    get_policy_on_minibatch,
    predict_values_on_minibatch,
)
from ajax.agents.update_steps import (
    apply_gradients_clipped_together,
    compute_gradients,
    derive_extensions_key,
)
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.networks.distributions import policy_entropy, recompute_log_prob
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import predict_value
from ajax.random_keys import derive_key, split_by_index
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

    Defaults are ppo2's (ppo2/ppo2.py:22, model.py:75, :91,
    :100-107): Adam at 3e-4 with eps 1e-5, its gradient clipped to 0.5
    over both networks at once, vf_coef 0.5 on half the squared value
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
        # networks is minibatch_step's (update_steps.apply_gradients_clipped_together),
        # so neither network's own optimiser clips.
        return {"eps": self.adam_eps, "max_grad_norm": None}

    def setup(self) -> None:
        # Checked here, not in check(): n_envs and the memory are known once built.
        check_rollout_split(
            self, self.env_args.n_envs, self.network_args.memory is not None
        )

    def init(self, key: jax.Array, context: TrainContext) -> PPOState:
        """The initial actor, critic and collector (holding a stateful
        expert's initial state)."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        env_args = context.env_args
        continuous = check_if_environment_has_continuous_actions(
            env_args.env, env_params=env_args.env_params
        )
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=continuous,
            action_value=False,
            squash=context.network_args.squash,
            num_critics=1,
        )
        return PPOState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=context.make_collector(collector_key),
            n_updates=0,
        )

    def update(
        self,
        agent_state: PPOState,
        rollout: Transition,
        agent_state_at_rollout_start: PPOState,
        context: TrainContext,
    ) -> tuple[PPOState, AuxiliaryLogs]:
        """One PPO update on the (n_steps, n_envs) rollout collected from
        agent_state_at_rollout_start: n_epochs passes over its minibatches."""
        n_steps, n_envs = rollout.obs.shape[:2]
        # Cut the rollout into minibatches, in one of three ways the hyperparameters choose:
        # - shuffled single steps (ppo2): advantages computed once, on the whole rollout;
        # - shuffled pieces of unroll_length consecutive steps (brax): advantages
        #   recomputed in each minibatch, with the current critic;
        # - pieces of bptt_length consecutive steps (a recurrent policy), each replayed
        #   from the memory it started with: advantages computed once.
        split = choose_rollout_split(self, n_steps, n_envs, context.recurrent)
        actions, log_probs = get_actions_and_log_probs(rollout)
        extensions, total_timesteps = context.extensions, context.total_timesteps
        # The update's key: it shuffles the rollout, and the extensions' reward
        # and on_target draw on keys derived from it.
        shuffle_key, agent_state = agent_state.get_keys_and_new_state()
        reward_key, target_key = jax.random.split(derive_extensions_key(shuffle_key))
        rewards = rollout.reward
        reward_metrics: dict[str, Any] = {}
        if "reward" in extensions.implemented_phases():
            # The extensions' rewards replace the env's (GAIL's discriminator).
            agent_state, rewards, reward_metrics = extensions.apply_reward(
                agent_state,
                {
                    "observations": rollout.obs,
                    "raw_observations": rollout.raw_obs,
                    "actions": actions,
                },
                rewards,
                agent_state.collector_state.timestep,
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
        # What a recurrent split's sequences replay their networks from; None
        # without memory.
        memories = None

        if split.advantages_per_minibatch:
            # brax's PPO computes GAE inside its loss, on each minibatch, with the
            # critic as it is at that step (brax/training/agents/ppo/losses.py,
            # compute_ppo_loss), so each minibatch carries what GAE reads.
            batch.update(outcomes)
        else:
            # GAE once, on the rollout, with the critic that collected it.
            agent_state, values, next_values, memories = compute_rollout_values(
                agent_state, rollout, agent_state_at_rollout_start, split.length
            )
            batch_with_outcomes = dict(batch)
            batch_with_outcomes.update(outcomes)
            gae, value_targets, rollout_metrics = self.compute_advantages(
                agent_state,
                batch_with_outcomes,
                values,
                next_values,
                target_key,
                context,
            )
            batch.update({"targets": value_targets, "gae": gae})

        def cut_rollout_into_minibatches(epoch_key: jax.Array) -> dict:
            """One epoch's partition of the rollout, each minibatch with a key
            for its entropy sample and its extensions, derived from the
            epoch's: the number of minibatches moves no other draw."""
            minibatches = split.cut_into_minibatches(batch, epoch_key, memories)
            minibatch_keys = derive_key(epoch_key, "minibatch_entropy")
            entropy_keys = split_by_index(minibatch_keys, split.num_minibatches)
            minibatches.update({"entropy_key": entropy_keys})
            return minibatches

        step_on_minibatch = partial(self.minibatch_step, split=split, context=context)
        agent_state, aux = run_epochs(
            agent_state,
            shuffle_key,
            cut_rollout_into_minibatches,
            step_on_minibatch,
            self.n_epochs,
        )
        # on_target ran once, over the rollout, unless the advantages are per minibatch.
        aux = aux.replace(
            extensions=merge_metrics(reward_metrics, rollout_metrics, aux.extensions)
        )
        agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
        return agent_state, aux

    def compute_advantages(
        self,
        agent_state: PPOState,
        steps: dict,
        values: jax.Array,
        next_values: jax.Array,
        key: jax.Array,
        context: TrainContext,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """GAE and the value targets A + V on steps (the rollout's, or a
        minibatch's: obs, next_obs, rewards, terminated and truncated); the
        extensions' on_target reshapes the targets, drawing on key. Returns
        the advantages, the targets and the extensions' metrics.

        They enter the losses as inputs, computed before the losses are
        differentiated, so no gradient flows back through them (brax computes
        GAE inside its loss and stops the gradient there instead)."""
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=steps["rewards"],
            terminateds=steps["terminated"],
            truncateds=steps["truncated"],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if not context.extensions:
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
        timestep, horizon = (
            agent_state.collector_state.timestep,
            context.total_timesteps,
        )
        value_targets, metrics = context.extensions.apply_on_target(
            agent_state, target_batch, value_targets, timestep, key, horizon
        )
        return gae, value_targets, metrics

    def minibatch_step(
        self,
        agent_state: PPOState,
        minibatch: dict,
        split: RolloutSplit,
        context: TrainContext,
    ) -> tuple[PPOState, AuxiliaryLogs]:
        """One gradient step of the critic and one of the actor on the
        minibatch. Both gradients are taken at the parameters this step starts
        from, then clipped together and applied. That is ppo2's single step
        down the sum of the two losses: the networks share no parameter, so
        each network's gradient of the sum is the gradient of its own loss."""
        # The minibatch's key: a squashed policy's entropy is sampled with it,
        # and the extensions' keys below are derived from it.
        entropy_key = minibatch["entropy_key"]
        target_metrics: dict = {}
        critic, actor = agent_state.critic_state, agent_state.actor_state
        extensions, total_timesteps = context.extensions, context.total_timesteps
        # The extensions' target and loss terms, each on a key of its own.
        target_key, critic_key, actor_key = jax.random.split(
            derive_extensions_key(entropy_key), 3
        )
        compute_extensions_critic_loss = _make_extensions_critic_loss(
            extensions, agent_state, critic_key, total_timesteps
        )
        compute_extensions_actor_loss = _make_extensions_actor_loss(
            extensions, agent_state, actor_key, total_timesteps, minibatch["raw_obs"]
        )

        if split.advantages_per_minibatch:
            # brax's PPO: GAE again, on this minibatch, with the current critic.
            # Only a feedforward critic gets here (check_rollout_split refuses
            # unroll_length with a memory), so it needs no stored memory.
            values = predict_value(
                critic_state=critic, critic_params=critic.params, x=minibatch["obs"]
            ).squeeze(0)
            next_values = predict_value(
                critic_state=critic,
                critic_params=critic.params,
                x=minibatch["next_obs"],
            ).squeeze(0)
            gae, value_targets, target_metrics = self.compute_advantages(
                agent_state, minibatch, values, next_values, target_key, context
            )
        else:
            gae, value_targets = minibatch["gae"], minibatch["targets"]

        def value_loss(critic_params: FrozenDict) -> tuple:
            return compute_value_loss(
                critic_params,
                critic,
                minibatch,
                value_targets,
                vf_coef=self.vf_coef,
                compute_extensions_loss=compute_extensions_critic_loss,
            )

        def policy_loss(actor_params: FrozenDict) -> tuple:
            return compute_policy_loss(
                actor_params,
                actor,
                minibatch,
                gae,
                clip_coef=at(self.clip_range, agent_state.collector_state.timestep),
                ent_coef=self.ent_coef,
                normalize_advantage=self.normalize_advantage,
                compute_extensions_loss=compute_extensions_actor_loss,
                entropy_key=entropy_key,
            )

        value_grads, (value_aux, value_metrics) = compute_gradients(
            value_loss, critic.params
        )
        policy_grads, (policy_aux, policy_metrics) = compute_gradients(
            policy_loss, actor.params
        )
        actor, critic, policy_grad_norm, value_grad_norm = (
            apply_gradients_clipped_together(
                actor, critic, policy_grads, value_grads, self.max_grad_norm
            )
        )
        agent_state = agent_state.replace(critic_state=critic, actor_state=actor)
        metrics = merge_metrics(target_metrics, value_metrics, policy_metrics)
        # Each network's gradient norm is the one before the joint clip.
        return agent_state, AuxiliaryLogs(
            policy=policy_aux.replace(grad_norm=policy_grad_norm),
            value=value_aux.replace(grad_norm=value_grad_norm),
            extensions=metrics,
        )

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "PPO"
    action_space = "any"
    capabilities = frozenset(
        {
            "actor",
            "discounted",
            "obs_networks",
            "policy_density",
            "v_critic",
            "reward_learning",
        }
    )
    supported_extension_phases = LOOP_PHASES.union(
        {
            "reward",
            "on_target",
            "critic_loss",
            "actor_loss",
            "transform_action",
            "actuate",
        }
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "reward": frozenset({"observations", "raw_observations", "actions"}),
        "on_target": frozenset({"terminated", "truncated", "gae", "values"}),
        "critic_loss": frozenset({"critic_params", "critic_state"}),
        "actor_loss": frozenset(
            {"actor_state", "observations", "raw_observations", "pi_mean"}
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
    delta_t = r_t + gamma V(s_{t+1}) (1 - terminated_t) - V(s_t) and
    A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}: a truncation cuts
    the trace but still bootstraps. Returns (A, A + V)."""

    def _get_advantages(
        gae: jax.Array,
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        episode_continues = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * episode_continues * gae

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


def compute_value_loss(
    critic_params: FrozenDict,
    critic_state: LoadedTrainState,
    minibatch: dict,
    value_targets: jax.Array,
    vf_coef: float = 1.0,
    compute_extensions_loss: Optional[Callable] = None,
) -> Tuple[jax.Array, Tuple[ValueAuxiliaries, dict]]:
    """vf_coef * 0.5 * MSE(V(s), value_targets) over the minibatch's
    observations (obs), plus the extensions' term (compute_extensions_loss),
    with the metrics and the extensions' ({} without them). A critic with
    memory replays the minibatch's sequences (predict_values_on_minibatch)."""
    value_preds = predict_values_on_minibatch(critic_state, critic_params, minibatch)
    loss = vf_coef * 0.5 * jnp.mean((value_preds - value_targets) ** 2)
    metrics: dict = {}
    if compute_extensions_loss is not None:
        extensions_loss, metrics = compute_extensions_loss(
            critic_params, critic_state, minibatch["obs"], value_targets
        )
        loss = loss + extensions_loss

    aux = ValueAuxiliaries(
        critic_loss=loss,
        predictions=value_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
        explained_variance=explained_variance(value_preds, value_targets),
    )
    return loss, (aux, metrics)


def compute_policy_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    minibatch: dict,
    gae: jax.Array,
    clip_coef: float,
    ent_coef: float,
    normalize_advantage: bool,
    compute_extensions_loss: Optional[Callable] = None,
    entropy_key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, dict]]:
    """The clipped surrogate minus ent_coef times the entropy bonus, plus the
    extensions' term (compute_extensions_loss); with normalize_advantage the
    advantages are normalised per minibatch. The minibatch holds the
    observations (obs), the actions taken, their log-probabilities when
    collected, and a squashed policy's actions before the squash
    (raw_actions). An actor with memory replays the minibatch's sequences
    (get_policy_on_minibatch)."""
    policy = get_policy_on_minibatch(actor_state, actor_params, minibatch)
    actions, log_probs = minibatch["actions"], minibatch["log_probs"]
    new_log_probs = recompute_log_prob(policy, actions, minibatch["raw_actions"])
    ratio = jnp.exp(new_log_probs - log_probs)

    if normalize_advantage:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    surrogate, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    loss_actor = surrogate.mean()
    entropy = policy_entropy(policy, entropy_key).mean()

    total_loss = loss_actor - ent_coef * entropy
    metrics: dict = {}
    if compute_extensions_loss is not None:
        # A discrete policy has no mean action.
        pi_mean = None if isinstance(policy, distrax.Categorical) else policy.mean()
        extensions_loss, metrics = compute_extensions_loss(
            actor_params, actor_state, minibatch["obs"], pi_mean
        )
        total_loss = total_loss + extensions_loss

    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
        approx_kl=approximate_kl(new_log_probs - log_probs),
        policy_std=mean_policy_std(policy),
    )
    return total_loss, (aux, metrics)


def clipped_surrogate(
    ratio: jax.Array, advantages: jax.Array, clip_coef: float
) -> tuple[jax.Array, jax.Array]:
    """The per-sample clipped surrogate loss -min(r A, clip(r) A) and
    the fraction of samples whose ratio is clipped."""
    loss = -jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages,
    )
    return loss, (jnp.abs(ratio - 1) > clip_coef).mean()


# -- The extensions' loss terms, as the losses' compute_extensions_loss -------
def _make_extensions_critic_loss(
    extensions: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
) -> Optional[Callable]:
    """The extensions' additive critic-loss term on key, as
    compute_value_loss's compute_extensions_loss: (critic_params,
    critic_state, observations, value_targets) -> (term, metrics);
    None without extensions. The extensions read their state from
    agent_state.
    """
    if not extensions:
        return None

    def compute_extensions_critic_loss(
        critic_params, critic_state, observations, value_targets
    ):
        critic_batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": critic_params,
            "critic_state": critic_state,
        }
        return extensions.compute_extensions_critic_loss(
            agent_state,
            critic_batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return compute_extensions_critic_loss


def _make_extensions_actor_loss(
    extensions: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
    raw_observations: Optional[jax.Array] = None,
) -> Optional[Callable]:
    """The extensions' additive actor-loss term on key, as
    compute_policy_loss's compute_extensions_loss: (actor_params,
    actor_state, observations, pi_mean) -> (term, metrics), the minibatch's
    raw_observations beside them; None without extensions."""
    if not extensions:
        return None

    def compute_extensions_actor_loss(actor_params, actor_state, observations, pi_mean):
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        return extensions.compute_extensions_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return compute_extensions_actor_loss


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass
class PPOState(BaseAgentState):
    """PPO's state: the base state's fields."""

    #: Its networks' gradients are clipped together
    #: (update_steps.apply_gradients_clipped_together): a checkpoint from when each
    #: clipped its own restores (ajax.checkpoint).
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
