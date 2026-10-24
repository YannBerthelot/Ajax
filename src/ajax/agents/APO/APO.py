"""APO (Ma et al., 2021, https://arxiv.org/pdf/2106.03442): PPO for the
average-reward criterion, as its reference code, xtma/apo @bfcd997 (on the
authors' rlpyt fork, xtma/rlpyt @e28bb20), runs it.

One update on the (n_steps, n_envs) rollout the on-policy family
collects (OnPolicyAgent); each line is a
function below, or PPO's::

    rho <- (1 - alpha) rho + alpha mean r_t                        APO.update
    b <- (1 - alpha) b + alpha mean V(s_t)  (the first update sets both
         to the rollout's means)
    A_t = GAE(gamma = 1, lambda) of the differential TD error
          r_t - rho + V(s_{t+1}) - V(s_t), cut at every episode end
                                                            PPO's _compute_gae
    for each of n_epochs, for each minibatch of shuffled single steps:
        L = clipped surrogate - ent_coef H(pi)          policy_loss_function
            + vf_coef 0.5 (V(s_t) - (V_targ - nu b))^2   value_loss_function
        one step of both networks down L, its gradient clipped over both
        at once (torch's clip_grad_norm_)               APO.minibatch_step

A row that hit a time limit leaves both losses (valid_mean).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Optional, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.diagnostics import approximate_kl, explained_variance, mean_policy_std
from ajax.agents.families import (
    ExpertGuided,
    ObservationNormalization,
    OnPolicyAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.minibatch import (
    get_actions_and_log_probs,
    get_minibatches_from_batch,
    resolve_num_minibatches,
    run_epochs,
)
from ajax.agents.PPO.PPO import _compute_gae, clipped_surrogate
from ajax.agents.steps import (
    apply_gradients_clipped_together,
    compute_gradients,
    extension_key,
)
from ajax.environments.interaction import get_pi
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import merge_metrics
from ajax.networks.distributions import policy_entropy, recompute_log_prob
from ajax.networks.networks import predict_value
from ajax.networks.utils import TORCH_LINEAR
from ajax.schedule import Schedule, at
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture, FloatOrCallable, InitializationFunction


@dataclass(kw_only=True, eq=False, repr=False)
class APO(ObservationNormalization, RewardNormalization, ExpertGuided, OnPolicyAgent):
    """APO for discrete and continuous actions.

    Defaults are its reference code's, each cited where it is declared. An
    expert feeds the CloningPretrain extension and the evaluation's
    expert-bias metric; online cloning is the ImitationLoss extension, which
    reads the expert's actions that collection stores at each row.
    """

    n_envs: int = 4
    # APO's official code steps the whole model with one Adam at 3e-4,
    # unscheduled, torch's other defaults (xtma/apo @bfcd997,
    # apo/algos/apg/base.py:28; apo/experiments/configs/mujoco/apg/
    # mujoco_appo.py:5, :16). Adam acts per coordinate, so the two
    # networks' Adams, stepped together on one gradient
    # (APO.minibatch_step), are that Adam, each at its rate.
    actor_learning_rate: FloatOrCallable = 3e-4
    critic_learning_rate: FloatOrCallable = 3e-4
    # The hidden layers of APO's official code: xtma/apo @bfcd997 sets
    # only normalize_observation in the model config
    # (apo/experiments/configs/mujoco/apg/mujoco_appo.py:20, exp.py:38),
    # so rlpyt's MujocoFfModel (xtma/rlpyt @e28bb20,
    # rlpyt/models/pg/mujoco_ff_model.py:21-22, :33-49) builds its mean
    # and its value MLPs each with two hidden layers of 64 tanh units,
    # torch Linear layers (rlpyt/models/mlp.py:27-31) at torch's default
    # initialisation: "torch_linear" (TorchLinear, networks/utils.py).
    # The output layers are not the reference's: the mean layer's
    # initialiser is in actor_critic_args, the value layer's
    # orthogonal(1.0), both biases zero.
    actor_architecture: Architecture = ("64", "tanh", "64", "tanh")
    critic_architecture: Architecture = ("64", "tanh", "64", "tanh")
    # The official code's clip of the gradient's global norm over the
    # whole model, actor and critic at once (apo/algos/apg/appo.py:111,
    # mujoco_appo.py:8): neither network's Adam clips its own. None, no
    # clip.
    max_grad_norm: Optional[float] = 10.0
    # The entropy bonus's weight, the official code's entropy_loss_coeff
    # (mujoco_appo.py:9): the loss subtracts it times the batch mean of
    # the policy's entropy (appo.py:155-158), the Gaussian's summed over
    # the action dimensions (rlpyt/distributions/gaussian.py:81-103).
    ent_coef: float = 0.01
    # The value loss's weight in the summed loss, the official code's
    # value_loss_coeff (appo.py:27, which mujoco_appo.py leaves).
    vf_coef: float = 1.0
    clip_range: Schedule = 0.2  # or a function of the timestep
    n_steps: int = 2048
    #: The samples of a minibatch, unless num_minibatches is set.
    batch_size: int = 64
    n_epochs: int = 10
    #: When > 0, the minibatches of each epoch (brax); at 0, Ajax's legacy
    #: max(batch_size, n_steps) // min(batch_size, n_steps).
    num_minibatches: int = 0
    adam_eps: float = 1e-8
    gae_lambda: float = 0.95
    #: The rate of the reward rate's and the value bias's moving averages.
    alpha: float = 0.1
    #: The value-bias penalty's weight (value_loss_function).
    nu: float = 0.1
    normalize_advantage: bool = True
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = TORCH_LINEAR
    # The (low, high) clip of the continuous actor's log-std. None, no
    # clip, as APO's official code (actor_critic_args).
    log_std_bounds: Optional[Tuple[float, float]] = None

    def check(self) -> None:
        n_steps, batch_size = self.n_steps, self.batch_size
        if self.num_minibatches <= 0 and max(n_steps, batch_size) % min(
            n_steps, batch_size
        ):
            raise ValueError(
                f"n_steps ({n_steps}) and batch_size ({batch_size}) must divide"
                " one another: the rollout splits into max // min minibatches"
                " (or set num_minibatches)."
            )

    def actor_critic_args(self) -> dict[str, Any]:
        # The continuous head of APO's official code: xtma/apo @bfcd997
        # (exp.py:17, :38) runs rlpyt's MujocoFfAgent, from the authors'
        # fork xtma/rlpyt @e28bb20. The Gaussian's mean is tanh of the
        # mean layer (rlpyt/models/pg/mujoco_ff_model.py:23, :40-41);
        # one learned log-std per action dimension starts at 0 and is
        # not clipped (:24, :50, :78; rlpyt/agents/pg/gaussian.py:34-38
        # sets no min_std, max_std, clip or squash). So the sample is
        # not squashed: its log-probability and the entropy are the
        # Gaussian's, of the raw sample (rlpyt/distributions/
        # gaussian.py:81-97, :108-134, :208). The reference hands that
        # sample to the env (rlpyt/envs/gym.py:66-67); here, as for
        # every agent, agent_action_to_env clips it to [-1, 1] on its
        # way there. Discrete APO ignores the head. Not the reference's:
        # the lecun_uniform mean layer (torch's default Linear init
        # there), whose moderate initial actions keep evaluation close
        # to training (orthogonal 0.01 makes the deterministic action
        # essentially zero). As the reference's MlpModel
        # (rlpyt/models/mlp.py:27-35), the encoder has no output
        # LayerNorm (the default).
        return {
            "squash": False,
            "tanh_mean": True,
            "log_std_state_independent": True,
            "log_std_init": 0.0,
            "mean_kernel_init": "lecun_uniform",
        }

    def optimizer_settings(self) -> dict[str, Any]:
        # Adam's eps: 1e-8, torch's default the official code keeps, not
        # ActorCritic's 1e-5. Its one clip over both networks is
        # minibatch_step's (steps.apply_gradients_clipped_together), so
        # neither network's own optimiser clips.
        return {"eps": self.adam_eps, "max_grad_norm": None}

    def evaluation_options(self) -> dict[str, Any]:
        return {"avg_reward_mode": True}  # each evaluation measures rho too

    def init(self, key: jax.Array, context: TrainContext) -> APOState:
        """PPO's initial actor, critic and collector, on PPO's key layout,
        and the reward rate and the value bias at 0."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        env_args = context.env_args
        continuous = check_if_environment_has_continuous_actions(
            env_args.env, env_params=env_args.env_params
        )
        actor_state, critic_state = context.actor_critic(
            init_key,
            continuous=continuous,
            action_value=False,
            squash=context.network_args.squash,
            num_critics=1,
        )
        return APOState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=context.collector(collector_key),
            n_updates=0,
            average_reward=0.0,
            b=0.0,
        )

    def update(
        self,
        agent_state: APOState,
        rollout: Transition,
        agent_state_at_rollout_start: APOState,
        context: TrainContext,
    ) -> tuple[APOState, AuxiliaryLogs]:
        """One APO update (the module docstring); with an expert, the
        rollout's a_expert reaches the actor-loss extensions.

        The GAE is PPO's with gamma = 1 on r - rho, as in APO's
        official code (xtma/apo, apo/algos/utils.py,
        generalized_advantage_estimation with discount = 1, and
        apo/algos/apg/base.py, process_returns): V(s') is dropped
        at every episode end, a time limit included (the reference's
        done, gym's, includes them), and the lambda-carry is cut there; a
        time-limited row then leaves the loss (bootstrap_timelimit,
        base.py:73-77).
        """
        del (
            agent_state_at_rollout_start
        )  # a feedforward policy: nothing to replay from it
        extensions, total_timesteps = context.extensions, context.total_timesteps
        critic = agent_state.critic_state
        values = predict_value(critic, critic.params, rollout.obs).squeeze(0)
        next_values = predict_value(critic, critic.params, rollout.next_obs).squeeze(0)

        average_reward = (
            1 - self.alpha
        ) * agent_state.average_reward + self.alpha * jnp.mean(rollout.reward)

        value_bias = (1 - self.alpha) * agent_state.b + self.alpha * jnp.mean(values)
        # The first update sets both to the rollout's means, as xtma/apo does
        # (apo/algos/apg/base.py:43-50, eta and value_bias start as None): an
        # EMA from 0 would leave rho a fraction 0.9^k too small after k updates.
        first = agent_state.n_updates == 0
        average_reward = jnp.where(first, jnp.mean(rollout.reward), average_reward)
        value_bias = jnp.where(first, jnp.mean(values), value_bias)
        agent_state = agent_state.replace(average_reward=average_reward, b=value_bias)

        ends = jnp.maximum(rollout.terminated, rollout.truncated)
        # A time-out as gym's TimeLimit flags it: the limit hit, no termination.
        valid = 1.0 - rollout.truncated * (1.0 - rollout.terminated)
        gae, value_targets = _compute_gae(
            rewards=rollout.reward - average_reward,
            values=values,
            next_values=next_values,
            terminateds=ends,
            truncateds=rollout.truncated,
            gamma=1.0,
            gae_lambda=self.gae_lambda,
        )

        # The update's key: it shuffles the rollout, and on_target draws on a
        # key derived from it.
        shuffle_key, agent_state = agent_state.get_keys_and_new_state()
        target_metrics: dict = {}
        if extensions:
            target_batch = {
                "observations": rollout.obs,
                "next_observations": rollout.next_obs,
                "rewards": rollout.reward,
                "terminated": rollout.terminated,
                "truncated": rollout.truncated,
                "gae": gae,
                "values": values,
                "average_reward": average_reward,
                "gamma": None,  # APO is average-reward (differential V)
            }
            value_targets, target_metrics = extensions.apply_on_target(
                agent_state,
                target_batch,
                value_targets,
                agent_state.collector_state.timestep,
                extension_key(shuffle_key),
                total_timesteps,
            )

        # Normalised once over the rollout, by the trained rows' mean and
        # torch's unbiased std, floored at 1e-6 (base.py:79-88).
        if self.normalize_advantage:
            n_trained = valid.sum()
            mean = (gae * valid).sum() / n_trained
            std = jnp.sqrt(((gae - mean) ** 2 * valid).sum() / (n_trained - 1))
            gae = (gae - mean) / jnp.maximum(std, 1e-6)

        actions, log_probs = get_actions_and_log_probs(rollout)
        batch = {
            "obs": rollout.obs,
            "actions": actions,
            "targets": value_targets,
            "gae": gae,
            "log_probs": log_probs,
            "valid": valid,
            "raw_obs": rollout.raw_obs,
            "raw_actions": rollout.raw_action,
            # None leaves the minibatches as they were without an expert.
            "a_expert": rollout.a_expert if self.expert is not None else None,
        }
        # APO always cuts the rollout one way: shuffled single steps, as xtma/apo does.
        num_minibatches = resolve_num_minibatches(self)

        def minibatches(key: jax.Array) -> dict:
            """One epoch's partition of the rollout, each minibatch with a key
            for its entropy sample, derived from the epoch's key so the agent's
            own key stream is drawn as before."""
            parts = get_minibatches_from_batch(batch, key, num_minibatches)
            keys = jax.random.split(jax.random.fold_in(key, 1), num_minibatches)
            parts.update({"entropy_key": keys})
            return parts

        step = partial(self.minibatch_step, value_bias=value_bias, context=context)
        agent_state, aux = run_epochs(
            agent_state, shuffle_key, minibatches, step, self.n_epochs
        )
        aux = aux.replace(extensions=merge_metrics(target_metrics, aux.extensions))
        return agent_state.replace(n_updates=agent_state.n_updates + 1), aux

    def minibatch_step(
        self,
        agent_state: APOState,
        minibatch: dict,
        value_bias: jax.Array,
        context: TrainContext,
    ) -> tuple[APOState, AuxiliaryLogs]:
        """One step of both networks on the minibatch, as APO's
        official code takes it (xtma/apo, apo/algos/apg/): one Adam
        over the whole model (base.py:28) down the clipped surrogate plus
        vf_coef times the value loss minus ent_coef times the entropy
        (appo.py:145-158), its gradient clipped to global norm
        max_grad_norm over every parameter (appo.py:106-112): the two
        networks' Adams on that one gradient (actor_learning_rate).

        The extensions' critic and actor terms join the sum, on two keys
        derived from the minibatch's entropy key; the actor's also read
        a_expert, the expert's actions collection stored at the rows
        (None without an expert).
        """
        actor_state, critic_state = agent_state.actor_state, agent_state.critic_state
        critic_key, actor_key = jax.random.split(
            extension_key(minibatch["entropy_key"])
        )
        timestep = agent_state.collector_state.timestep
        extensions, total_timesteps = context.extensions, context.total_timesteps

        def summed_loss(params: tuple) -> Tuple[jax.Array, AuxiliaryLogs]:
            actor_params, critic_params = params
            policy_loss, (policy_aux, pi_mean) = policy_loss_function(
                actor_params,
                actor_state,
                observations=minibatch["obs"],
                actions=minibatch["actions"],
                log_probs=minibatch["log_probs"],
                gae=minibatch["gae"],
                clip_coef=at(self.clip_range, timestep),
                ent_coef=self.ent_coef,
                raw_actions=minibatch["raw_actions"],
                entropy_key=minibatch["entropy_key"],
                valid=minibatch.get("valid"),
            )
            value_loss, value_aux = value_loss_function(
                critic_params,
                critic_state,
                minibatch["obs"],
                minibatch["targets"],
                self.nu,
                value_bias,
                self.vf_coef,
                minibatch.get("valid"),
            )
            critic_batch = {
                "observations": minibatch["obs"],
                "targets": minibatch["targets"],
                "critic_params": critic_params,
                "critic_state": critic_state,
            }
            actor_batch = {
                "observations": minibatch["obs"],
                "raw_observations": minibatch["raw_obs"],
                "pi_mean": pi_mean,
                "actor_params": actor_params,
                "actor_state": actor_state,
                "a_expert": minibatch["a_expert"],
            }
            critic_extra, critic_metrics = extensions.apply_critic_loss(
                agent_state, critic_batch, timestep, critic_key, total_timesteps
            )
            actor_extra, actor_metrics = extensions.apply_actor_loss(
                agent_state, actor_batch, timestep, actor_key, total_timesteps
            )
            loss = policy_loss + value_loss + critic_extra + actor_extra
            metrics = merge_metrics(critic_metrics, actor_metrics)
            return loss, AuxiliaryLogs(
                policy=policy_aux, value=value_aux, extensions=metrics
            )

        (actor_grads, critic_grads), aux = compute_gradients(
            summed_loss, (actor_state.params, critic_state.params)
        )
        actor_state, critic_state, actor_grad_norm, critic_grad_norm = (
            apply_gradients_clipped_together(
                actor_state,
                critic_state,
                actor_grads,
                critic_grads,
                self.max_grad_norm,
                norm_eps=TORCH_CLIP_EPS,
            )
        )
        # Each network's gradient norm before the joint clip, as PPO logs them.
        aux = AuxiliaryLogs(
            policy=aux.policy.replace(grad_norm=actor_grad_norm),
            value=aux.value.replace(
                grad_norm=critic_grad_norm, average_reward=agent_state.average_reward
            ),
            extensions=aux.extensions,
        )
        return agent_state.replace(
            actor_state=actor_state, critic_state=critic_state
        ), aux

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "APO"
    action_space = "any"
    capabilities = frozenset(
        {"actor", "obs_networks", "policy_density", "v_critic", "reward_learning"}
    )
    supported_extension_phases = LOOP_PHASES.union(
        {"on_target", "critic_loss", "actor_loss", "transform_action", "actuate"}
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "on_target": frozenset({"terminated", "truncated", "gae", "values"}),
        "critic_loss": frozenset({"critic_params", "critic_state"}),
        "actor_loss": frozenset(
            {"actor_state", "observations", "raw_observations", "pi_mean", "a_expert"}
        ),
    }
    extension_context = frozenset({"actor_optimizer_args", "env_args", "expert_policy"})
    stores_expert_actions = True


# -- APO's equations, in the order the update calls them ---------------------
def valid_mean(row_values: jax.Array, valid: Optional[jax.Array]) -> jax.Array:
    """The mean over the rows trained on, rlpyt's valid_mean
    (rlpyt/utils/tensor.py); every row when valid is None. A
    minibatch with no row to train on gives 0, not rlpyt's 0 / 0."""
    if valid is None:
        return jnp.mean(row_values)
    valid = valid.reshape(row_values.shape)
    return jnp.sum(row_values * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    clip_coef: float,
    ent_coef: float,
    raw_actions: Optional[jax.Array] = None,
    entropy_key: Optional[jax.Array] = None,
    valid: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, Optional[jax.Array]]]:
    """PPO's clipped surrogate with PPO's entropy bonus (APO's Gaussian's,
    exact; a squashed policy's executed action's, sampled with
    entropy_key), each averaged over the valid rows (appo.py:149-158);
    also returns the policy mean for extensions (None for a discrete
    policy, which has no mean action)."""
    policy, _ = get_pi(actor_state, actor_params, observations)
    new_log_probs = recompute_log_prob(policy, actions, raw_actions)
    entropy = policy_entropy(policy, entropy_key)
    ratio = jnp.exp(new_log_probs - log_probs)
    assert ratio.shape[0] == gae.shape[0], (
        f"Mismatch between ratio shape ({ratio.shape}) and gae shape ({gae.shape})"
    )
    loss_actor, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    total_loss = valid_mean(loss_actor, valid) - ent_coef * valid_mean(entropy, valid)
    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy.mean(),
        approx_kl=approximate_kl(new_log_probs - log_probs),
        policy_std=mean_policy_std(policy),
    )
    pi_mean = None if isinstance(policy, distrax.Categorical) else policy.mean()
    return total_loss, (aux, pi_mean)


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    nu: float,
    b: float | jax.Array,
    vf_coef: float = 1.0,
    valid: Optional[jax.Array] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """The differential value loss vf_coef 0.5 (V(s) - (target - nu b))^2.

    APO's official code (xtma/apo, apo/algos/apg/) subtracts
    nu b from the critic's target, so with b the mean value each fit
    pulls the values back towards zero mean (Ma et al., 2021, the value
    bias penalty); adding it would push them away and let |b| grow.
    vf_coef is its value_loss_coeff (appo.py:152-153), the mean
    over the valid rows.
    """
    # The single critic still has the ensemble's leading axis.
    value_preds = predict_value(critic_states, critic_params, observations).squeeze(0)
    loss = (
        vf_coef * 0.5 * valid_mean((value_preds - (value_targets - nu * b)) ** 2, valid)
    )
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=value_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
        explained_variance=explained_variance(value_preds, value_targets),
        value_bias=jnp.asarray(b),
    )


#: What torch's clip_grad_norm_ adds to the norm before dividing by it
#: (torch 1.5.1, rlpyt's pin, torch/nn/utils/clip_grad.py:33-37).
TORCH_CLIP_EPS = 1e-6


# -- State, logged metrics and the legacy view --------------------------------
@struct.dataclass(kw_only=True)
class APOState(BaseAgentState):
    """APO's state: the base state's, the reward rate and the value bias."""

    average_reward: float
    b: float

    #: Its networks' gradients are clipped together
    #: (steps.apply_gradients_clipped_together): a checkpoint from when each
    #: clipped its own restores (ajax.checkpoint).
    #: A ClassVar: a flag on the class, shared by every APO state, not stored in its arrays.
    joint_gradient_clip: ClassVar[bool] = True
    #: How to restore a checkpoint of APO's former default network, added to
    #: a shape mismatch's error (ajax.checkpoint). Expand-then-contract
    #: (CLAUDE.md, refactoring rule 10): delete it once no such checkpoint
    #: remains.
    restore_hint: ClassVar[str] = (
        '. APO\'s default hidden layers were ("128", "tanh", "128", "tanh")'
        " before they became its reference code's two of 64 units: an APO"
        ' checkpoint saved then restores into APO(..., actor_architecture=("128",'
        ' "tanh", "128", "tanh"), critic_architecture=("128", "tanh", "128",'
        ' "tanh")), its other settings unchanged.'
    )


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_probs: jax.Array
    old_log_probs: jax.Array
    clip_fraction: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    policy_std: jax.Array
    # The loss builds these metrics before the gradient exists; the step
    # fills it in after.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass(kw_only=True)
class ValueAuxiliaries:
    """The critic's metrics, the value bias b the loss used and the
    reward rate rho the targets used (filled in after the step)."""

    critic_loss: jax.Array
    predictions: jax.Array
    targets: jax.Array
    explained_variance: jax.Array
    grad_norm: jax.Array | float = jnp.nan  # filled in after the step, as the policy's
    value_bias: jax.Array
    average_reward: jax.Array | float = jnp.nan


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


#: agent.agent_config's class: a view of the fields until 0.5.0.
APOConfig = APO.declare_config(
    "ent_coef clip_range n_steps batch_size n_epochs gae_lambda"
    " normalize_advantage alpha nu num_minibatches vf_coef max_grad_norm"
    " expose_recent_rollout"
)
