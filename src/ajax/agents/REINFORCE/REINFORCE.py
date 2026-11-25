"""REINFORCE (Williams, 1992), as Spinning Up's Vanilla Policy Gradient
(openai/spinningup @038665d, spinup/algos/pytorch/vpg/vpg.py) runs it.

One iteration of Spinning Up's VPG pseudocode. The on-policy family runs the
policy for n_steps in each of n_envs envs (OnPolicyAgent); each line after
that is a function below::

    G_t = sum_k gamma^k r_{t+k}, bootstrapped with V where an episode
          is cut, not ended (rollout end, time limit)    compute_returns
    A_t = GAE(gamma, lambda) of V, normalised over the rollout
                                                         compute_returns
    one Adam step of the policy down
        L_pi = -mean log pi(a_t | s_t) A_t               compute_policy_loss
    value_steps Adam steps of the critic down
        L_V = mean (V(s_t) - G_t)^2                      compute_value_loss

Williams's update is grad log pi(a | s) (G - b), b any baseline that does
not depend on the action (his reinforcement baseline); the return from
step t on (reward-to-go) and a learned V(s) as b are Sutton and Barto's
(2018, Sec. 13.3-13.4), and GAE (Schulman et al., 2016) is Spinning Up's
estimate of G - b. With baseline=False the advantage is the reward-to-go
itself, as Spinning Up's 2_rtg_pg.py (examples/pytorch/pg_math) weights it.
That script trains on whole episodes only; a rollout of n_steps cuts the
last episode of each env, whose steps get no gradient here (owner,
2026-11-25): with no critic, nothing can stand in for their missing rewards.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.diagnostics import approximate_kl, explained_variance, mean_policy_std
from ajax.agents.families import (
    GradientClipping,
    ObservationNormalization,
    OnPolicyAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.minibatch import get_actions_and_log_probs
from ajax.agents.presets import Preset
from ajax.agents.recurrent import compute_rollout_values, get_policy_on_minibatch
from ajax.agents.returns import compute_gae
from ajax.agents.update_steps import gradient_step
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.networks.distributions import policy_entropy, recompute_log_prob
from ajax.networks.networks import predict_value
from ajax.networks.utils import TORCH_LINEAR
from ajax.random_keys import derive_key
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture, FloatOrCallable, InitializationFunction

#: Spinning Up's plain reward-to-go policy gradient (spinningup @038665d,
#: spinup/examples/pytorch/pg_math/2_rtg_pg.py:24-25, :38, :55): one 32-unit
#: tanh layer, Adam at 1e-2, batches of 5000 steps, undiscounted
#: reward-to-go weights, no baseline and no normalisation.
PRESETS = {
    "rtg_pg": Preset(
        {
            "baseline": False,
            "normalize_advantage": False,
            "gamma": 1.0,
            "n_steps": 5000,
            "actor_architecture": ("32", "tanh"),
            "actor_learning_rate": 1e-2,
        },
        "openai/spinningup@038665d spinup/examples/pytorch/pg_math/2_rtg_pg.py:24-55",
    ),
}


@dataclass(kw_only=True, eq=False, repr=False)
class REINFORCE(
    ObservationNormalization,
    RewardNormalization,
    GradientClipping,
    OnPolicyAgent,
):
    """REINFORCE with a learned baseline, for discrete and continuous actions.

    Defaults are Spinning Up's vpg() (vpg.py:88-90, core.py:84, :111-124):
    4000 steps per update from one env, gamma 0.99, GAE lambda 0.97, the
    advantages normalised over the rollout (vpg.py:80-81), one full-batch
    Adam step of the policy at 3e-4, then 80 of the critic at 1e-3, torch's
    Adam (eps 1e-8), no gradient clipping; separate 64-64 tanh networks with
    torch's nn.Linear initialisation; the Gaussian's log-std one learned
    parameter per action dimension starting at -0.5; nothing normalised but
    the advantages. vpg's max_ep_len (1000) is the env's time limit here.
    """

    n_envs: int = 1
    n_steps: int = 4000
    #: Not keywords: one full-batch policy step per rollout (vpg.py:237-250);
    #: a second pass would follow a policy that did not collect the rollout.
    n_epochs: int = dataclasses.field(default=1, init=False)
    num_minibatches: int = dataclasses.field(default=1, init=False)
    actor_learning_rate: FloatOrCallable = 3e-4
    critic_learning_rate: FloatOrCallable = 1e-3
    actor_architecture: Architecture = ("64", "tanh", "64", "tanh")
    critic_architecture: Architecture = ("64", "tanh", "64", "tanh")
    gamma: float = 0.99
    gae_lambda: float = 0.97
    #: Subtract a learned V(s) (GAE) from the returns; False weights each
    #: log-probability by its reward-to-go alone (PRESETS["rtg_pg"]).
    baseline: bool = True
    #: The critic's full-batch Adam steps per rollout (vpg's train_v_iters).
    value_steps: int = 80
    normalize_advantage: bool = True
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = TORCH_LINEAR
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = TORCH_LINEAR
    logits_kernel_init: Optional[Union[str, InitializationFunction]] = TORCH_LINEAR
    value_kernel_init: Optional[Union[str, InitializationFunction]] = TORCH_LINEAR
    log_std_state_independent: bool = True
    log_std_init: float = -0.5

    sweepable: ClassVar[frozenset[str]] = frozenset(
        {"actor_learning_rate", "critic_learning_rate", "gamma", "gae_lambda"}
    )

    def optimizer_settings(self) -> dict[str, Any]:
        return {"eps": 1e-8}  # torch.optim.Adam's default eps (vpg.py:231-232)

    def init(self, key: jax.Array, context: TrainContext) -> REINFORCEState:
        # Each stream by name (CONTRIBUTING.md, "Random keys"), as RVISAC's.
        training_key, init_key, collector_key, evaluation_key = (
            derive_key(key, name)
            for name in ("updates", "networks", "collection", "evaluation")
        )
        env_args = context.env_args
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=check_if_environment_has_continuous_actions(
                env_args.env, env_params=env_args.env_params
            ),
            action_value=False,
            squash=context.network_args.squash,
            num_critics=1,
        )
        return REINFORCEState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=context.make_collector(collector_key),
            n_updates=0,
        )

    def update(
        self,
        agent_state: REINFORCEState,
        rollout: Transition,
        agent_state_at_rollout_start: REINFORCEState,
        context: TrainContext,
    ) -> tuple[REINFORCEState, AuxiliaryLogs]:
        """One policy step, then value_steps critic steps, on the
        (n_steps, n_envs) rollout; both read the returns and advantages
        computed before either moves (vpg.py:237-259)."""
        del context  # no extension phase of REINFORCE's reads it
        entropy_key, agent_state = agent_state.get_keys_and_new_state()
        agent_state, values, next_values, _ = compute_rollout_values(
            agent_state, rollout, agent_state_at_rollout_start, rollout.obs.shape[0]
        )
        returns, advantages, weights = compute_returns(
            rollout, values, next_values, self.gamma, self.gae_lambda, self.baseline
        )
        if self.normalize_advantage:
            advantages = normalise(advantages, weights)
        actions, old_log_probs = get_actions_and_log_probs(rollout)
        batch = {
            "obs": rollout.obs,
            "actions": actions,
            "raw_actions": rollout.raw_action,
            "log_probs": old_log_probs,
        }

        def policy_loss(params: FrozenDict) -> tuple:
            return compute_policy_loss(
                params, agent_state.actor_state, batch, advantages, weights, entropy_key
            )

        actor, policy_aux, policy_grad_norm = gradient_step(
            agent_state.actor_state, policy_loss
        )
        # vpg logs the KL before its step, where it is 0; this is the step's.
        policy = get_policy_on_minibatch(actor, actor.params, batch)
        log_ratio = old_log_probs - recompute_log_prob(
            policy, actions, rollout.raw_action
        )
        policy_aux = policy_aux.replace(
            approx_kl=approximate_kl(log_ratio), grad_norm=policy_grad_norm
        )

        def value_step(critic: LoadedTrainState, _: Any) -> tuple:
            critic, aux, grad_norm = gradient_step(
                critic,
                lambda params: compute_value_loss(params, critic, batch, returns),
            )
            return critic, aux.replace(grad_norm=grad_norm)

        critic = agent_state.critic_state
        value_aux = ValueAuxiliaries(critic_loss=jnp.nan, explained_variance=jnp.nan)
        if self.baseline:  # without one, the critic is never read
            stepped: tuple[LoadedTrainState, ValueAuxiliaries] = jax.lax.scan(
                value_step, critic, length=self.value_steps
            )
            critic = stepped[0]
            value_aux = jax.tree.map(
                lambda leaf: leaf[-1], stepped[1]
            )  # the last step's
        agent_state = agent_state.replace(
            actor_state=actor, critic_state=critic, n_updates=agent_state.n_updates + 1
        )
        return agent_state, AuxiliaryLogs(
            policy=policy_aux, value=value_aux, extensions={}
        )

    name = "REINFORCE"
    action_space = "any"
    capabilities = frozenset(
        {"actor", "discounted", "obs_networks", "policy_density", "v_critic"}
    )
    supported_extension_phases = LOOP_PHASES


# -- REINFORCE's equations, in the order the update calls them ----------------
def compute_returns(
    rollout: Transition,
    values: jax.Array,
    next_values: jax.Array,
    gamma: float,
    gae_lambda: float,
    baseline: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The critic's targets G_t, the policy's advantages and each step's
    weight in the policy loss, on the time-major rollout.

    With baseline, G_t is the reward-to-go bootstrapped with V(s_{t+1})
    wherever an episode is cut rather than terminated (lambda = 1 of GAE:
    vpg's finish_path, vpg.py:42-68), the advantages GAE(lambda), and every
    step weighs 1. Without, the advantage is the reward-to-go stopped at
    every episode end (2_rtg_pg.py:17-22), and the steps of an episode the
    rollout cuts weigh 0."""
    ends = jnp.logical_or(rollout.terminated, rollout.truncated)
    if baseline:
        args = (rollout.reward, values, next_values, rollout.terminated)
        _, returns = compute_gae(*args, rollout.truncated, gamma, 1.0)
        advantages, _ = compute_gae(*args, rollout.truncated, gamma, gae_lambda)
        return returns, advantages, jnp.ones_like(returns)
    zeros = jnp.zeros_like(rollout.reward)
    returns, _ = compute_gae(rollout.reward, zeros, zeros, ends, ends, gamma, 1.0)
    # A step weighs 1 when its episode ends inside the rollout: some end at
    # or after it.
    finished = jax.lax.cummax(ends.astype(returns.dtype), axis=0, reverse=True)
    return returns, returns, finished


def normalise(advantages: jax.Array, weights: jax.Array) -> jax.Array:
    """The advantages less their mean, over their standard deviation, both
    over the steps that weigh (vpg.py:80-81, over the whole batch). The
    1e-8 keeps a rollout of equal advantages finite; vpg divides by the
    bare deviation."""
    count = jnp.maximum(weights.sum(), 1.0)
    mean = (weights * advantages).sum() / count
    std = jnp.sqrt((weights * (advantages - mean) ** 2).sum() / count)
    return (advantages - mean) / (std + 1e-8)


def compute_policy_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    batch: dict,
    advantages: jax.Array,
    weights: jax.Array,
    entropy_key: jax.Array,
) -> tuple[jax.Array, PolicyAuxiliaries]:
    """-mean log pi(a_t | s_t) A_t over the steps that weigh (vpg.py:211-216)."""
    policy = get_policy_on_minibatch(actor_state, actor_params, batch)
    log_probs = recompute_log_prob(policy, batch["actions"], batch["raw_actions"])
    count = jnp.maximum(weights.sum(), 1.0)
    loss = -(weights * log_probs * advantages).sum() / count
    aux = PolicyAuxiliaries(
        policy_loss=loss,
        entropy=policy_entropy(policy, entropy_key).mean(),
        policy_std=mean_policy_std(policy),
        weighted_fraction=weights.mean(),
    )
    return loss, aux


def compute_value_loss(
    critic_params: FrozenDict,
    critic_state: LoadedTrainState,
    batch: dict,
    returns: jax.Array,
) -> tuple[jax.Array, ValueAuxiliaries]:
    """mean (V(s_t) - G_t)^2 (vpg.py:226-228), with no 1/2."""
    predictions = predict_value(critic_state, critic_params, batch["obs"]).squeeze(0)
    loss = jnp.mean((predictions - returns) ** 2)
    aux = ValueAuxiliaries(
        critic_loss=loss, explained_variance=explained_variance(predictions, returns)
    )
    return loss, aux


# -- State and logged metrics -------------------------------------------------
@struct.dataclass
class REINFORCEState(BaseAgentState):
    """REINFORCE's state: the base state's fields."""


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    entropy: jax.Array
    policy_std: jax.Array
    #: The fraction of the rollout's steps in the loss (1 with a baseline).
    weighted_fraction: jax.Array
    # Filled in after the step.
    approx_kl: jax.Array | float = jnp.nan
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    # NaN without a baseline, whose critic is never trained.
    critic_loss: jax.Array | float
    explained_variance: jax.Array | float
    grad_norm: jax.Array | float = jnp.nan  # filled in after the step


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics (none here)
