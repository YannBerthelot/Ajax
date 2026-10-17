"""The ImitationLoss extension: its term, its Q-filter, its annealed weight,
and its wiring into the actor losses.

TD3, REDQ, ASAC, AVG, APO, SAC and PPO put the actor's mean action
(``pi_mean``) and the raw observations into their actor-loss batch. One actor update per agent,
run eagerly on a tiny network, checks that the extension's term changes the
actor gradient and has no effect at zero weight or when absent.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Callable

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC import train_ASAC
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG import train_AVG
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.PPO import PPO
from ajax.agents.REDQ import train_REDQ
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC import train_SAC
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3 import train_TD3
from ajax.agents.TD3.TD3 import TD3
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.extensions.expert import ImitationLoss, ResidualPolicy, q_filter_weight
from ajax.logging.wandb_logging import LoggingConfig
from ajax.types import NormalizationInfo

NET = ("8", "relu")
BATCH = 4


@dataclasses.dataclass(frozen=True)
class ConstantExpert:
    action: float

    def __call__(self, obs: jax.Array) -> jax.Array:
        return jnp.full(obs.shape[:-1] + (1,), self.action, jnp.float32)


EXPERT = ConstantExpert(0.9)
KEY = jax.random.PRNGKey(0)
CTX = ExtensionContext(step=jnp.asarray(0), rng=KEY)


def test_term_is_the_scaled_squared_distance_to_the_expert() -> None:
    pi_mean = jnp.array([[0.1], [-0.3]])
    obs = jnp.zeros((2, 3))
    batch: dict[str, Any] = {"pi_mean": pi_mean, "raw_observations": obs}
    term = ImitationLoss(expert_policy=EXPERT, coef=2.0).actor_loss(
        None, (), batch, CTX
    )
    expected = 2.0 * np.mean((np.array([0.1, -0.3]) - 0.9) ** 2) / 4.0
    np.testing.assert_allclose(float(term), expected, rtol=1e-6)
    # Without raw observations the expert reads the agent's observations.
    batch = {
        "pi_mean": pi_mean,
        "raw_observations": None,
        "observations": obs,
    }
    again = ImitationLoss(expert_policy=EXPERT, coef=2.0).actor_loss(
        None, (), batch, CTX
    )
    np.testing.assert_allclose(float(again), expected, rtol=1e-6)


def test_the_annealed_term_reaches_zero_at_its_fraction_of_training() -> None:
    """With ``anneal_frac`` the weight is 1 at the start, half way down at
    half that fraction, and 0 from it on; it must lie in (0, 1]."""
    batch: dict[str, Any] = {
        "pi_mean": jnp.array([[0.1], [-0.3]]),
        "raw_observations": jnp.zeros((2, 3)),
    }
    full = float(ImitationLoss(expert_policy=EXPERT).actor_loss(None, (), batch, CTX))
    term = ImitationLoss(expert_policy=EXPERT, anneal_frac=0.5)

    def at(step: int) -> float:
        ctx = ExtensionContext(step=jnp.asarray(step), rng=KEY, total_steps=1_000)
        return float(term.actor_loss(None, (), batch, ctx))

    np.testing.assert_allclose(at(0), full, rtol=1e-6)
    np.testing.assert_allclose(at(250), full / 2, rtol=1e-6)
    assert at(500) == 0.0 and at(1_000) == 0.0
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="anneal_frac"):
            ImitationLoss(expert_policy=EXPERT, anneal_frac=bad)


def test_agent_without_policy_mean_raises() -> None:
    """With the observations (test_no_silent_noop's empty batch has none)."""
    batch = {"observations": jnp.zeros((2, 3))}
    with pytest.raises(ValueError, match="pi_mean"):
        ImitationLoss(expert_policy=EXPERT).actor_loss(None, (), batch, CTX)


def _critic() -> Any:
    """Two critics, Q_1 = a and Q_2 = 2 - a: on [-1, 1] their minimum is Q_1,
    their mean a constant (no row kept), their maximum Q_2 (the rows
    inverted)."""

    def apply(params: Any, x: jax.Array) -> jax.Array:
        a = x[..., -1:]
        return jnp.stack([a, 2.0 - a])

    return types.SimpleNamespace(apply_fn=apply, params=None)


PI = jnp.array([[-0.5], [0.0], [0.89], [0.9], [0.91], [1.0]])
KEPT = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # Q(s, a_E = 0.9) > Q(s, pi) strictly


def test_the_q_filter_weight_is_zero_wherever_the_critic_prefers_the_policy() -> None:
    """Nair et al. 2018, eq. 8, on the ensemble minimum: 1 where Q(s, a_E) >
    Q(s, pi), 0 where it is below or equal; the term then divides by every
    row, filtered or not."""
    obs = jnp.zeros((PI.shape[0], 3))
    keep = q_filter_weight(_critic(), obs, EXPERT(obs), PI)
    np.testing.assert_array_equal(keep[:, 0], KEPT)
    batch = {"pi_mean": PI, "raw_observations": obs, "observations": obs}
    ext = ImitationLoss(expert_policy=EXPERT, coef=2.0, q_filter=True)
    with pytest.raises(ValueError, match="critic_state"):
        ext.actor_loss(None, (), batch, CTX)
    term = ext.actor_loss(None, (), batch | {"critic_state": _critic()}, CTX)
    want = 2.0 * np.mean(np.array(KEPT) * (np.asarray(PI)[:, 0] - 0.9) ** 2) / 4.0
    np.testing.assert_allclose(float(term), want, rtol=1e-6)


def test_the_agents_own_expert_is_read_as_the_agent_stored_it() -> None:
    """Bound to an agent holding the same expert, the term reads the stored
    actions (SAC's ``a_expert``, from the expert's state); another expert is
    called. With the filter, ResidualPolicy is refused: its critic reads the
    executed action, not the policy's mean."""
    obs = jnp.zeros((PI.shape[0], 3))
    batch = {
        "pi_mean": PI,
        "raw_observations": obs,
        "a_expert": jnp.full_like(PI, -0.2),
    }
    own = ImitationLoss(expert_policy=EXPERT).bind_to_agent(expert_policy=EXPERT)
    want = np.mean((np.asarray(PI) + 0.2) ** 2) / 4.0
    np.testing.assert_allclose(float(own.actor_loss(None, (), batch, CTX)), want)
    other = ImitationLoss(expert_policy=ConstantExpert(0.9))
    assert not other.bind_to_agent(expert_policy=EXPERT).reads_stored_expert
    residual = {"buffer": object(), "extensions": (ResidualPolicy(),)}
    with pytest.raises(ValueError, match="ResidualPolicy"):
        ImitationLoss(expert_policy=EXPERT, q_filter=True).bind_to_agent(**residual)


class _StatefulHalfTorque:
    """A stateful expert (SAC's convention), which only its stored actions
    let ImitationLoss read."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0), state


@dataclasses.dataclass(frozen=True)
class _FirstInput:
    """An actor state acting tanh of its first input, as get_pi reads it."""

    obs_norm_info: Any
    params: Any = None
    hidden_state: Any = None

    def apply(self, params: Any, obs: jax.Array) -> Any:
        return distrax.Deterministic(jnp.tanh(obs[..., :1]))

    def replace(self, **changes: Any) -> "_FirstInput":
        return dataclasses.replace(self, **changes)


def test_sac_logs_the_filtered_fraction_on_a_replay_batch() -> None:
    """SAC with its stateful expert trains with the filter and logs, at each
    evaluation, the share of a replay batch it keeps; the metric's
    definition on the batch its key draws; the actor reads its observations
    normalised."""
    expert = _StatefulHalfTorque()
    logged = LoggingConfig({}, log_frequency=32, use_wandb=False)
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        extensions=(ImitationLoss(expert_policy=expert, q_filter=True),),
    )
    state, rows = agent.train(seed=0, n_timesteps=64, logging_config=logged)
    frac = np.asarray(rows["imitation/q_filter_frac"][0, :2])  # at 32 and 64 steps
    assert np.all((frac >= 0.0) & (frac <= 1.0)), frac
    state = jax.tree.map(lambda x: x[0], state)
    held = agent.extension_stack.extensions[0]  # its expert in agent units
    assert isinstance(held, ImitationLoss)
    ext = held.bind_to_agent(buffer=agent.buffer, expert_policy=agent.expert_policy)
    assert ext.reads_stored_expert
    got = ext.eval_metrics(state, (), KEY, CTX)["imitation/q_filter_frac"]
    batch = get_batch_from_buffer(agent.buffer, state.collector_state.buffer_state, KEY)
    actor = state.actor_state
    pi_mean = get_pi(actor, actor.params, batch.obs)[0].mean()
    keep = q_filter_weight(state.critic_state, batch.obs, batch.a_expert, pi_mean)
    np.testing.assert_allclose(float(got), float(keep.mean()))
    # Under Q = a, an actor acting tanh of its first input, normalised from a
    # mean of -100 (tanh(+100) = 1, above every a_E = -cos(theta) / 4),
    # keeps no row; read raw (below a_E wherever the pendulum hangs), most.
    shift = jnp.full((3,), -100.0)
    norm = NormalizationInfo(
        var=jnp.ones(3), count=jnp.ones(()), mean=shift, mean_2=shift
    )
    shifted = state.replace(actor_state=_FirstInput(norm), critic_state=_critic())
    assert (
        float(ext.eval_metrics(shifted, (), KEY, CTX)["imitation/q_filter_frac"]) == 0.0
    )


def _ext_state(state: Any, stack: ExtensionStack) -> Any:
    """What make_train does before training: one ext_state per extension."""
    return stack.fold_init_states(state, KEY)


def _obs(agent: Any) -> jax.Array:
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    return jax.random.normal(jax.random.PRNGKey(1), (BATCH, *obs_shape))


def _td3_update(stack: ExtensionStack) -> Any:
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = train_TD3.init_TD3(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.buffer,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    state, _ = train_TD3.update_policy(state, obs, obs, stack, total_timesteps=1)
    return state.actor_state.params


def _sac_family_update(agent_cls: type, module: Any, init: Callable) -> Callable:
    def update(stack: ExtensionStack) -> Any:
        agent = agent_cls(
            "Pendulum-v1", actor_architecture=NET, critic_architecture=NET
        )
        state = _ext_state(init(agent), stack)
        obs = _obs(agent)
        out = module.update_policy(state, obs, obs, stack, total_timesteps=1)
        return out[0].actor_state.params

    return update


def _init_redq(agent: REDQ) -> Any:
    return train_REDQ.init_REDQ(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
        number_of_critics=agent.agent_config.num_critics,
    )


def _init_asac(agent: ASAC) -> Any:
    return train_ASAC.init_ASAC(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
    )


def _sac_update(stack: ExtensionStack) -> Any:
    agent = SAC("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = train_SAC.init_SAC(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    state, _, _ = train_SAC.update_policy(
        state, obs, obs, extension_stack=stack, total_timesteps=1
    )
    return state.actor_state.params


def _apo_update(stack: ExtensionStack) -> Any:
    agent = APO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    state = train_APO.init_APO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    state, _ = train_APO.update_policy(
        state,
        obs,
        actions=jnp.zeros((BATCH, 1)),
        gae=jnp.ones((BATCH, 1)),
        log_probs=jnp.zeros((BATCH, 1)),
        clip_coef=0.2,
        ent_coef=0.0,
        extension_stack=stack,
        total_timesteps=1,
        extension_key=KEY,
        raw_observations=obs,
        entropy_key=KEY,  # a squashed policy's entropy is estimated by sampling
    )
    return state.actor_state.params


def _ppo_update(stack: ExtensionStack) -> Any:
    agent = PPO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    state = train_PPO.init_PPO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    state = _ext_state(state, stack)
    obs, actor = _obs(agent), state.actor_state
    extra = train_PPO._stack_actor_loss(stack, state, KEY, 1, raw_observations=obs)

    def loss(params: Any) -> Any:
        return train_PPO.policy_loss_function(
            params,
            actor,
            obs,
            actions=jnp.zeros((BATCH, 1)),
            log_probs=jnp.zeros((BATCH, 1)),
            gae=jnp.ones((BATCH, 1)),
            recurrent=False,
            clip_coef=0.2,
            ent_coef=0.0,
            advantage_normalization=False,
            extra_loss_fn=extra,
            entropy_rng=KEY,
        )

    grads, _ = jax.grad(loss, has_aux=True)(actor.params)
    return actor.apply_gradients(grads=grads).params


def _avg_update(stack: ExtensionStack) -> Any:
    agent = AVG("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = train_AVG.init_AVG(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    raw_actions = jnp.zeros((BATCH, 1))  # the pre-squash sample AVG steps through
    state, _ = train_AVG.update_policy(
        state, obs, raw_actions, stack, total_timesteps=1, raw_observations=obs
    )
    return state.actor_state.params


UPDATES: dict[str, Callable[[ExtensionStack], Any]] = {
    "TD3": _td3_update,
    "REDQ": _sac_family_update(REDQ, train_REDQ, _init_redq),
    "ASAC": _sac_family_update(ASAC, train_ASAC, _init_asac),
    "APO": _apo_update,
    "SAC": _sac_update,
    "PPO": _ppo_update,
    "AVG": _avg_update,
}


def _stack(coef: float) -> ExtensionStack:
    return ExtensionStack((ImitationLoss(expert_policy=EXPERT, coef=coef),))


def _same(a: Any, b: Any) -> bool:
    leaves = zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    return all(np.allclose(x, y, rtol=1e-6, atol=1e-8) for x, y in leaves)


@pytest.mark.parametrize("agent", sorted(UPDATES))
def test_imitation_term_steers_the_actor_update(agent: str) -> None:
    update = UPDATES[agent]
    absent = update(ExtensionStack())
    assert _same(update(_stack(0.0)), absent), "zero weight must change nothing"
    assert not _same(update(_stack(100.0)), absent), "the term must reach the actor"


@pytest.mark.parametrize("geometry", [{}, {"unroll_length": 8, "num_minibatches": 2}])
def test_ppo_trains_with_the_term_in_its_minibatches(geometry: dict) -> None:
    """PPO's minibatches carry the raw observations the term's expert reads,
    flat and time-preserving (``unroll_length``) alike: the term moves the
    actor."""

    def actor(*extensions: Any) -> Any:
        agent = PPO(
            "Pendulum-v1",
            n_envs=2,
            n_steps=16,
            actor_architecture=NET,
            critic_architecture=NET,
            extensions=extensions,
            **geometry,
        )
        return agent.train(seed=0, n_timesteps=32)[0].actor_state.params

    assert not _same(actor(ImitationLoss(EXPERT, coef=100.0)), actor())


def test_apg_trains_with_the_term_along_its_rollout() -> None:
    """APG's actor batch is its rollout: the observations its controller saw
    and the controller's own actions, before their map onto each system's
    bounds; the term moves the controller."""
    from tests.agents.APG.test_APG import make_agent, tracking_class, tracking_env

    env, params = tracking_env()

    def actor(*extensions: Any) -> Any:
        agent = make_agent(env, tracking_class(params), extensions=extensions)
        steps = agent.env_args.n_envs * agent.agent_config.horizon
        return agent.train(seed=0, n_timesteps=steps)[0].actor_state.params

    assert not _same(actor(ImitationLoss(EXPERT, coef=100.0)), actor())


def test_udrl_trains_with_the_term_and_refuses_it_on_normalised_observations() -> None:
    """Continuous UDRL's actor batch carries the env's part of each
    observation and the policy's mean action: the term moves the actor. Its
    segment buffer keeps normalised observations, so an expert, which reads
    the env's own, is refused under normalize_observations."""
    from ajax import UDRL

    kw: dict[str, Any] = {
        "n_envs": 2,
        "n_steps": 8,
        "batch_size": 8,
        "n_updates_per_iter": 2,
    }

    def actor(*extensions: Any) -> Any:
        agent = UDRL("Pendulum-v1", actor_architecture=NET, extensions=extensions, **kw)
        return agent.train(seed=0, n_timesteps=32)[0].actor_state.params

    assert not _same(actor(ImitationLoss(EXPERT, coef=100.0)), actor())
    with pytest.raises(ValueError, match="normalize_observations"):
        UDRL(
            "Pendulum-v1",
            normalize_observations=True,
            extensions=(ImitationLoss(EXPERT),),
            **kw,
        )
