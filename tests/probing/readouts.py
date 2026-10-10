"""One readout layer: what a trained agent says at chosen inputs, per seed.

Readers take one seed's ``Nets`` (``per_seed`` maps them over seeds). Inputs
pass the env-side normaliser (``env_stats``) when given, then ``get_pi`` and
``predict_value``, training's own paths; recurrent readers start from a
fresh carry; draws take explicit keys; state readers return numpy."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.environments.interaction import get_pi, get_pi_sequence
from ajax.networks.memory import zeros_carry_like
from ajax.networks.networks import predict_value, predict_value_sequence
from ajax.utils import online_normalize

from .agents import FAMILY

KEY = jax.random.PRNGKey(0)


@struct.dataclass
class Nets:
    actor: Any
    critic: Any = None
    alpha: Any = None
    extra: Any = None
    stats: Any = None


def nets(state: Any, extra: Any = None, stats: Any = None) -> Nets:
    """``extra``: any other seed-batched pytree a reader needs."""
    critic, alpha = getattr(state, "critic_state", None), getattr(state, "alpha", None)
    return Nets(state.actor_state, critic, alpha, extra, stats)


def env_stats(run: Any, applied: bool = True) -> Any:
    """The env-side observation normaliser's statistics; None without one
    or, if ``applied``, when the agent is handed raw observations (PPO)."""
    env = run.agent.env_args.env
    while env is not None and not hasattr(env, "apply_normalization"):
        env = getattr(env, "_env", None)
    on = env is not None and env.normalize_obs
    on = on and (env.apply_normalization or not applied)
    return run.state.collector_state.env_state.normalization_info.obs if on else None


def per_seed(fn: Callable[[Nets], Mapping[str, Any]], n: Nets) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in jax.vmap(fn)(n).items()}


def _x(x: Any) -> jax.Array:
    return jnp.asarray(x, jnp.float32).reshape(1, -1)


def inp(n: Nets, x: Any) -> jax.Array:
    """What the networks receive for the raw observation ``x``: the env-side
    normaliser's own ``online_normalize``, with env 0's final statistics."""
    if n.stats is None:
        return _x(x)
    s = n.stats  # leaves (n_envs, 1, obs_dim), every row the same
    return online_normalize(_x(x), s.count[0], s.mean[0], s.mean_2[0], train=False)[0]


def input_error(n: Nets, raw: jax.Array, last_obs: jax.Array) -> jax.Array:
    """max |inp(raw) - last_obs|, ``raw`` each env's current raw observation."""
    built = jax.vmap(lambda x: inp(n, x))(raw).reshape(-1)
    return jnp.max(jnp.abs(built - last_obs.reshape(-1)))


def pi(n: Nets, x: Any) -> Any:
    return get_pi(n.actor, n.actor.params, inp(n, x))[0]


def q_values(n: Nets, x: Any) -> jax.Array:
    return pi(n, x).q_values.reshape(-1)


def critic(n: Nets, x: Any, a: Any = None, params: Any = None) -> jax.Array:
    """V(x), or Q(x, a), averaged over the critic ensemble."""
    xs = inp(n, x) if a is None else jnp.concatenate([inp(n, x), _x(a)], -1)
    params = n.critic.params if params is None else params
    return predict_value(n.critic, params, xs).mean()


def value(agent: str, n: Nets, x: Any) -> jax.Array:
    """The agent's own value: max_a Q, V, or Q at the policy's mean action."""
    family = FAMILY.get(agent, "q")
    if family == "dqn":
        return q_values(n, x).max()
    return critic(n, x, None if family == "v" else pi(n, x).mean())


def action(n: Nets, x: Any, clip: bool = True) -> jax.Array:
    """The deterministic action, clipped to [-1, 1] as the env applies it."""
    a = pi(n, x).mean().reshape(())
    return jnp.clip(a, -1.0, 1.0) if clip else a


def alpha(n: Nets) -> jax.Array:
    return jnp.exp(n.alpha.params["log_alpha"]).reshape(())


def entropy(p: Any, key: jax.Array, samples: int = 512) -> jax.Array:
    """Monte Carlo entropy of the distribution ``p``."""
    _, log_prob = p.sample_and_log_prob(seed=key, sample_shape=(samples,))
    return -jnp.mean(log_prob)


def soft_value(p: Any, alpha: Any, reward: Callable, key: Any, n: int = 4096) -> Any:
    """E_{a ~ p}[reward(a) - alpha log p(a)] by Monte Carlo, 1-D actions."""
    a, log_prob = p.sample_and_log_prob(seed=key, sample_shape=(n,))
    return jnp.mean(reward(a.reshape(-1)) - alpha * log_prob.reshape(-1))


def step_actor(n: Nets, obs: jax.Array, starts: jax.Array) -> jax.Array:
    """(T, B, act) mean actions stepped one at a time through (T, B, obs),
    each flagged with its episode start, as collection runs the actor."""

    def step(carry: Any, x: tuple) -> tuple:
        actor = n.actor.replace(hidden_state=carry)
        p, stepped = get_pi(actor, actor.params, x[0], x[1], recurrent=True)
        return stepped.hidden_state, p.mean()[0]

    carry = zeros_carry_like(n.actor.hidden_state, obs.shape[1], batch_axis=0)
    return jax.lax.scan(step, carry, (obs, starts))[1]


def actor_sequence(n: Nets, obs: jax.Array, starts: jax.Array) -> Any:
    """The policy over a (T, B, obs) sequence, as training reads it."""
    carry = zeros_carry_like(n.actor.hidden_state, obs.shape[1], batch_axis=0)
    return get_pi_sequence(n.actor, n.actor.params, obs, starts, carry)[0]


def critic_sequence(n: Nets, x: jax.Array, starts: jax.Array) -> jax.Array:
    """(T, B) values over a (T, B, obs [+ action]) sequence, ensemble mean."""
    carry = zeros_carry_like(n.critic.hidden_state, x.shape[1], batch_axis=1)
    v, _ = predict_value_sequence(n.critic, n.critic.params, x, starts, carry)
    return v.mean(axis=0)[..., 0]


def field(state: Any, name: str) -> np.ndarray:
    """A field of the (possibly wrapped) env state."""
    env_state = state.collector_state.env_state
    while not hasattr(env_state, name):
        env_state = env_state.env_state
    return np.asarray(getattr(env_state, name))


def record(state: Any, which: str) -> np.ndarray:
    """The env's ``first_*`` or ``last_*`` record in time order, (seeds,
    envs, steps); the ``last_*`` ring is read oldest first."""
    values = field(state, which)
    if which.startswith("last_"):
        n = values.shape[-1]
        idx = (field(state, "clock")[..., None] + np.arange(n)) % n
        values = np.take_along_axis(values, idx, axis=-1)
    return values


def replay_rows(state: Any, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The written rows of the replay buffer, (seeds, envs, rows, ...);
    raises if the buffer wrapped or seeds wrote different counts."""
    buffer = state.collector_state.buffer_state
    written = np.asarray(buffer.current_index).reshape(-1)
    if np.asarray(buffer.is_full).any() or (written != written[0]).any():
        raise RuntimeError("replay buffer wrapped or uneven across seeds")
    return {k: np.asarray(buffer.experience[k])[..., : written[0], :] for k in keys}


def rollout_rows(state: Any, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The last rollout's fields, (seeds, envs, steps, ...)."""
    r = state.last_rollout
    return {k: np.moveaxis(np.asarray(getattr(r, k)), 1, 2) for k in keys}


def optimizer_steps(state: Any) -> dict[str, jax.Array]:
    """Each optimiser's ``.step``: actor, critic and temperature."""
    out = {"actor": state.actor_state.step}
    for name, attr in (("critic", "critic_state"), ("alpha", "alpha")):
        if hasattr(getattr(state, attr, None), "step"):
            out[name] = getattr(state, attr).step
    return out


def checksums(params: Any, n_seeds: int) -> np.ndarray:
    """One float64 sum of every parameter per seed."""
    leaves = [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(params)]
    return np.sum([x.reshape(n_seeds, -1).sum(1) for x in leaves], 0)


def differing_leaves(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-6) -> list:
    """Where two states differ: floating leaves beyond ``allclose`` (NaN
    equal to NaN), any other leaf (keys as their data) not exactly equal."""
    la, lb = (jax.tree_util.tree_leaves_with_path(t) for t in (a, b))
    names = [jax.tree_util.keystr(p) for p, _ in la]
    if names != [jax.tree_util.keystr(p) for p, _ in lb]:
        return ["the two states have different structures"]
    found = []
    for name, (_, x), (_, y) in zip(names, la, lb):
        x, y = (np.asarray(_key_data(v)) for v in (x, y))
        if x.shape != y.shape:
            found.append(f"{name}: shape {x.shape} against {y.shape}")
        elif x.dtype.kind in "fc":
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                gap = np.nanmax(np.abs(x.astype(np.float64) - y))
                found.append(f"{name}: largest difference {gap:.3g}")
        elif not np.array_equal(x, y):
            found.append(f"{name}: {x.ravel()[:4]} against {y.ravel()[:4]}")
    return found


def _key_data(v: Any) -> Any:
    is_key = isinstance(v, jax.Array) and jnp.issubdtype(v.dtype, jax.dtypes.prng_key)
    return jax.random.key_data(v) if is_key else v
