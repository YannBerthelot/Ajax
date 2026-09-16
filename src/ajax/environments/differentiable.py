"""Differentiable closed-loop rollouts (backpropagation through the simulator).

The collectors in :mod:`ajax.environments.interaction` wrap every env step
in ``stop_gradient``: RL agents treat the environment as a black box. This
module is the opposite path, for *analytic policy gradient* methods that
differentiate an objective through the dynamics themselves (Busetto et al.
2024's in-context controller, APG/SHAC-style trainers, ...):

* :func:`with_transition_gradients` configures a (possibly wrapped) gymnax
  env so its transitions keep gradients. gymnax >= 1.0 detaches them by
  default and only some envs support the opt-in (Pendulum,
  MountainCarContinuous, PointRobot, Reacher, Swimmer at the time of
  writing); unsupported envs raise instead of silently training on zero
  gradients.
* :func:`closed_loop_rollout` runs a fixed-horizon closed loop between a
  stateful policy and the env with no ``stop_gradient`` anywhere, so
  ``jax.grad`` of any function of the returned :class:`Rollout` flows
  through the policy, the environment, and the policy's carry across
  time (BPTT over the whole horizon).

Gymnax only: brax/playground envs take no per-step params and have their
own (MJX) gradient story.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.environments.interaction import reset, step
from ajax.environments.system_class import env_params_is_batched

# (carry, obs (B, obs_dim), resets (B,) bool) -> (action (B, act_dim), new_carry)
PolicyStepFn = Callable[[Any, jax.Array, jax.Array], Tuple[jax.Array, Any]]


@struct.dataclass
class Rollout:
    """Time-major closed-loop trajectory, leaves shaped ``(T, B, ...)``.

    ``obs[t]`` is what the policy saw at step ``t``, ``action[t]`` what it
    applied, ``reward[t]`` / ``done[t]`` / ``next_obs[t]`` the env's
    response. ``resets[t]`` is the episode-start flag the policy was given
    at step ``t`` (True at ``t=0`` and after a done). ``info`` is the env's
    per-step info dict with every leaf stacked over time (e.g. a tracking
    wrapper's desired output), for metrics that need more than the reward.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array
    resets: jax.Array
    info: Any = None


def with_transition_gradients(env: Any) -> Any:
    """Return a copy of ``env`` whose transitions preserve gradients.

    Walks Ajax's ``GymnaxWrapper`` chain (wrappers keep the inner env on
    ``_env``) so a wrapped env is re-wrapped around the gradient-enabled
    core. Raises ``ValueError`` when the core env does not implement
    differentiable transitions.
    """
    inner = getattr(env, "_env", None)
    if inner is not None:
        # Shallow-copy by hand: ``copy.copy`` probes ``__setstate__`` on the
        # half-built copy, which ``GymnaxWrapper.__getattr__`` forwards to
        # a not-yet-set ``_env`` and recurses forever.
        rewrapped = object.__new__(type(env))
        rewrapped.__dict__.update(env.__dict__)
        rewrapped.__dict__["_env"] = with_transition_gradients(inner)
        return rewrapped
    if not getattr(env, "supports_transition_gradients", False):
        raise ValueError(
            f"{type(env).__name__} does not support transition gradients; "
            "differentiable rollouts need a gymnax env implementing "
            "with_transition_gradients()."
        )
    if getattr(env, "transition_gradients_enabled", False):
        return env
    return env.with_transition_gradients()


def closed_loop_rollout(
    policy_step: PolicyStepFn,
    policy_carry: Any,
    rng: jax.Array,
    env: Any,
    env_params: Any,
    horizon: int,
    n_envs: Optional[int] = None,
) -> Tuple[Rollout, Any]:
    """Run ``horizon`` closed-loop steps with gradients flowing end to end.

    Args:
        policy_step: ``(carry, obs, resets) -> (action, new_carry)``. Called
            once per step on the batch of ``B`` envs. ``resets`` is a
            ``(B,)`` bool array marking episode starts so recurrent
            policies can clear their memory.
        policy_carry: initial policy carry (any pytree, ``None`` for a
            memoryless policy).
        rng: PRNG key; split into per-env reset and step keys.
        env: gymnax env, already configured with
            :func:`with_transition_gradients` when gradients are wanted.
        env_params: unbatched ``EnvParams`` (broadcast to every env) or
            batched ones from a :class:`~ajax.environments.system_class.
            SystemClass` (one system per env, ``B`` inferred from them).
        horizon: number of steps.
        n_envs: number of parallel envs; required when ``env_params`` is
            unbatched, inferred (and must match) otherwise.

    Returns:
        ``(rollout, final_carry)``; ``rollout`` leaves are ``(horizon, B, ...)``.
    """
    if env_params_is_batched(env_params):
        batch = int(jax.tree.leaves(env_params)[0].shape[0])
        if n_envs is not None and n_envs != batch:
            raise ValueError(
                f"n_envs={n_envs} but env_params carry a batch of {batch} systems"
            )
    elif n_envs is None:
        raise ValueError("n_envs is required when env_params are unbatched")
    else:
        batch = n_envs

    reset_key, step_key = jax.random.split(rng)
    obs, env_state = reset(
        jax.random.split(reset_key, batch), env, "gymnax", env_params
    )
    step_keys = jax.random.split(step_key, horizon)
    init_resets = jnp.ones((batch,), dtype=bool)

    def body(carry, key):
        obs, env_state, policy_carry, resets = carry
        action, policy_carry = policy_step(policy_carry, obs, resets)
        next_obs, env_state, reward, terminated, truncated, info = step(
            jax.random.split(key, batch), env_state, action, env, "gymnax", env_params
        )
        done = jnp.logical_or(terminated, truncated).astype(bool)
        out = Rollout(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=next_obs,
            resets=resets,
            info=info,
        )
        return (next_obs, env_state, policy_carry, done), out

    (_, _, final_carry, _), rollout = jax.lax.scan(
        body, (obs, env_state, policy_carry, init_resets), step_keys
    )
    return rollout, final_carry


__all__ = [
    "PolicyStepFn",
    "Rollout",
    "closed_loop_rollout",
    "with_transition_gradients",
]
