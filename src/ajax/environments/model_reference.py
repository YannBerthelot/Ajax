"""Model-reference tracking tasks on top of any gymnax environment.

Model-reference control (Busetto et al. 2024, arXiv:2411.06482, §2)
asks a controller to make a plant's measured output ``y_k`` follow the
output ``y^d_k`` of a *reference model* ``M`` driven by a user reference
``r_k``. This module turns any gymnax env into such a task:

* :class:`StepReference` is the reference distribution ``p(R)``:
  piecewise-constant random steps of random amplitude and duration.
* :class:`LinearReferenceModel` is a discrete-time linear ``M``
  (``x_{k+1} = A x_k + B r_k``, ``y^d_k = C x_k + D r_k``); the paper's
  first-order model is :meth:`LinearReferenceModel.first_order`.
* :class:`ModelReferenceWrapper` exposes the *controller's* view of the
  problem: observation ``[e_k, u_{k-1}]`` (tracking error and previous
  input, optionally followed by the raw plant observation) and reward
  ``-||y^d_k - y_k||^2``, so the undiscounted return is the negative
  closed-loop matching cost of eq. (9) and the paper's M-RMSE is
  ``sqrt(-return / horizon)``.

Every operation is plain ``jnp`` so the wrapper is differentiable when
the inner env is (see :mod:`ajax.environments.differentiable`), and it
composes with :mod:`ajax.environments.system_class` batched params
because the wrapper is per-env (Ajax vmaps ``reset``/``step``).
"""

from __future__ import annotations

from math import prod
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from ajax.wrappers import GymnaxWrapper


@struct.dataclass
class StepReference:
    """Random piecewise-constant references ``r_{[0, N-1]}``.

    Each segment holds a value drawn uniformly in ``[min_value,
    max_value]`` for a duration drawn uniformly in ``[min_duration,
    max_duration]`` steps (both inclusive), independently per output.
    The paper's evaporator case uses values in ``[20, 25]`` % and
    durations in ``[20, 50]`` s at 1 s sampling over ``N = 100``.
    """

    horizon: int = struct.field(pytree_node=False)
    min_value: float
    max_value: float
    min_duration: int = struct.field(pytree_node=False)
    max_duration: int = struct.field(pytree_node=False)
    n_outputs: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not 1 <= self.min_duration <= self.max_duration:
            raise ValueError(
                "need 1 <= min_duration <= max_duration, got "
                f"{self.min_duration} and {self.max_duration}"
            )

    def sample(self, rng: jax.Array) -> jax.Array:
        """One reference sequence, shape ``(horizon, n_outputs)``."""

        def body(carry, key):
            remaining, value = carry
            k_val, k_dur = jax.random.split(key)
            new_value = jax.random.uniform(
                k_val, (self.n_outputs,), minval=self.min_value, maxval=self.max_value
            )
            new_duration = jax.random.randint(
                k_dur, (self.n_outputs,), self.min_duration, self.max_duration + 1
            )
            start_new = remaining <= 0
            value = jnp.where(start_new, new_value, value)
            remaining = jnp.where(start_new, new_duration, remaining) - 1
            return (remaining, value), value

        init = (jnp.zeros((self.n_outputs,), jnp.int32), jnp.zeros((self.n_outputs,)))
        _, sequence = jax.lax.scan(body, init, jax.random.split(rng, self.horizon))
        return sequence


@struct.dataclass
class LinearReferenceModel:
    """Discrete-time linear reference model ``M``.

    ``x_{k+1} = A x_k + B r_k``, ``y^d_k = C x_k + D r_k`` with
    ``A: (nx, nx)``, ``B: (nx, ny)``, ``C: (ny, nx)``, ``D: (ny, ny)``.
    """

    A: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array

    @classmethod
    def first_order(
        cls,
        a: float = 0.4286,
        b: float = 0.7143,
        c: float = 0.5669,
        d: float = 0.2914,
        n_outputs: int = 1,
    ) -> "LinearReferenceModel":
        """Diagonal first-order model, one independent channel per output.

        Defaults are the paper's eq. (12): ``x_{k+1} = 0.4286 x_k + 0.7143
        r_k``, ``y^d_k = 0.5669 x_k + 0.2914 r_k`` (unit DC gain).
        """
        eye = jnp.eye(n_outputs)
        return cls(A=a * eye, B=b * eye, C=c * eye, D=d * eye)

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    def init_state(self, y0: jax.Array, r0: jax.Array) -> jax.Array:
        """State whose output equals the plant's initial output.

        The paper initialises ``x^M_0`` at the plant's initial condition
        so the desired trajectory starts where the plant starts. Solves
        ``C x = y0 - D r0`` in the least-squares sense.
        """
        return jnp.linalg.pinv(self.C) @ (y0 - self.D @ r0)

    def step(self, x: jax.Array, r: jax.Array) -> jax.Array:
        return self.A @ x + self.B @ r

    def output(self, x: jax.Array, r: jax.Array) -> jax.Array:
        return self.C @ x + self.D @ r


@struct.dataclass
class ModelReferenceState:
    env_state: Any
    reference: jax.Array  # (horizon, n_y)
    model_state: jax.Array  # (nx,)
    u_prev: jax.Array  # (n_u,)
    t: jax.Array  # scalar int32, index of the CURRENT reference sample


class ModelReferenceWrapper(GymnaxWrapper):
    """Turn a gymnax env into a model-reference tracking task.

    Args:
        env: gymnax env (the plant).
        reference: reference distribution ``p(R)``.
        model: reference model ``M``.
        output_idx: observation dims forming the measured output ``y``.
            Ignored when ``output_fn`` is given.
        output_fn: ``obs -> y`` (shape ``(n_y,)``); use it when the
            output is not a plain slice of the observation (e.g. an angle
            recovered from ``(cos, sin)``).
        include_raw_obs: append the plant observation to ``[e, u_prev]``.
        action_dim: dimension of the input ``u`` (needed for ``u_prev``).

    The reference sequence is re-sampled whenever the plant episode ends
    (the inner env's auto-reset). Past the reference horizon the last
    sample is held; size ``reference.horizon`` to the episode length.
    """

    def __init__(
        self,
        env: Any,
        reference: StepReference,
        model: LinearReferenceModel,
        output_idx: Sequence[int] = (0,),
        output_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
        include_raw_obs: bool = False,
        action_dim: int = 1,
    ):
        super().__init__(env)
        if reference.n_outputs != model.n_outputs:
            raise ValueError(
                "reference.n_outputs and model.n_outputs differ: "
                f"{reference.n_outputs} vs {model.n_outputs}"
            )
        self.reference = reference
        self.model = model
        self._output_idx = jnp.asarray(tuple(output_idx), dtype=jnp.int32)
        self._output_fn = output_fn
        self.include_raw_obs = include_raw_obs
        self.action_dim = action_dim

    # ---------------------------------------------------------------- output
    def output(self, obs: jax.Array) -> jax.Array:
        """Measured output ``y`` of the plant, shape ``(n_y,)``."""
        if self._output_fn is not None:
            return jnp.reshape(self._output_fn(obs), (-1,))
        return obs[self._output_idx]

    def _observation(
        self, error: jax.Array, u_prev: jax.Array, raw_obs: jax.Array
    ) -> jax.Array:
        parts = [error, u_prev]
        if self.include_raw_obs:
            parts.append(jnp.reshape(raw_obs, (-1,)))
        return jnp.concatenate(parts, axis=-1)

    def observation_space(self, params) -> spaces.Box:
        raw_dim = prod(self._env.observation_space(params).shape)
        dim = self.model.n_outputs + self.action_dim
        if self.include_raw_obs:
            dim += raw_dim
        return spaces.Box(-jnp.inf, jnp.inf, (dim,), jnp.float32)

    # ------------------------------------------------------------ transitions
    def reset(self, key, params=None):
        k_env, k_ref = jax.random.split(key)
        raw_obs, env_state = self._env.reset(k_env, params)
        state = self._fresh_task(k_ref, raw_obs, env_state)
        return self.get_obs(state, params), state

    def _fresh_task(self, key, raw_obs, env_state) -> ModelReferenceState:
        reference = self.reference.sample(key)
        y0 = self.output(raw_obs)
        return ModelReferenceState(
            env_state=env_state,
            reference=reference,
            model_state=self.model.init_state(y0, reference[0]),
            u_prev=jnp.zeros((self.action_dim,), jnp.float32),
            t=jnp.asarray(0, jnp.int32),
        )

    def step(self, key, state: ModelReferenceState, action, params=None):
        k_env, k_ref = jax.random.split(key)
        raw_obs, env_state, _, terminated, truncated, info = self._env.step(
            k_env, state.env_state, action, params
        )
        y = self.output(raw_obs)
        r_now = state.reference[state.t]
        t_next = jnp.minimum(state.t + 1, self.reference.horizon - 1)
        model_state = self.model.step(state.model_state, r_now)
        y_desired = self.model.output(model_state, state.reference[t_next])
        reward = -jnp.sum((y_desired - y) ** 2)

        continued = ModelReferenceState(
            env_state=env_state,
            reference=state.reference,
            model_state=model_state,
            u_prev=jnp.reshape(jnp.asarray(action, jnp.float32), (self.action_dim,)),
            t=t_next,
        )
        # The plant auto-reset on done: start a fresh task on its new obs.
        done = jnp.logical_or(terminated, truncated)
        fresh = self._fresh_task(k_ref, raw_obs, env_state)
        state = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), fresh, continued)
        info = {**info, "y": y, "y_desired": y_desired, "r": r_now}
        return self.get_obs(state, params), state, reward, terminated, truncated, info

    def get_obs(self, state: ModelReferenceState, params=None, key=None):
        """Controller observation ``[r_t - y_t, u_{t-1} (, raw obs)]``."""
        raw_obs = self._env.get_obs(state.env_state, params, key)
        error = state.reference[state.t] - self.output(raw_obs)
        return self._observation(error, state.u_prev, raw_obs).astype(jnp.float32)


__all__ = [
    "LinearReferenceModel",
    "ModelReferenceState",
    "ModelReferenceWrapper",
    "StepReference",
]
