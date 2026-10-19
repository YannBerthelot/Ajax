"""Imitation agents on a known answer: DAgger.

DAgger (Ross, Gordon and Bagnell, 2011) labels with the expert's action the
states the learner visits. On a one-hot chain, from state 0 an action above
1/2 leads to 1 then 2, any other to 3; the expert takes 0.8 there, so its own
path never shows state 3. The actor starts near 0, below 1/2: it reaches 3
first, its labels then lift a(0) past 1/2, and the path 0 -> 1 -> 2 gets
labelled too. Cloning on the expert's states never labels 3, and labels read
from the executed action leave the actor where it started (with a first
iteration of 500 steps, the default, a(3) only drifts, to 0.26). The discrete
twin, two actions, starts at 0 or 3 at random: a categorical actor's first
choice cannot be steered, so it reads the cross-entropy fit on every state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from . import agents, envs, runs
from . import readouts as R
from .verdict import Case, Query, check, params

LABELS = (0.8, 0.4, -0.2, 0.6)  # the expert's action at states 0-3
CHOICES = (1, 0, 1, 0)  # the discrete expert's


def chain(actions: int = 0) -> envs.Spec:
    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        a = a.reshape(-1)[0]
        nxt = jnp.select([st.s == 0, st.s == 1], [jnp.where(a > 0.5, 1, 3), 2], 0)
        return st.replace(s=nxt), 0.0, (st.s == 2) | (st.s == 3)

    def reset(key: jax.Array, params: Any) -> tuple:
        return (3 * jax.random.bernoulli(key) if actions else 0), 0.0

    obs = lambda st: envs.one_hot(st.s, 4)  # noqa: E731
    return envs.Spec(
        "dagger_chain", transition, obs, reset, 4, actions=actions, limit=4
    )


def expert(obs: jax.Array) -> jax.Array:
    return obs @ jnp.asarray(LABELS)[:, None]


def chooser(obs: jax.Array) -> jax.Array:
    return jnp.asarray(CHOICES)[jnp.argmax(obs, -1)][..., None]


def dagger_readings(discrete: bool) -> Any:
    def build() -> Any:
        env, params = chain(2 * discrete).make()
        kw: dict[str, Any] = {"expert_policy": chooser if discrete else expert}
        kw |= {"n_envs": 4, "iteration_steps": 4}  # beta 1 on the first tick only
        return agents.make("DAgger", env, params, **kw)

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets, i: int) -> jax.Array:
            if discrete:  # the probability of the expert's action
                return R.pi(n, envs.one_hot(i, 4)).prob(CHOICES[i]).reshape(())
            return R.action(n, envs.one_hot(i, 4))

        def actions(n: R.Nets) -> dict:
            return {f"a({i})": at(n, i) for i in (0, 1, 3)}

        return R.per_seed(actions, R.nets(run.state))

    return runs.readings(build, read)


_SELF = {"labels from the executed action": 0.0}
CASES: dict[str, Case] = {
    "dagger-off-path-DAgger": Case(
        "dagger-off-path-DAgger",
        (
            Query("a(0)", LABELS[0], _SELF),
            Query("a(1)", LABELS[1], _SELF),
            Query("a(3)", LABELS[3], _SELF | {"no label off the expert's path": 0.0}),
        ),
        dagger_readings(False),
        500,
        (0.057, 0.022, 0.071),
        note="ladder 250-4000 on 1000-1031, 2000-2031: 500; cert 32/32",
    ),
    "dagger-discrete-DAgger": Case(
        "dagger-discrete-DAgger",
        tuple(Query(f"a({i})", 1.0, {"untrained": 0.5}) for i in (0, 1, 3)),
        dagger_readings(True),
        500,
        (0.064, 0.098, 0.048),
        note="ladder 250-4000 on 1000-1031, 250-2000 on 2000-2031: 500; cert 32/32",
    ),
}
ANSWER_DIGEST = "cda15133aef7"


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
