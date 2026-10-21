"""Every recurrent agent on the shared loop restarts its actor's memory at
each episode start in collection, for every memory kind, and its next action
reads that memory.

On two envs of the identity task (tests/environments/identity_env.py), an
episode's observations [0, c], ..., [t - 1, c] are known from the one it
shows next, [t, c]. With the actor's parameters held fixed (no update
runs: the off-policy agents stay in their warm-up, PPO steps at learning
rate 0), the live carry a run ends with must be the actor's run from a
fresh carry over the current episode's observations alone: every earlier
episode, ended by a termination or a time limit, is forgotten.

APG's controller has its own test (tests/agents/APG/test_episode_ends.py),
PPO's forced reset too (tests/agents/PPO/test_PPO.py), and the row
collector's ``is_first`` its (tests/environments/test_row_collector.py).
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig, zeros_carry_like
from tests.environments.identity_env import IdentityEnv

NET = ("16", "relu")
OFF_POLICY: dict[str, Any] = {"actor_architecture": NET, "critic_architecture": NET}
OFF_POLICY |= {"learning_starts": 10_000, "batch_size": 8, "buffer_size": 128}
OFF_POLICY |= {"burn_in": 2, "sequence_length": 4}
#: The agent's options with its parameters held fixed, and its budget.
AGENTS: dict[str, tuple[dict[str, Any], int]] = {
    "PPO": (
        {
            "actor_architecture": NET,
            "critic_architecture": NET,
            "n_steps": 8,
            "batch_size": 16,  # one minibatch: the whole rollout
            "n_epochs": 1,
            "actor_learning_rate": 0.0,
            "critic_learning_rate": 0.0,
        },
        32,
    ),
    "SAC": (OFF_POLICY, 32),
    "ASAC": (OFF_POLICY, 32),
    "REDQ": (
        OFF_POLICY | {"num_critics": 3, "subset_size": 2, "num_critic_updates": 2},
        32,
    ),
    "TD3": (OFF_POLICY, 32),
}
SEEDS = [0, 1, 2, 3]


def test_every_recurrent_agent_has_a_row() -> None:
    """A new agent taking memory joins AGENTS (APG is tested on its own)."""
    recurrent = {
        n for n in ajax.__all__ if getattr(getattr(ajax, n), "supports_memory", False)
    }
    assert recurrent - {"APG"} == set(AGENTS)


def _at(tree: Any, index: Any) -> Any:
    return jax.tree.map(lambda x: x[index], tree)


def _next_action(actor: Any, params: Any, obs: Any, carry: Any) -> Any:
    """The action sampled from ``carry`` on ``obs`` (one step, one env, no
    episode start), on a fixed key: equal carries draw equal actions."""
    goes_on = jnp.zeros((1, 1), bool)
    pi, _ = actor.apply(params, obs, hidden_state=carry, done=goes_on)
    return pi.sample(seed=jax.random.PRNGKey(0))


#: The fast tier's rows: every agent and every memory kind once (TD3's own
#: action pipeline with a carry holding flags); the others run in the slow
#: tier.
FAST = {("PPO", "gru"), ("SAC", "lstm"), ("ASAC", "mamba"), ("REDQ", "gru")}
FAST |= {("TD3", "transformer")}
ROWS = [
    pytest.param(n, k, id=f"{n}-{k}", marks=() if (n, k) in FAST else pytest.mark.slow)
    for n in sorted(AGENTS)
    for k in MEMORY_KINDS
]


@pytest.mark.parametrize("name, kind", ROWS)
def test_the_actor_remembers_only_its_current_episode(name: str, kind: str) -> None:
    """Read through the next action: a transformer's cache keeps the entries
    a restart masks out, so the carries themselves may differ."""
    options, budget = AGENTS[name]
    env = IdentityEnv()
    memory = MemoryConfig(kind=kind, hidden_size=8, window=8)
    agent = getattr(ajax, name)(
        env_id=env, env_params=env.default_params, n_envs=2, memory=memory, **options
    )
    state, _ = agent.train(seed=SEEDS, n_timesteps=budget)
    actor = state.actor_state
    shown = np.asarray(state.collector_state.last_obs)
    checked = 0
    for s in range(len(SEEDS)):
        live, params = _at(actor.hidden_state, s), _at(actor.params, s)
        for b in range(2):
            t, c = int(shown[s, b, 0]), shown[s, b, 1]
            if t == 0:  # the episode acted in ended: its observations are gone
                continue
            obs = jnp.stack([jnp.arange(t + 1.0), jnp.full(t + 1, c)], -1)[:, None]
            starts = jnp.zeros((t, 1), bool).at[0].set(True)
            fresh = zeros_carry_like(live, 1)
            _, replayed = actor.apply(params, obs[:-1], hidden_state=fresh, done=starts)
            got, want = (
                _next_action(actor, params, obs[-1:], h)
                for h in (_at(live, slice(b, b + 1)), replayed)
            )
            np.testing.assert_allclose(
                got, want, rtol=1e-5, atol=1e-5, err_msg=f"seed {s}, env {b}, t {t}"
            )
            checked += 1
    # Each env ran past its first episode (they last at most 5 steps).
    assert checked >= 2 and budget // 2 > 5, checked


@pytest.mark.parametrize("name, kind", ROWS)
def test_the_next_action_reads_the_carry(name: str, kind: str) -> None:
    """From a carry built over 4 steps of an episode, the actor's next action
    differs, bit for bit, from the zero carry's: the action head reads the
    memory's output, not the features before it. Untrained networks: a
    transformer or mamba changes its output little at initialisation, so the
    check is inequality, not a margin (equal bits mean an unread carry)."""
    options, _ = AGENTS[name]
    env = IdentityEnv()
    memory = MemoryConfig(kind=kind, hidden_size=8, window=8)
    agent = getattr(ajax, name)(
        env_id=env, env_params=env.default_params, n_envs=2, memory=memory, **options
    )
    state, _ = agent.train(seed=[0], n_timesteps=0)
    actor = state.actor_state
    params = _at(actor.params, 0)
    fresh = zeros_carry_like(_at(actor.hidden_state, 0), 1)
    t = 4
    obs = jnp.stack([jnp.arange(t + 1.0), jnp.full(t + 1, 0.7)], -1)[:, None]
    starts = jnp.zeros((t, 1), bool).at[0].set(True)
    _, carry = actor.apply(params, obs[:-1], hidden_state=fresh, done=starts)
    live = np.asarray(_next_action(actor, params, obs[-1:], carry))
    blank = np.asarray(_next_action(actor, params, obs[-1:], fresh))
    assert not np.array_equal(live, blank), (live, blank)
