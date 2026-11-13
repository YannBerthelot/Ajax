"""Hyperparameter configurations trained in one program (ActorCritic.train's
sweep, ajax.agents.sweep), on SAC (off-policy, evaluating between chunks of
training) and PPO (on-policy, one jitted scan).

Contract (owner's design decisions of 2026-11-13 and train's docstring):

- each member, a configuration with a seed, trains as a separate run of that
  configuration with that seed does, up to float32 rounding;
- members run configuration-major, as agent.members lists them, and a dict
  sweep trains its grid;
- every field an agent declares sweepable reaches its training;
- other fields, and SAC's gamma with an n-step replay (which sums rewards
  with the gamma it was built with), are refused before training;
- a sweep resumed from its state continues as the uninterrupted sweep.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, SAC, TDMPC2
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY

RELU = ("16", "relu", "16", "relu")
TANH = ("16", "tanh", "16", "tanh")
#: A separate run folds its Python numbers in double precision before XLA sees
#: them (1 - tau, gamma * gae_lambda); a member's are float32 throughout. The
#: last-bit differences this leaves (6e-8 measured after about 100 updates of
#: these networks) stay well under 1e-6, while the configurations below move
#: the parameters by 1e-3 to 1e-2.
ROUNDING = 1e-6
#: The logged evaluations: returns summed over up to 200 steps of rewards
#: computed from those parameters, so the same rounding, a few hundred times
#: over (Pendulum's episode returns are of order 1e3).
LOGGED_ROUNDING = 1e-3


def tiny_sac(**fields: Any) -> SAC:
    return SAC(
        "Pendulum-v1",
        n_envs=1,
        buffer_size=128,
        batch_size=8,
        learning_starts=16,
        policy_update_start=16,
        alpha_update_start=16,
        actor_architecture=RELU,
        critic_architecture=RELU,
        **fields,
    )


def tiny_ppo(**fields: Any) -> PPO:
    return PPO(
        "CartPole-v1",
        n_envs=2,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        actor_architecture=TANH,
        critic_architecture=TANH,
        **fields,
    )


def floating_leaves(state: Any) -> dict[str, jax.Array]:
    """Everything training writes, by its path in the state: networks,
    optimisers, normalisers, replay. A member's optimiser state also holds
    its learning rate, which a separate run's does not."""
    leaves = jax.tree_util.tree_leaves_with_path(state)
    return {
        jax.tree_util.keystr(path): leaf
        for path, leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def assert_rows_match(swept: Any, rows: slice, separate: Any) -> None:
    ours, theirs = floating_leaves(swept), floating_leaves(separate)
    assert set(theirs).issubset(ours)
    for path, leaf in theirs.items():
        np.testing.assert_allclose(
            ours[path][rows], leaf, atol=ROUNDING, rtol=0, err_msg=path
        )


def assert_members_train_as_separate_runs(
    make: Callable[..., Any],
    sweep: Any,
    configurations: list[dict[str, float]],
    n_timesteps: int,
    logging_config: Any = None,
) -> None:
    seeds = [0, 1]
    agent = make()
    state, out = agent.train(
        seed=seeds, n_timesteps=n_timesteps, sweep=sweep, logging_config=logging_config
    )
    expected = []
    for configuration in configurations:
        for seed in seeds:
            row: dict[str, float] = {"seed": seed}
            row.update(configuration)
            expected.append(row)
    assert agent.members == expected
    for k, configuration in enumerate(configurations):
        rows = slice(k * len(seeds), (k + 1) * len(seeds))
        separate, separate_out = make(**configuration).train(
            seed=seeds, n_timesteps=n_timesteps, logging_config=logging_config
        )
        assert_rows_match(state, rows, separate)
        if logging_config is not None:
            for name, logged in separate_out.items():
                np.testing.assert_allclose(
                    np.asarray(out[name])[rows], logged, atol=LOGGED_ROUNDING, rtol=0
                )


def test_sac_members_train_and_evaluate_as_their_separate_runs() -> None:
    """Catches a member reading another's values, or the base agent's: the
    learning rate, closed over by the optimiser one chunk builds and the
    next steps, and gamma, which the evaluation's value estimate reads."""
    configurations = [
        {"actor_learning_rate": 1e-3, "gamma": 0.98, "tau": 0.01, "alpha_init": 0.5},
        {"actor_learning_rate": 3e-4, "gamma": 0.9, "tau": 0.05, "alpha_init": 1.0},
    ]
    evaluations = LoggingConfig(log_frequency=32, backends=())
    assert_members_train_as_separate_runs(
        tiny_sac, configurations, configurations, 64, evaluations
    )


def test_ppo_trains_a_dict_sweeps_grid_configuration_major() -> None:
    """Catches members ordered seed-major, or a grid built in another order:
    each block of rows must be its configuration's separate run. gamma
    reaches GAE, which once took it as a compile-time constant."""
    grid = {"actor_learning_rate": [3e-4, 1e-3], "gamma": [0.9]}
    configurations = [
        {"actor_learning_rate": 3e-4, "gamma": 0.9},
        {"actor_learning_rate": 1e-3, "gamma": 0.9},
    ]
    assert_members_train_as_separate_runs(tiny_ppo, grid, configurations, 128)


#: A value off each sweepable field's default; a field joins sweepable with
#: its row here.
SAC_CHANGES = {
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "alpha_learning_rate": 1e-3,
    "alpha_init": 0.5,
    "target_entropy_per_dim": -0.5,
    "gamma": 0.9,
    "tau": 0.05,
    "reward_scale": 2.0,
}
PPO_CHANGES = {
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "gamma": 0.9,
    "gae_lambda": 0.5,
    "ent_coef": 0.1,
    "vf_coef": 2.0,
    "clip_range": 0.001,  # small enough to clip in the second epoch
}


@pytest.mark.parametrize(
    "make, changes",
    [(tiny_sac, SAC_CHANGES), (tiny_ppo, PPO_CHANGES)],
    ids=["SAC", "PPO"],
)
def test_every_sweepable_field_reaches_the_training(
    make: Callable[..., Any], changes: dict[str, float]
) -> None:
    """Catches a field captured when the agent is built, which every member
    would train with at its base value: the member changing it alone must
    train otherwise than the base member."""
    agent = make()
    assert set(changes) == agent.sweepable
    base = {name: getattr(agent, name) for name in changes}
    configurations = [base]
    for name, value in changes.items():
        changed = dict(base)
        changed[name] = value
        configurations.append(changed)
    state, _ = agent.train(seed=0, n_timesteps=64, sweep=configurations)
    leaves = list(floating_leaves(state).values())
    for row, name in enumerate(changes, start=1):
        # equal_nan: NaN fills a metrics slot no update wrote yet, in every row.
        moved = [
            not np.allclose(leaf[row], leaf[0], atol=ROUNDING, equal_nan=True)
            for leaf in leaves
        ]
        assert any(moved), f"{name} changed nothing: the member kept the base value"


@pytest.mark.parametrize(
    "make, sweep, refused",
    [
        (tiny_sac, {"batch_size": [8, 16]}, "batch_size"),
        (tiny_sac, {"tau": [0.01], "learning_rate": [1e-3]}, "learning_rate"),
        (lambda: tiny_sac(n_step=2), {"gamma": [0.9, 0.99]}, "gamma"),
        (tiny_ppo, {"n_epochs": [2, 4]}, "n_epochs"),
        (tiny_sac, {"tau": ["0.01"]}, "numbers"),
        (tiny_sac, [{"tau": 0.01}, {"gamma": 0.9}], "same fields"),
        (tiny_sac, {}, "setting a field"),
        # An agent with its own replay refuses before preparing it.
        (lambda: TDMPC2("Pendulum-v1", **TDMPC2_TINY), {"tau": [0.01]}, "none"),
    ],
    ids=[
        "shape",
        "unknown",
        "n-step-gamma",
        "loop-length",
        "text",
        "mixed",
        "empty",
        "no-sweepable-field",
    ],
)
def test_a_sweep_outside_the_sweepable_numbers_is_refused_before_training(
    make: Callable[[], Any], sweep: Any, refused: str
) -> None:
    """Catches a sweep that would compile one program per value, or train
    every member with a value its replay or loop fixed when built."""
    with pytest.raises(ValueError, match=refused):
        make().train(seed=0, n_timesteps=16, sweep=sweep)


def test_a_resumed_sweep_continues_as_the_uninterrupted_sweep() -> None:
    """Catches a resume that drops the members' values (each would continue
    with the base agent's) or reorders them; a state of other members is
    refused."""
    sweep = {"tau": [0.005, 0.05], "actor_learning_rate": [1e-3]}
    whole, _ = tiny_sac().train(seed=0, n_timesteps=64, sweep=sweep)
    agent = tiny_sac()
    half, _ = agent.train(seed=0, n_timesteps=32, sweep=sweep, total_timesteps=64)
    with pytest.raises(ValueError, match="same seed and sweep"):
        agent.train(seed=[0, 1], n_timesteps=32, sweep=sweep, initial_state=half)
    resumed, _ = agent.train(
        seed=0, n_timesteps=32, sweep=sweep, total_timesteps=64, initial_state=half
    )
    assert_rows_match(resumed, slice(None), whole)
