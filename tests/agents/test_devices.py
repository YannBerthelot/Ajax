"""``train(devices=...)``: a run's members split over devices.

Contract (docs/guide/performance.md, ``ActorCritic.train``):
  - the members are split into consecutive slices, one per device, sizes
    differing by one at most; fewer members than devices use fewer devices;
  - each member trains as it does on one device, but for rounding, and the
    results come back in the members' order, on the first device;
  - a fresh run, a sweep and a resumed run all split;
  - a Gymnasium environment, an evaluation_device and an empty list are
    refused.

CI has one device, so the split is over that device twice: what each slice
receives and how the results are gathered do not depend on the devices being
distinct. Two real GPUs are checked by hand (the PR says where).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.devices import member_slices, train_on_devices
from ajax.agents.PPO.PPO import PPO
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.TDMPC2.test_TDMPC2 import PENDULUM_PARAMS, PENDULUM_S, TINY
from tests.probing import readouts

CPU = jax.devices("cpu")[0]

# A split run's slices and the whole run are separately compiled programs,
# of different batch sizes: they may round apart (1.67e-6 seen between two
# programs on CI's runners). A member given another member's seed or
# configuration differs by 1e-2 or more in its weights.
ROUNDING = {"rtol": 1e-4, "atol": 1e-4}


@pytest.mark.parametrize(
    "n_members, n_devices, sizes",
    [
        (1, 1, [1]),
        (4, 2, [2, 2]),
        (5, 2, [3, 2]),
        (7, 3, [3, 2, 2]),
        (2, 4, [1, 1]),
        (3, 1, [3]),
    ],
)
def test_members_split_into_consecutive_near_equal_slices(n_members, n_devices, sizes):
    slices = member_slices(n_members, n_devices)
    assert [s.stop - s.start for s in slices] == sizes
    covered = [row for s in slices for row in range(s.start, s.stop)]
    assert covered == list(range(n_members))


@pytest.mark.parametrize("n_members, n_devices", [(0, 2), (3, 0)])
def test_no_member_or_no_device_is_refused(n_members, n_devices):
    with pytest.raises(ValueError, match="at least 1"):
        member_slices(n_members, n_devices)


def test_each_device_trains_its_slice_and_results_keep_the_members_order():
    """Five members on two devices: the calls receive rows 0-2 and 3-4 of
    every batched argument, and the results are the members' in order."""
    seen = []

    def train(x, tree):
        seen.append(np.asarray(x).tolist())
        return {"double": 2 * x, "sum": x + tree["y"]}

    x = jnp.arange(5.0)
    out = train_on_devices(train, (x, {"y": 10 * x}), [CPU, CPU])
    assert sorted(seen) == [[0.0, 1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_array_equal(out["double"], [0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(out["sum"], [0.0, 11.0, 22.0, 33.0, 44.0])
    assert out["double"].devices() == {CPU}


def test_a_failure_on_one_device_is_raised():
    def train(x):
        if float(x[0]) == 1.0:
            raise RuntimeError("this slice failed")
        return x

    with pytest.raises(RuntimeError, match="this slice failed"):
        train_on_devices(train, (jnp.arange(2.0),), [CPU, CPU])


def tiny_ppo() -> PPO:
    return PPO("Pendulum-v1", n_envs=2, n_steps=8, n_epochs=1)


def test_seeds_split_over_devices_train_as_on_one_and_resume():
    """Three seeds, alone then split 2 + 1; then both resumed for a second
    leg, the split one split again."""
    whole, _ = tiny_ppo().train(
        seed=[0, 1, 2], n_timesteps=32, total_timesteps=64, logging_config=None
    )
    split, _ = tiny_ppo().train(
        seed=[0, 1, 2],
        n_timesteps=32,
        total_timesteps=64,
        devices=[CPU, CPU],
        logging_config=None,
    )
    assert not readouts.differing_leaves(whole, split, **ROUNDING)

    whole, _ = tiny_ppo().train(
        seed=[0, 1, 2], n_timesteps=32, initial_state=whole, logging_config=None
    )
    split, _ = tiny_ppo().train(
        seed=[0, 1, 2],
        n_timesteps=32,
        initial_state=split,
        devices=[CPU, CPU],
        logging_config=None,
    )
    np.testing.assert_array_equal(split.collector_state.timestep, 64)
    assert not readouts.differing_leaves(whole, split, **ROUNDING)


def test_a_sweep_split_over_devices_trains_each_member_as_on_one():
    """Two learning rates over two seeds: four members, two per device, the
    second device's holding only the second configuration."""

    def agent() -> SAC:
        return SAC(
            "Pendulum-v1",
            n_envs=1,
            buffer_size=128,
            batch_size=8,
            learning_starts=16,
            policy_update_start=16,
            alpha_update_start=16,
        )

    sweep = {"actor_learning_rate": [1e-3, 1e-2]}
    whole, _ = agent().train(
        seed=[0, 1], n_timesteps=48, sweep=sweep, logging_config=None
    )
    split_agent = agent()
    split, _ = split_agent.train(
        seed=[0, 1],
        n_timesteps=48,
        sweep=sweep,
        devices=[CPU, CPU],
        logging_config=None,
    )
    assert len(split_agent.members) == 4
    assert not readouts.differing_leaves(whole, split, **ROUNDING)


def test_an_agent_with_its_own_replay_splits_its_seeds():
    """TD-MPC2 and DreamerV3 size their replay for the call before training
    (their own ``train``): three seeds split 2 + 1 train as on one device."""

    def agent() -> TDMPC2:
        return TDMPC2(
            "Pendulum-v1",
            env_params=PENDULUM_PARAMS,
            n_envs=2,
            seed_steps=PENDULUM_S,
            **TINY,
        )

    options = {
        "seed": [0, 1, 2],
        "n_timesteps": 52,
        "num_episode_test": 2,
        "logging_config": None,
    }
    whole, _ = agent().train(**options)
    split, _ = agent().train(**options, devices=[CPU, CPU])
    assert not readouts.differing_leaves(whole, split, **ROUNDING)


def test_devices_refuses_what_it_cannot_split():
    with pytest.raises(ValueError, match="devices is empty"):
        tiny_ppo().train(seed=0, n_timesteps=16, devices=[], logging_config=None)
    scored_elsewhere = LoggingConfig(
        run_name="run", log_frequency=16, backends=(), evaluation_device=CPU
    )
    with pytest.raises(ValueError, match="no evaluation_device"):
        tiny_ppo().train(
            seed=0, n_timesteps=16, devices=[CPU], logging_config=scored_elsewhere
        )
    in_python = PPO("gymnasium:CartPole-v1", n_envs=1, n_steps=8, n_epochs=1)
    with pytest.raises(ValueError, match="takes no devices"):
        in_python.train(seed=0, n_timesteps=16, devices=[CPU], logging_config=None)
