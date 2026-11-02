"""benchmarks/expert_guidance: the experiment's expert and its AdaStop
driver (PREREG.md there). No arm trains here but in the slow test, at a
tiny budget.

The expert's contract, from Pendulum-v1's physics and Astrom and Furuta
(2000): its torque stays within [-2, 2]; away from the top it moves the
energy E = thdot^2 / 6 + 5 cos(theta) towards the upright's 5, through
dE/dt = u thdot; near the top it pushes the pole back up; from the bottom
it swings the pole up and holds it within one 200-step episode.
"""

import guidance
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import EnvState
from refcmp_specs import COMPARISONS
from swing_up import SwingUpPD

EXPERT = SwingUpPD()


def observations(angle, velocity):
    angle, velocity = np.broadcast_arrays(
        np.asarray(angle, float), np.asarray(velocity, float)
    )
    return jnp.stack([np.cos(angle), np.sin(angle), velocity], -1), angle, velocity


def test_the_torque_stays_within_the_motors_limit():
    obs, _, _ = observations(
        *np.meshgrid(np.linspace(-np.pi, np.pi, 61), np.linspace(-8, 8, 33))
    )
    assert np.abs(EXPERT(obs)).max() <= 2.0


def test_away_from_the_top_it_moves_the_energy_towards_the_upright_s():
    grid = np.meshgrid(np.linspace(0.45, np.pi, 40), np.linspace(-8, 8, 41))
    obs, angle, velocity = observations(
        np.concatenate([grid[0], -grid[0]]), np.concatenate([grid[1], grid[1]])
    )
    torque = np.asarray(EXPERT(obs))[..., 0]
    energy = velocity**2 / 6 + 5 * np.cos(angle)
    moving = (np.abs(velocity) > 0.01) & (np.abs(5 - energy) > 0.01)
    # dE/dt = u thdot has the sign of 5 - E: up below the upright's, down above.
    assert (np.sign(torque * velocity)[moving] == np.sign(5 - energy)[moving]).all()


def test_near_the_top_it_pushes_the_pole_back_up():
    obs, _, _ = observations([0.2, -0.2, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0])
    torque = np.asarray(EXPERT(obs))[..., 0]
    assert torque[0] < 0 < torque[1]  # back towards 0 from either side
    assert torque[2] < 0  # against a pole moving away
    assert torque[3] == 0  # upright at rest: nothing to do


def test_from_the_bottom_it_swings_up_and_holds_the_pole():
    env, params = gymnax.make("Pendulum-v1")
    state = EnvState(time=0, theta=jnp.pi, theta_dot=0.0, last_u=0.0)
    obs = env.get_obs(state)
    angles = []
    for key in jax.random.split(jax.random.PRNGKey(0), 200):
        obs, state, *_ = env.step_env(key, state, EXPERT(obs), params)
        angles.append(float(jnp.arctan2(obs[1], obs[0])))
    # Up within the episode, and still up over its last quarter.
    assert np.abs(angles[-50:]).max() < 0.1


def test_the_measured_return_sums_pendulum_costs():
    returns = guidance.expert_returns(n_episodes=8)
    worst = -200 * (np.pi**2 + 0.1 * 8**2 + 0.001 * 2**2)  # every step at its worst
    assert returns.shape == (8,) and (returns <= 0).all() and (returns >= worst).all()


def test_the_arms_run_the_sac_comparisons_matched_settings():
    # Version 2 of that spec, as registered, has version 1's settings.
    assert guidance.SETTINGS == COMPARISONS["SAC"].matched


@pytest.mark.parametrize("arm", list(guidance.ARMS))
def test_every_arm_builds_its_agent(arm):
    agent = guidance.build(arm)
    assert type(agent).__name__ == "SAC"
    assert guidance.N_TIMESTEPS % guidance.EVAL_EVERY == 0


def test_steps_to_solved_count_to_the_first_evaluation_at_minus_200():
    assert guidance.first_solved([-300.0, -199.0, -250.0]) == 2 * guidance.EVAL_EVERY
    assert np.isnan(guidance.first_solved([-300.0, -201.0]))


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(guidance, "RECORDS", tmp_path / "results.jsonl")
    return tmp_path / "results.jsonl"


def fake_train(trained):
    """A1 scores far above the rest, which all score alike."""

    def train(arm, seeds):
        trained.append((arm, tuple(seeds)))
        value = -100.0 if arm == "A1" else -500.0
        return {
            "curve": [[value] * 20] * len(seeds),
            "expert_curve": [[-185.0] * 20] * len(seeds),
        }

    return train


def test_the_driver_trains_what_adastop_asks_and_reuses_the_records(records):
    trained = []
    test, batches = guidance.decide(train=fake_train(trained))
    decisions = test.decisions
    assert decisions[("A1", "A0")] == "larger"
    assert {decisions[(arm, "A0")] for arm in ("A2", "A3", "A4")} == {"equal"}
    # A0 runs every interim; A1 stops once its comparison is decided.
    assert len(batches["A0"]) == guidance.K > len(batches["A1"])
    assert len(trained) == len(set(trained)) == len(records.read_text().splitlines())

    def refuse(arm, seeds):
        raise AssertionError("a recorded batch trained again")

    again, _ = guidance.decide(train=refuse)
    assert again.decisions == decisions


def test_the_report_reads_recorded_batches_only(records):
    with pytest.raises(LookupError, match="not recorded"):
        guidance.decide(train=None)


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(guidance.ARMS))
def test_every_arm_trains_and_logs_its_evaluations_and_the_experts(arm):
    batch = guidance.train(arm, [0, 1], n_timesteps=400, eval_every=200)
    assert np.asarray(batch["curve"]).shape == (2, 2)
    assert (
        np.isfinite(batch["curve"]).all() and np.isfinite(batch["expert_curve"]).all()
    )
