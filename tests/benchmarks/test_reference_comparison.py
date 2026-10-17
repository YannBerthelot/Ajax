"""``benchmarks/reference``: every arm builds the agent it names and
evaluates inside its run, the matched arms copy the reference's shared
settings, the references' adapters run on gymnax 1.0, and the report's
verdicts. No comparison runs here: they take minutes (the README there).
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import refcmp_ajax
import refcmp_jax
import refcmp_report
import refcmp_sb3
from refcmp_specs import COMPARISONS

ARMS = [(name, arm) for name in COMPARISONS for arm in ("matched", "defaults")]


@pytest.mark.parametrize(("name", "arm"), ARMS)
def test_every_arm_builds_its_agent_and_evaluates_inside_the_run(name, arm):
    spec = COMPARISONS[name]
    agent = refcmp_ajax.build_agent(spec, arm)
    assert type(agent).__name__ == spec.agent
    if spec.eval_every:  # no evaluation slot may stay empty (NaN)
        assert spec.n_timesteps % spec.eval_every == 0
        n_steps = getattr(agent.agent_config, "n_steps", 1)
        assert spec.eval_every % (agent.env_args.n_envs * n_steps) == 0


@pytest.mark.parametrize("name", list(COMPARISONS))
def test_matched_arm_copies_every_setting_it_shares_with_the_reference(name):
    spec = COMPARISONS[name]
    pairs = []
    for key, value in spec.reference_config.items():
        ajax_keys = spec.renames.get(key, key)
        for ajax_key in (ajax_keys,) if isinstance(ajax_keys, str) else ajax_keys:
            if ajax_key in spec.matched:
                pairs.append((key, ajax_key))
                expected = value * spec.scale.get(key, 1)
                assert spec.matched[ajax_key] == expected, (key, ajax_key)
    renamed = {k for k, _ in pairs} & spec.renames.keys()
    assert renamed == spec.renames.keys(), "every rename names a copied setting"
    assert len(pairs) >= 3


@pytest.mark.parametrize(
    "name", [n for n, s in COMPARISONS.items() if s.reference == "sb3"]
)
def test_sb3_keywords_leave_the_harness_keys_to_the_runner(name):
    kwargs = refcmp_sb3.sb3_kwargs(COMPARISONS[name])
    harness = kwargs.pop("_harness")
    assert "net_arch" in harness
    assert not {"n_envs", "net_arch", "activation", "action_noise_std"} & kwargs.keys()


def test_ajax_runs_return_one_curve_per_seed():
    curve = refcmp_ajax.run(COMPARISONS["DQN"], "matched", [0, 1], 2_000, 1_000)
    assert np.asarray(curve).shape == (2, 2)
    # APG is scored on its final controller only, by the shared evaluator.
    curve = refcmp_ajax.run(COMPARISONS["APG"], "matched", [0], 2 * 64 * 32)
    assert np.asarray(curve).shape == (1, 1) and np.isfinite(curve).all()


def test_gymnax_as_brax_keeps_transition_gradients():
    env = refcmp_jax.GymnaxAsBrax("Pendulum-v1")
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (env.observation_size,)
    grad = jax.grad(lambda a: env.step(state, a).obs.sum())(jnp.zeros(env.action_size))
    assert jnp.abs(grad).sum() > 0


def test_purejaxql_runs_unmodified_on_gymnax_1():
    spec = COMPARISONS["PQN"]
    rollout = spec.reference_config["NUM_ENVS"] * spec.reference_config["NUM_STEPS"]
    tiny = dataclasses.replace(
        spec,
        eval_every=2 * rollout,
        reference_config=dict(
            spec.reference_config,
            TOTAL_TIMESTEPS=4 * rollout,
            TOTAL_TIMESTEPS_DECAY=4 * rollout,
            TEST_INTERVAL=0.5,
        ),
    )
    curve = np.asarray(refcmp_jax.run_purejaxql(tiny, [0, 1]))
    assert curve.shape == (2, 2) and np.isfinite(curve).all()


@pytest.mark.slow
def test_brax_apg_trains_on_the_gymnax_adapter():
    spec = COMPARISONS["APG"]
    tiny = dataclasses.replace(
        spec,
        reference_config=dict(
            spec.reference_config, policy_updates=2, num_envs=4, horizon_length=8
        ),
    )
    curve = np.asarray(refcmp_jax.run_brax_apg(tiny, [0]))
    assert curve.shape == (1, 1) and np.isfinite(curve).all()


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_report_verdicts(tmp_path):
    solved, unsolved = [500.0] * 6, [100.0] * 6
    _write(
        tmp_path / "r.jsonl",
        [
            {
                "agent": "DQN",
                "arm": "reference",
                "seeds": list(range(16)),
                "curve": [solved] * 16,
            },
            {
                "agent": "DQN",
                "arm": "matched",
                "seeds": list(range(16)),
                "curve": [unsolved] * 16,
            },
            # A re-run replaces the seeds it records: matched now matches.
            {
                "agent": "DQN",
                "arm": "matched",
                "seeds": list(range(16)),
                "curve": [solved] * 16,
            },
            {
                "agent": "DQN",
                "arm": "defaults",
                "seeds": list(range(16)),
                "curve": [unsolved] * 16,
            },
            {
                "agent": "PQN",
                "arm": "reference",
                "seeds": list(range(8)),
                "curve": [solved] * 8,
            },
            # 4/8 against 8/8: adjusted p = 0.15, too weak to call on 8 seeds.
            {
                "agent": "PQN",
                "arm": "matched",
                "seeds": list(range(8)),
                "curve": [solved] * 4 + [unsolved] * 4,
            },
        ],
    )
    rows = {
        tuple(cell.strip() for cell in row.split("|")[1:3]): row
        for row in refcmp_report.report(tmp_path).splitlines()[2:]
    }
    assert "| 16 | 16/16 |" in rows[("DQN", "Ajax matched")]
    assert rows[("DQN", "Ajax matched")].endswith("| match | equal, 16 seeds |")
    # One interim of 4 seeds a side cannot reject: 2 of its 70 permutations
    # are as extreme as the data. Two can.
    assert rows[("DQN", "Ajax defaults")].endswith("| gap | Ajax smaller, 8 seeds |")
    assert rows[("PQN", "Ajax matched")].endswith(
        "| unclear: run 16 seeds | Ajax smaller, 8 seeds |"
    )
