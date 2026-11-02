"""The expert-guidance experiment, as PREREG.md beside this file registers
it: does each of four published ways of using a controller make SAC learn
Pendulum faster than SAC alone?

usage, from the repository root:
    uv run python benchmarks/expert_guidance/guidance.py expert
    uv run python benchmarks/expert_guidance/guidance.py run
    uv run python benchmarks/expert_guidance/guidance.py report

``expert`` measures the controller's own return, with no training. ``run``
replays AdaStop over the batches already in ``results.jsonl`` and trains
only the seeds it lacks: per interim, the ``n`` next seeds of every arm
still in an undecided comparison, all of an arm's seeds in one program,
each batch appended as one record. ``report`` prints the registered results.
"""

import datetime
import json
import pathlib
import platform
import subprocess
import sys
import time

import gymnax
import jax
import numpy as np
from swing_up import SwingUpPD

from ajax import SAC
from ajax.compare import AdaStop, auc, difference, resolution
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy
from ajax.extensions.pretrain import ExpertPrefill
from ajax.extensions.target_mods import IBRL
from ajax.logging import LoggingConfig

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
RECORDS = HERE / "results.jsonl"
ENV_ID = "Pendulum-v1"
N_TIMESTEPS, EVAL_EVERY, N_EVAL_EPISODES = 20_000, 1_000, 10
N, K, ALPHA = 4, 12, 0.05
SOLVED = -200.0

# Every arm runs the matched settings of the SAC reference comparison at its
# version 2 (benchmarks/reference/refcmp_specs.py: RL Zoo's Pendulum SAC),
# written out so that the registered run cannot move with that spec; a test
# holds them equal.
SETTINGS = {
    "n_envs": 1,
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "alpha_learning_rate": 1e-3,
    "buffer_size": 20_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "alpha_init": 1.0,
    "target_entropy_per_dim": -1.0,
    "policy_update_start": 0,
    "alpha_update_start": 0,
    "actor_architecture": ("256", "relu", "256", "relu"),
    "critic_architecture": ("256", "relu", "256", "relu"),
}
EXPERT = SwingUpPD()
# A0 logs the expert's return on its evaluations, and takes no other part.
ARMS = {
    "A0": {"eval_expert_policy": EXPERT},
    "A1": {"expert_policy": EXPERT, "extensions": (ExpertPrefill(n_steps=2_000),)},
    "A2": {
        "expert_policy": EXPERT,
        "extensions": (JSRLCurriculum(episode_length=200, decay_frac=0.5),),
    },
    "A3": {"expert_policy": EXPERT, "extensions": (IBRL(expert_policy=EXPERT),)},
    "A4": {"expert_policy": EXPERT, "extensions": (ResidualPolicy(scale=1.0),)},
}
LABELS = {
    "A0": "SAC",
    "A1": "ExpertPrefill (DDPGfD's demonstrations in the replay)",
    "A2": "JSRLCurriculum (JSRL)",
    "A3": "IBRL",
    "A4": "ResidualPolicy (residual RL)",
}
COMPARED = [(arm, "A0") for arm in ARMS if arm != "A0"]


def build(arm: str) -> SAC:
    return SAC(ENV_ID, **SETTINGS, **ARMS[arm])


def train(
    arm: str, seeds: list, n_timesteps: int = N_TIMESTEPS, eval_every: int = EVAL_EVERY
) -> dict:
    """One batch: every seed's evaluation curve, the expert's on the same
    episodes, and what the batch cost."""
    logging = LoggingConfig(log_frequency=eval_every, backends=())
    wall, cpu = time.perf_counter(), time.process_time()
    _, evaluations = build(arm).train(
        seed=seeds,
        n_timesteps=n_timesteps,
        num_episode_test=N_EVAL_EPISODES,
        logging_config=logging,
    )
    return {
        "curve": np.asarray(evaluations["Eval/episodic mean reward"]).tolist(),
        "expert_curve": np.asarray(
            evaluations["Eval/episodic mean expert reward"]
        ).tolist(),
        "wall_s": round(time.perf_counter() - wall, 1),
        "cpu_s": round(time.process_time() - cpu, 1),
    }


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def manifest() -> dict:
    return {
        "ajax_commit": _git("rev-parse", "HEAD"),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "jax": jax.__version__,
        "platform": platform.platform(),
    }


def records() -> list:
    if not RECORDS.exists():
        return []
    return [json.loads(line) for line in RECORDS.read_text().splitlines() if line]


def decide(train=train) -> tuple:
    """AdaStop on each seed's mean evaluation return (``auc``), each arm
    against A0: the test and each arm's batches, recorded ones reused. With
    ``train`` None, only recorded batches are read."""
    test = AdaStop(list(ARMS), n=N, K=K, alpha=ALPHA, comparisons=COMPARED)
    recorded = {(r["arm"], tuple(r["seeds"])): r for r in records()}
    batches: dict = {arm: [] for arm in ARMS}
    while not test.finished:
        seeds = list(range(test.k * N, (test.k + 1) * N))
        new = {}
        for arm in ARMS:
            if arm not in test.active:
                continue
            record = recorded.get((arm, tuple(seeds)))
            if record is None and train is None:
                raise LookupError(f"{arm}, seeds {seeds}: not recorded yet")
            if record is None:
                record = {
                    "arm": arm,
                    "seeds": seeds,
                    **train(arm, seeds),
                    "manifest": manifest(),
                }
                with RECORDS.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            batches[arm].append(record)
            new[arm] = auc(np.asarray(record["curve"]))
        test.update(new)
    return test, batches


def first_solved(curve: list) -> float:
    """The steps a seed took to its first evaluation at or above SOLVED."""
    hits = [i for i, value in enumerate(curve) if value >= SOLVED]
    return (hits[0] + 1) * EVAL_EVERY if hits else float("nan")


def report() -> None:
    test, batches = decide(train=None)  # recorded batches only
    curves = {arm: np.concatenate([b["curve"] for b in batches[arm]]) for arm in ARMS}
    expert = np.concatenate(
        [np.asarray(b["expert_curve"]) for arm in ARMS for b in batches[arm]]
    )
    print(f"Expert's return on the evaluations: {np.nanmean(expert):.1f}\n")
    print(
        "| Arm | Seeds | Mean evaluation return | Final evaluation | Steps to -200 (median; seeds never) |"
    )
    print("|---|---|---|---|---|")
    for arm, curve in curves.items():
        steps = np.array([first_solved(c) for c in curve.tolist()])
        median = np.nanmedian(steps) if np.isfinite(steps).any() else float("nan")
        print(
            f"| {arm} {LABELS[arm]} | {len(curve)} | {auc(curve).mean():.1f} | {curve[:, -1].mean():.1f}"
            f" | {median:,.0f}; {int(np.isnan(steps).sum())} |"
        )
    print(
        "\n| Comparison | AdaStop | Difference [95%] | Resolution |\n|---|---|---|---|"
    )
    for arm, base in COMPARED:
        a, b = auc(curves[arm]), auc(curves[base])
        gap, (low, high) = difference(a, b)
        print(
            f"| {arm} vs {base} | {test.decisions[(arm, base)]} | {gap:+.1f} [{low:+.1f}, {high:+.1f}]"
            f" | {resolution(a, b, N, K, ALPHA):.1f} |"
        )


def expert_returns(n_episodes: int = 1_000, seed: int = 0) -> np.ndarray:
    """The expert's return on n_episodes 200-step episodes of gymnax's
    Pendulum-v1 from its own resets, with no training."""
    env, params = gymnax.make(ENV_ID)

    def episode(key):
        reset_key, key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)

        def step(carry, step_key):
            obs, state = carry
            obs, state, reward, *_ = env.step(step_key, state, EXPERT(obs), params)
            return (obs, state), reward

        _, rewards = jax.lax.scan(step, (obs, state), jax.random.split(key, 200))
        return rewards.sum()

    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    return np.asarray(jax.jit(jax.vmap(episode))(keys))


def main(argv: list) -> None:
    if argv == ["expert"]:
        returns = expert_returns()
        print(
            f"mean {returns.mean():.1f}, sd {returns.std(ddof=1):.1f}, median {np.median(returns):.1f}"
        )
    elif argv == ["run"]:
        if _git("status", "--porcelain", "--", ".", f":!{RECORDS.relative_to(ROOT)}"):
            sys.exit("commit first: every record names the commit it ran")
        decide()
        report()
    elif argv == ["report"]:
        report()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
