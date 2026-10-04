"""Run Ajax DreamerV3 on CartPole-v1 at the protocol config (PROTOCOL.md 3.6, 5.1).

Usage (Ajax venv, CPU; run from outside the Ajax checkout, e.g. from the
output directory, so that nothing is written there; ``launch.sh`` runs it this
way, README.md):

    cd OUT && JAX_PLATFORMS=cpu PYTHONPATH=$AJ/src AJAX_SRC=$AJ/src $APY $HERE/run_ajax.py \\
        --seeds 0 1 2 [--rows 20000 --log-every 2000 --eval-episodes 10] --out OUT/ajax [--tiny]

with ``AJ`` the Ajax checkout to run, ``APY`` the python of the Ajax venv and
``HERE`` this directory. ``AJAX_SRC`` (default: the ``src`` directory of the
imported ``ajax`` package) is the root the imported file is recorded against.

One vmapped run over the given seeds (as benchmarks/learning_checks.py runs a
check): ``ajax.DreamerV3("CartPole-v1", model_size="1m")`` with every other
argument at its default; ``agent.train(seed=seeds, n_timesteps=rows,
num_episode_test=eval_episodes, logging_config=LoggingConfig(config={},
log_frequency=log_every, use_wandb=False, use_tensorboard=False))``.
``--tiny`` (smoke only) sets units = hidden = 16, deter = 64, classes = 4
(the reference smoke's ``--tiny`` widths).

Outputs in ``--out`` (created; must not exist). Recorded locations are
relative to named roots (provenance.py, README.md "Recorded locations"):
``<out>`` is the directory holding ``--out``, ``<ajax-src>`` is ``AJAX_SRC``.
  config.json                   resolved Ajax config (constructor config,
                                DreamerV3Config, agent config, schedule, env;
                                run: the arguments with out as <out>/..., the
                                imported ajax file as <ajax-src>/..., the
                                commit of that tree and whether src had
                                uncommitted changes)
  s{seed}/evals.jsonl           {rows, updates, returns: null, mean, len_mean}
                                (Ajax's evaluation returns only the mean)
  s{seed}/train_metrics.jsonl   {rows, updates, n_updates_window, s2_logged,
                                 metrics{every Train/<name> without prefix}}
  s{seed}/records.jsonl         PROTOCOL.md 5.1 schema
  ajax_records.jsonl            all seeds' records (seed field per line)
  ajax_final.npz                per seed: final episodic_return_state
                                (buffer, count, index, sum), last_episode_length,
                                retnorm lo / hi, n_updates (S2 audit)
  all_metrics.npz               every logged metric at the logging ticks
  timing.json                   wall time of the train call (compile included)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import provenance  # noqa: E402  (this directory)

import ajax  # noqa: E402
from ajax.logging.wandb_logging import LoggingConfig  # noqa: E402

TINY = {"units": 16, "hidden": 16, "deter": 64, "classes": 4}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return repr(x)


def _f(v):
    v = float(v)
    return None if not np.isfinite(v) else v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out).resolve()
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)

    # Recorded relative to named roots (provenance.py).
    ajax_src = os.environ.get("AJAX_SRC") or pathlib.Path(ajax.__file__).parents[1]
    ajax_rev = provenance.git_state(ajax_src)

    kw = dict(TINY) if a.tiny else {}
    agent = ajax.DreamerV3("CartPole-v1", model_size="1m", **kw)
    sched = agent.schedule
    cfg = {
        "constructor": _jsonable(
            {k: v for k, v in agent.config.items() if k != "env_params"}
        ),
        "dreamer_config": _jsonable(dataclasses.asdict(agent.dreamer_config)),
        "agent_config": _jsonable(
            {
                k: getattr(agent.agent_config, k)
                for k in (
                    "train_ratio",
                    "batch_size",
                    "batch_length",
                    "replay_capacity",
                )
            }
        ),
        "schedule": {
            "repr": repr(sched),
            "first_update_step": int(sched.first_update_step),
            "first_update_tick": int(sched.first_update_tick),
            "total_updates": int(sched.total_updates(a.rows // agent.env_args.n_envs)),
        },
        "n_envs": agent.env_args.n_envs,
        "ring_rows": agent._ring_rows(a.rows),
        "run": {
            **vars(a),
            "out": provenance.located(out, {"out": out.parent}),
            "mode": "one vmapped run over seeds",
            "jax": jax.__version__,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "ajax_file": provenance.located(ajax.__file__, {"ajax-src": ajax_src}),
            "ajax_commit": ajax_rev["commit"],
            "ajax_dirty": ajax_rev["dirty"],
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg["schedule"]), flush=True)

    logging_config = LoggingConfig(
        config={}, log_frequency=a.log_every, use_wandb=False, use_tensorboard=False
    )
    t0 = time.perf_counter()
    state, metrics = agent.train(
        seed=list(a.seeds),
        n_timesteps=a.rows,
        num_episode_test=a.eval_episodes,
        logging_config=logging_config,
    )
    jax.block_until_ready(state)
    wall = time.perf_counter() - t0
    (out / "timing.json").write_text(json.dumps({"train_wall_s": wall}, indent=1))
    print(f"train wall {wall:.0f}s", flush=True)

    ev = np.asarray(metrics["Eval/episodic mean reward"])
    ticks = np.flatnonzero(np.isfinite(ev[0]))
    dump = {k: np.asarray(v)[:, ticks] for k, v in metrics.items()}
    np.savez(
        out / "all_metrics.npz",
        seeds=np.asarray(a.seeds),
        ticks=ticks,
        **{k.replace("/", "|"): v for k, v in dump.items()},
    )

    # final state reads for the S2 audit (PROTOCOL.md 5.1)
    cs = state.collector_state
    ers = cs.episodic_return_state
    np.savez(
        out / "ajax_final.npz",
        seeds=np.asarray(a.seeds),
        buffer=np.asarray(ers.buffer),
        count=np.asarray(ers.count),
        index=np.asarray(ers.index),
        sum=np.asarray(ers.sum),
        episodic_mean_return=np.asarray(cs.episodic_mean_return),
        last_episode_length=np.asarray(cs.last_episode_length),
        retnorm_lo=np.asarray(state.retnorm.lo),
        retnorm_hi=np.asarray(state.retnorm.hi),
        n_updates=np.asarray(state.n_updates),
        rows=np.asarray(cs.rows),
    )

    train_keys = sorted(k for k in dump if k.startswith("Train/"))
    all_lines = []
    for si, seed in enumerate(a.seeds):
        d = out / f"s{seed}"
        d.mkdir()
        prev = 0
        with open(d / "evals.jsonl", "w") as fe, open(
            d / "train_metrics.jsonl", "w"
        ) as ft, open(d / "records.jsonl", "w") as fr:
            for j in range(len(ticks)):
                rows = int(dump["timestep"][si, j])
                upd = int(dump["Train/n_updates"][si, j])
                tm = {k[len("Train/") :]: _f(dump[k][si, j]) for k in train_keys}
                s2 = tm.pop("episodic mean reward")
                tm.pop("n_updates")
                emean = _f(dump["Eval/episodic mean reward"][si, j])
                elen = _f(dump["Eval/mean episodic length"][si, j])
                fe.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "rows": rows,
                            "updates": upd,
                            "returns": None,
                            "mean": emean,
                            "len_mean": elen,
                        }
                    )
                    + "\n"
                )
                ft.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "rows": rows,
                            "updates": upd,
                            "n_updates_window": upd - prev,
                            "s2_logged": s2,
                            "metrics": tm,
                        }
                    )
                    + "\n"
                )
                rec = {
                    "side": "ajax",
                    "seed": seed,
                    "rows": rows,
                    "updates": upd,
                    "eval_mean": emean,
                    "eval_len_mean": elen,
                    "eval_returns": None,
                    "s2_logged": s2,
                    "train": tm,
                    "train_raw": tm,
                    "n_calls": upd - prev,
                    "n_metric_samples": upd - prev,
                    "retnorm": None,
                    "wall_s": wall,
                }
                fr.write(json.dumps(rec) + "\n")
                all_lines.append(rec)
                prev = upd
        print(
            f"seed {seed}: eval",
            [
                round(r["eval_mean"] or float("nan"), 1)
                for r in all_lines
                if r["seed"] == seed
            ],
            flush=True,
        )
    with open(out / "ajax_records.jsonl", "w") as f:
        for r in all_lines:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
