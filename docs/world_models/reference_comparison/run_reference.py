# Contains code derived from danijar/dreamerv3 at commit 29eb964
# (29eb964e2918a3f4db04086f7f51b60388e97f3d), file dreamerv3/main.py:
# build_config reproduces main.py:35-47 (the replay_length update and the run
# arguments), main() reproduces main.py:48-59 (logdir, config.save, timer
# initialiser), make_agent is the body of main.make_agent (main.py:136-144)
# and make_logger a reduced main.make_logger (main.py:147-159).
# Modified: the timestamped logdir and the command-line flags are replaced by
# this script's arguments and the protocol's overrides; make_agent builds
# its spaces from the CartPole port (cartpole_env.py); make_logger keeps only
# the JSONL output (the stock one needs tensorflow); everything else in this
# file (Instr: evaluation, records, flush step) is new instrumentation.
#
# Copyright (c) 2023 Danijar Hafner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The license text is also in LICENSE.dreamerv3 in this directory.
"""Run the REAL reference DreamerV3 (danijar/dreamerv3 @ 29eb964) on the CartPole port.

PROTOCOL.md sections 1, 3, 4, 5.

Usage (reference venv, CPU; ``launch.sh`` runs it this way, README.md):

    cd $REF && JAX_PLATFORMS=cpu DREAMERV3_REF=$REF PYTHONPATH=$REF:$HERE \\
        $RPY $HERE/run_reference.py --seed S \\
        [--rows 20000 --eval-every 2000 --eval-episodes 10] --out OUT/ref_sS [--tiny]

with ``REF`` a checkout of danijar/dreamerv3 at 29eb964, ``RPY`` the python of
the reference venv and ``HERE`` this directory. ``DREAMERV3_REF`` (default
``HERE/.work/dreamerv3_29eb964``) is put first on ``sys.path``.

What it does: the stock ``dreamerv3/main.py`` logic for ``script == 'train'``
(config = defaults updated with the protocol's OVERRIDES, then the stock
``replay_length`` update and ``args``; ``config.save(logdir/'config.yaml')``,
with the recorded ``logdir`` relative as below; timer initialiser), then
``train_instrumented.train`` = the stock ``embodied/run/train.py`` plus
marked instrumentation, with
* ``make_agent``: the stock ``main.make_agent`` body (port spaces);
* ``make_replay``: the stock ``dreamerv3.main.make_replay`` (imported);
* ``make_env``: ``CartPolePort`` (cartpole_env.py) wrapped by the stock
  ``dreamerv3.main.wrap_env``;
* ``make_logger``: ``embodied.Logger`` with a single ``JSONLOutput``
  (``metrics.jsonl``); the stock one needs tensorflow (TensorBoardOutput).

Instrumentation (``Instr``; reads only, PROTOCOL.md 3.2 / 4.2 / 5.1): at
multiples of ``--eval-every`` steps (one step = one env row; reset rows
count), the training-metric window means (``agg.result()``, the stock
aggregator that nothing else reads with log_every = -1), the number of
updates, the return normaliser state, and an evaluation of
``--eval-episodes`` fresh port envs (first episode of each, latest trained
policy parameters, zero carry, sampled actions, at most 500 steps, same reset
draws and policy seeds at every evaluation of a seed). After the loop, one
flush env step records the episodes ending on the row the next vector step
would emit (row ``rows / 16``).

Outputs in ``--out`` (must not exist; the stock checkpoint logic would
otherwise resume from it). Recorded locations are relative to named roots
(provenance.py, README.md "Recorded locations"): ``<out>`` is the directory
holding ``--out`` (``launch.sh``'s output directory), ``<reference-checkout>``
is ``DREAMERV3_REF``.
  config.yaml          full resolved reference config (stock config.save; its
                       logdir recorded as <out>/<name of --out>, the run used
                       the real directory)
  run_args.json        the arguments (out as <out>/...), the overrides (logdir
                       likewise), jax version, XLA_FLAGS, the imported
                       reference files (<reference-checkout>/...) and that
                       checkout's commit and dirty flag
  episodes.jsonl       one line per finished training episode:
                       {seed, worker, row_last, step, length, score, terminal}
                       (step = 16 (row_last + 1), the global step count once
                       that row was added; the flush-step episodes have
                       row_last = rows / 16)
  evals.jsonl          {rows, updates, returns[], lengths[], mean, len_mean, wall_s}
  train_metrics.jsonl  {rows, updates, n_calls, n_metric_samples,
                        metrics{native name: window mean}, metrics_ajax{mapped}}
  records.jsonl        PROTOCOL.md 5.1 schema (the three above merged)
  progress.jsonl       {step, updates, t} every --progress-every steps (timing)
  timing.json          build/compile time, first-update time, totals
  metrics.jsonl        stock logger (episode/score, episode/length cross-check)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

T_START = time.time()

RR = pathlib.Path(__file__).resolve().parent
# The reference checkout (danijar/dreamerv3 at 29eb964); README.md "Setup".
REF = pathlib.Path(
    os.environ.get("DREAMERV3_REF", str(RR / ".work" / "dreamerv3_29eb964"))
)
for p in (str(RR), str(REF)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ruff: noqa: I001
# Import order kept as in the runs: the reference modules first (dreamerv3/main.py
# inserts the checkout and its parent at the front of sys.path when imported),
# then this directory's modules.
import embodied  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from dreamerv3 import agent as agt  # noqa: E402
from dreamerv3 import main as ref_main  # noqa: E402  (stock make_replay, wrap_env)

import cartpole_env  # noqa: E402
import namemap  # noqa: E402
import provenance  # noqa: E402
import train_instrumented  # noqa: E402

EVAL_ENV_TAG = 7_000_001
EVAL_SEED_TAG = 1_000_000
MAX_EVAL_STEPS = 500

# PROTOCOL.md section 1 "Complete override set" (seed / task / logdir added per run).
BASE_OVERRIDES = {
    "jax.platform": "cpu",
    "jax.compute_dtype": "float32",
    "jax.param_dtype": "float32",
    "jax.prealloc": False,
    "jax.transfer_guard": False,
    "dyn.rssm.deter": 512,
    "dyn.rssm.hidden": 64,
    "dyn.rssm.classes": 4,
    r".*\.units": 64,
    "run.train_ratio": 512.0,
    "run.steps": 20000,
    "run.num_envs": 16,
    "run.log_every": -1,
    "run.eval_every": -1,
    "run.save_every": -1,
    "run.driver_parallel": False,
}
# --tiny (smoke only): smaller widths, same structure (blocks 8 divides deter).
TINY_OVERRIDES = {
    "dyn.rssm.deter": 64,
    "dyn.rssm.hidden": 16,
    "dyn.rssm.classes": 4,
    r".*\.units": 16,
}


def build_config(seed: int, rows: int, logdir: str, tiny: bool):
    over = dict(BASE_OVERRIDES)
    if tiny:
        over.update(TINY_OVERRIDES)
    over.update(
        {"seed": seed, "task": "gymnax_cartpole", "logdir": logdir, "run.steps": rows}
    )
    config = embodied.Config(agt.Agent.configs["defaults"]).update(over)
    # stock main.py:35-38
    config = config.update(
        replay_length=config.replay_length or config.batch_length,
        replay_length_eval=config.replay_length_eval or config.batch_length_eval,
    )
    # stock main.py:39-47
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_length=config.replay_length,
        replay_length_eval=config.replay_length_eval,
        replay_context=config.replay_context,
    )
    return config, args, over


EPISODES: list = []  # finished training episodes, appended by the training envs


def make_env(config, index, recorder=EPISODES):
    env = cartpole_env.CartPolePort([int(config.seed), int(index)], index, recorder)
    return ref_main.wrap_env(env, config)


def make_agent(config):
    # stock main.py:136-144 body, with this file's make_env (never stepped)
    env = make_env(config, 0, recorder=None)
    if config.random_agent:
        agent = embodied.RandomAgent(env.obs_space, env.act_space)
    else:
        agent = agt.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent


def make_logger(config):
    step = embodied.Counter()
    return embodied.Logger(
        step, [embodied.logger.JSONLOutput(config.logdir, "metrics.jsonl")], 1
    )


def _scalar(v):
    a = np.asarray(v)
    return float(a) if a.ndim == 0 else None


class Instr:
    """The instrumentation hook of train_instrumented.py (reads only)."""

    def __init__(
        self,
        config,
        out: pathlib.Path,
        record_every: int,
        eval_episodes: int,
        progress_every: int,
        rows: int,
    ):
        self.config = config
        self.seed = int(config.seed)
        self.out = out
        self.record_every = int(record_every)
        self.eval_episodes = int(eval_episodes)
        self.progress_every = int(progress_every)
        self.rows = int(rows)
        self.prev_updates = 0
        self.records: list = []
        self.timing = {"t_start": T_START}
        self._first_update_seen = False

    # ------------------------------------------------------------ progress
    def progress(self, step: int, agent) -> None:
        updates = int(agent.updates)
        now = time.time()
        if "t_loop_start" not in self.timing:
            self.timing["t_loop_start"] = now
        if updates > 0 and not self._first_update_seen:
            self._first_update_seen = True
            self.timing["first_update_step"] = step
            self.timing["t_first_update"] = now
        if step % self.progress_every == 0 or (updates > 0 and updates < 4):
            with open(self.out / "progress.jsonl", "a") as f:
                f.write(
                    json.dumps({"step": step, "updates": updates, "t": now - T_START})
                    + "\n"
                )

    # ------------------------------------------------------------ record
    def record(self, rows: int, agent, agg, logger) -> None:
        t0 = time.time()
        res = agg.result()  # stock Agg, reset on read
        raw = {}
        for k, v in res.items():
            if not k.startswith("train/") or k.endswith("/dist"):
                continue
            s = _scalar(v)
            if s is not None:
                raw[k[len("train/") :]] = s
        updates = int(agent.updates)
        n_calls = updates - self.prev_updates
        # JAXAgent.train returns the metrics of the previous call (stock D1):
        # the window's Agg holds the metrics of calls prev-1 .. updates-2.
        n_samples = max(0, updates - 1) - max(0, self.prev_updates - 1)
        self.prev_updates = updates
        retnorm = self._retnorm(agent)
        ev = self.evaluate(agent)
        rec = {
            "side": "ref",
            "seed": self.seed,
            "rows": rows,
            "updates": updates,
            "eval_mean": ev["mean"],
            "eval_len_mean": ev["len_mean"],
            "eval_returns": ev["returns"],
            "eval_lengths": ev["lengths"],
            "s2_logged": None,
            "train": {namemap.to_ajax(k): v for k, v in raw.items()},
            "train_raw": raw,
            "n_calls": n_calls,
            "n_metric_samples": n_samples,
            "retnorm": retnorm,
            "wall_s": time.time() - T_START,
            "eval_s": ev["eval_s"],
            "record_s": None,
        }
        rec["record_s"] = time.time() - t0
        self.records.append(rec)
        with open(self.out / "records.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(self.out / "evals.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "seed": self.seed,
                        "rows": rows,
                        "updates": updates,
                        "returns": ev["returns"],
                        "lengths": ev["lengths"],
                        "mean": ev["mean"],
                        "len_mean": ev["len_mean"],
                        "wall_s": rec["wall_s"],
                    }
                )
                + "\n"
            )
        with open(self.out / "train_metrics.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "seed": self.seed,
                        "rows": rows,
                        "updates": updates,
                        "n_calls": n_calls,
                        "n_metric_samples": n_samples,
                        "metrics": raw,
                        "metrics_ajax": rec["train"],
                        "retnorm": retnorm,
                    }
                )
                + "\n"
            )
        self._write_episodes()
        logger.write()
        print(
            f"[record] rows {rows} updates {updates} eval_mean {ev['mean']:.1f}"
            f" (len {ev['len_mean']:.1f}) eval {ev['eval_s']:.1f}s wall {rec['wall_s']:.0f}s"
            f" episodes {len(EPISODES)}",
            flush=True,
        )

    def _retnorm(self, agent):
        names = [k for k in agent.params if "/retnorm/" in k]
        lo_keys = [k for k in names if k.endswith("low") or k.endswith("low/value")]
        hi_keys = [k for k in names if k.endswith("high") or k.endswith("high/value")]
        if len(lo_keys) != 1 or len(hi_keys) != 1:
            return {"error": f"retnorm keys {sorted(names)}"}
        lo = float(np.asarray(agent.params[lo_keys[0]]))
        hi = float(np.asarray(agent.params[hi_keys[0]]))
        limit = float(self.config.retnorm.limit)
        return {
            "lo": lo,
            "hi": hi,
            "scale": max(limit, hi - lo),
            "keys": [lo_keys[0], hi_keys[0]],
        }

    # ------------------------------------------------------------ evaluation
    def evaluate(self, agent) -> dict:
        """PROTOCOL.md 4.2: fresh envs, first episode each, latest params."""
        t0 = time.time()
        n = self.eval_episodes
        s = self.seed
        params = {k: agent.params[k].copy() for k in agent.policy_keys}
        params = jax.device_put(params, agent.policy_mirrored)
        envs = [
            ref_main.wrap_env(
                cartpole_env.CartPolePort([s, EVAL_ENV_TAG, i], i, None), self.config
            )
            for i in range(n)
        ]
        seed_rng = np.random.default_rng([s, EVAL_ENV_TAG, EVAL_SEED_TAG])

        def next_seed():
            seed = seed_rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
            return jax.device_put(seed, agent.policy_sharded)

        carry = agent._init_policy(params, next_seed(), n)
        obs = [env.step({"action": np.int32(0), "reset": True}) for env in envs]
        ret = np.zeros(n, np.float64)
        length = np.zeros(n, np.int64)
        done = np.zeros(n, bool)
        for _ in range(MAX_EVAL_STEPS):
            batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            batch = agent._filter_data(batch)
            acts, _, carry = agent._policy(params, batch, carry, next_seed(), "eval")
            action = np.asarray(acts["action"])
            for i, env in enumerate(envs):
                if done[i]:
                    continue  # a done env keeps its last obs; its outputs are ignored
                o = env.step({"action": np.int32(action[i]), "reset": False})
                obs[i] = o
                ret[i] += float(o["reward"])
                length[i] += 1
                done[i] = bool(o["is_last"])
            if done.all():
                break
        return {
            "returns": ret.tolist(),
            "lengths": length.tolist(),
            "mean": float(ret.mean()),
            "len_mean": float(length.mean()),
            "eval_s": time.time() - t0,
        }

    # ------------------------------------------------------------ end
    def flush(self, driver) -> None:
        """One env step per training env with the driver's pending actions."""
        n_before = len(EPISODES)
        for i, env in enumerate(driver.envs):
            env.step({k: v[i] for k, v in driver.acts.items()})
        self.timing["flush_episodes"] = len(EPISODES) - n_before

    def _write_episodes(self) -> None:
        tmp = self.out / "episodes.jsonl.tmp"
        with open(tmp, "w") as f:
            for e in EPISODES:
                f.write(
                    json.dumps(
                        {"seed": self.seed, "step": 16 * (e["row_last"] + 1), **e}
                    )
                    + "\n"
                )
        os.replace(tmp, self.out / "episodes.jsonl")

    def finish(self) -> None:
        self._write_episodes()
        t = self.timing
        t["t_end"] = time.time()
        t["build_and_compile_s"] = t.get("t_loop_start", t["t_end"]) - T_START
        t["total_s"] = t["t_end"] - T_START
        with open(self.out / "timing.json", "w") as f:
            json.dump(t, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=160)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    assert (
        a.rows % 16 == 0 and a.eval_every % 16 == 0
    ), "rows and eval-every are multiples of 16"

    out = pathlib.Path(a.out).resolve()
    assert (
        not out.exists()
    ), f"{out} exists: the stock checkpoint logic would resume from it"
    config, args, over = build_config(a.seed, a.rows, str(out), a.tiny)
    # Recorded relative to named roots (provenance.py); the run itself uses
    # the real logdir.
    out_rec = provenance.located(out, {"out": out.parent})
    ref_rev = provenance.git_state(REF)

    print("Run script:", args.script)
    print("Logdir:", args.logdir)
    logdir = embodied.Path(args.logdir)
    logdir.mkdir()
    config.update(logdir=out_rec).save(logdir / "config.yaml")  # stock main.py:54
    roots = {"reference-checkout": REF}
    with open(out / "run_args.json", "w") as f:
        json.dump(
            {
                **vars(a),
                "out": out_rec,
                "overrides": {**over, "logdir": out_rec},
                "jax": jax.__version__,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "agent_file": provenance.located(agt.__file__, roots),
                "embodied_file": provenance.located(embodied.__file__, roots),
                "reference_commit": ref_rev["commit"],
                "reference_dirty": ref_rev["dirty"],
            },
            f,
            indent=1,
        )

    def init():  # stock main.py:56-59
        embodied.timer.global_timer.enabled = args.timer

    embodied.distr.Process.initializers.append(init)
    init()

    assert args.script == "train"
    from functools import partial as bind

    train_instrumented.INSTR = Instr(
        config, out, a.eval_every, a.eval_episodes, a.progress_every, a.rows
    )
    train_instrumented.train(
        bind(make_agent, config),
        bind(ref_main.make_replay, config, "replay"),
        bind(make_env, config),
        bind(make_logger, config),
        args,
    )
    print(f"done: {time.time() - T_START:.0f}s, episodes {len(EPISODES)}", flush=True)


if __name__ == "__main__":
    main()
