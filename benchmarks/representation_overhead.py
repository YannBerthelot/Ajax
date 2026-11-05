"""What RepresentationProbe costs a run: probing on against off.

Compiles both programs once, then alternates their runs, so that the machine's
drift falls on both and compile time on neither, and reports each pair's CPU
time, which holds up on a busy machine where wall-clock time does not. The
probes come at the probe's defaults, ``--probes`` to a run::

    JAX_PLATFORMS=cpu uv run python benchmarks/representation_overhead.py \
        --env Pendulum-v1 --timesteps 50000 --pairs 3

Rows go to ``--out`` as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import statistics
import tempfile
import time
from typing import Callable

import jax
import jax.numpy as jnp

from ajax import SAC
from ajax.extensions.representation import RepresentationProbe


def compiled_run(args: argparse.Namespace, extensions: tuple) -> Callable:
    """A run of a fresh agent, compiled by this call."""
    agent = SAC(args.env, n_envs=args.n_envs, extensions=extensions)
    train = agent.get_make_train()(
        env_args=agent.env_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        network_args=agent.network_args,
        total_timesteps=args.timesteps,
        num_episode_test=1,
        run_ids=[],
        logging_config=None,
    )
    run = jax.jit(jax.vmap(train))
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(args.seeds))
    index = jnp.arange(args.seeds)
    jax.block_until_ready(run(keys, index))
    return lambda: jax.block_until_ready(run(keys, index))


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timed(run: Callable) -> tuple[float, float]:
    wall, cpu = time.perf_counter(), cpu_seconds()
    run()
    return time.perf_counter() - wall, cpu_seconds() - cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--probes", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    directory = tempfile.mkdtemp(prefix="representation_overhead")
    probe = RepresentationProbe(directory, args.timesteps // args.probes)
    runs = {False: compiled_run(args, ()), True: compiled_run(args, (probe,))}
    rows = []
    for pair in range(args.pairs):
        order = (False, True) if pair % 2 == 0 else (True, False)
        timings = {probed: timed(runs[probed]) for probed in order}
        row = {
            "env": args.env,
            "timesteps": args.timesteps,
            "seeds": args.seeds,
            "probes": args.probes,
            "pair": pair,
            "off_wall_s": round(timings[False][0], 2),
            "on_wall_s": round(timings[True][0], 2),
            "off_cpu_s": round(timings[False][1], 2),
            "on_cpu_s": round(timings[True][1], 2),
            "cpu_ratio": round(timings[True][1] / timings[False][1], 4),
        }
        rows.append(row)
        print(row, flush=True)
    shutil.rmtree(directory, ignore_errors=True)
    print(
        f"median CPU overhead {statistics.median(r['cpu_ratio'] for r in rows) - 1:+.1%}"
    )
    if args.out:
        with open(args.out, "a") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
