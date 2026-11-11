"""Every Ajax agent on one benchmark: the table, and the commands the cloud
sessions run.

usage, from the repository root:
    uv run python benchmarks/suite/agent_suite.py time LABEL [--n-seeds 1 10]
    uv run python benchmarks/suite/agent_suite.py run LABEL --seeds A-B [--budget N]

``time`` measures, for 1 and then 10 seeds trained at once (one vmapped
program): the seconds JAX spends tracing and compiling a fresh run, the run
steps per second once the agent's updates run (a few thousand steps resumed
past its warm-up, compiling aside, evaluations off) and the peak memory. It
appends one record per seed count to ``results/timing/<LABEL>.jsonl``.

``run`` trains seeds A to B at once, in chunks of whole evaluation intervals
of at most ``--chunk-minutes``: one interval long until a chunk has begun
past the agent's warm-up, then sized at the pace of the last chunk begun past
it. After each chunk it appends a record to
``results/<LABEL>/seeds-A-B.jsonl`` (the evaluations, the steps, the speed
and a manifest), then commits and pushes it. Each evaluation also records,
per seed, ``non_finite_steps``, the training steps the simulator diverged on
so far in the run, and ``eval_lengths``, the mean length of its episodes,
which a task whose episodes run to their limit shows a divergence by. It
stops before a chunk that
would end past ``--stop-after-minutes``: exit 0 when the run reached its
budget, 3 when the same command must run again to continue. A chunk resumes
from the previous chunk's checkpoint (``results/checkpoints``, git-ignored);
when the machine that kept it is gone, the run starts a new attempt from
step 0, the earlier attempt's records kept.
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Optional

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from agent_bench import compile_spans, covered_seconds

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
EVALUATION_KEY = "Eval/episodic mean reward"
NON_FINITE_KEY = "Train/non-finite env steps"
LENGTH_KEY = "Eval/mean episodic length"
#: The warm-up of an agent without learning_starts or a seed phase: past
#: DreamerV3's replay gate (1040 steps over its 16 envs); the on-policy
#: agents update from their first rollout.
WARM_UP_FLOOR = 2048


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """An agent's benchmark: its env, the run's steps, the steps between
    evaluations, why its env count, and the options it takes beyond its
    defaults."""

    env_id: str
    budget: int
    eval_every: int
    reason: str
    options: dict = dataclasses.field(default_factory=dict)


def cheetah(reason: str, budget: int = 2**20, eval_every: int = 2**15, **options):
    """DeepMind Control's cheetah-run on MJX (MuJoCo Playground): MuJoCo
    physics inside Ajax's compiled program, 1000-step episodes, rewards in [0,
    1]. 2^20 steps, about the million of the off-policy papers' MuJoCo runs, a
    power of two so that every agent's iterations divide it; 32 evaluations."""
    return Benchmark("CheetahRun", budget, eval_every, reason, options)


def minatar(
    reason: str, budget: int = 5 * 2**20, eval_every: int = 5 * 2**15, **options
):
    """gymnax's Breakout-MinAtar, the Atari-like suite that runs in JAX. 5 x
    2^20 steps, about MinAtar's 5 million frames (Young and Tian, 2019); 32
    evaluations."""
    return Benchmark("Breakout-MinAtar", budget, eval_every, reason, options)


#: Label -> benchmark. The agent is the label's part before any "-". Every
#: hyperparameter is the agent's default but n_envs, set to its paper's.
BENCHMARKS = {
    "SAC": cheetah("one env, as in its paper's MuJoCo runs"),
    "TD3": cheetah("one env, as in its paper's MuJoCo runs"),
    "REDQ": cheetah("one env, as in its paper's MuJoCo runs"),
    "TQC": cheetah("one env, as in its paper's MuJoCo runs"),
    "CrossQ": cheetah("one env, as in its paper's MuJoCo runs"),
    "ASAC": cheetah("one env, as SAC"),
    "RVISAC": cheetah("one env, as in its paper's MuJoCo runs"),
    "AVG": cheetah("one env: it learns from each step as it comes, no replay"),
    "APO": cheetah(
        "10 envs, its reference's MuJoCo samplers (batch_B): 50 rollouts of"
        " 10 x 2048 steps (2^20 is no whole number of them), evaluated every"
        " second one",
        budget=50 * 20480,
        eval_every=2 * 20480,
        n_envs=10,
    ),
    "PPO": cheetah("one env, as the PPO paper's MuJoCo runs", n_envs=1),
    "DreamerV3": cheetah("16 envs, its reference's run.num_envs"),
    "TDMPC2": cheetah("one env, as in its paper"),
    "DQN": minatar("one env, as MinAtar's DQN baseline"),
    "PQN": minatar("128 envs, its reference's MinAtar config", n_envs=128),
    "PPO-MinAtar": minatar("8 envs, the PPO paper's Atari actors", n_envs=8),
    "UDRL": minatar(
        "one env; discrete actions and episodic returns, as most of its paper's tasks"
    ),
}


def on_more_tasks(labels: tuple[str, ...], env_ids: tuple[str, ...]) -> dict:
    """Each label's benchmark on further envs, as LABEL-TASK (the env id
    without its "mujoco:" prefix or "-MinAtar" suffix): the same budget,
    evaluations and options."""
    benchmarks = {}
    for label in labels:
        for env_id in env_ids:
            task = env_id.removeprefix("mujoco:").removesuffix("-MinAtar")
            benchmarks[f"{label}-{task}"] = dataclasses.replace(
                BENCHMARKS[label], env_id=env_id
            )
    return benchmarks


#: The other DeepMind Control tasks on MJX. Timed with PPO on a CPU
#: (2026-11-03), only HumanoidWalk's physics costs more than CheetahRun's
#: (1.3 times); the others cost 0.2 to 0.9 times as much.
CONTROL_TASKS = (
    "WalkerWalk",
    "WalkerRun",
    "WalkerStand",
    "HopperHop",
    "HopperStand",
    "FingerSpin",
    "FingerTurnHard",
    "ReacherHard",
    "CartpoleSwingup",
    "AcrobotSwingup",
    "PendulumSwingup",
    "BallInCup",
    "FishSwim",
    "HumanoidWalk",
)
BENCHMARKS.update(
    on_more_tasks(("PPO", "APO", "AVG", "SAC", "TD3", "ASAC", "RVISAC"), CONTROL_TASKS)
)
#: Gymnasium's v5 MuJoCo tasks on MJX: every continuous agent, the slow ones
#: included (owner, 2026-11-11). Their episodes always run to the time limit.
GYMNASIUM_MUJOCO_FIXED_LENGTH = (
    "mujoco:HalfCheetah-v5",
    "mujoco:Swimmer-v5",
    "mujoco:Reacher-v5",
    "mujoco:Pusher-v5",
)
#: The ones whose episodes end when the body falls. Paper-era TD-MPC2 refuses
#: them (it supports no terminations), so it runs on the tasks above only.
GYMNASIUM_MUJOCO_TERMINATING = (
    "mujoco:Hopper-v5",
    "mujoco:Walker2d-v5",
    "mujoco:InvertedPendulum-v5",
    "mujoco:InvertedDoublePendulum-v5",
)
BENCHMARKS.update(
    on_more_tasks(
        (
            "PPO",
            "APO",
            "AVG",
            "SAC",
            "TD3",
            "ASAC",
            "RVISAC",
            "REDQ",
            "TQC",
            "CrossQ",
            "DreamerV3",
        ),
        GYMNASIUM_MUJOCO_FIXED_LENGTH + GYMNASIUM_MUJOCO_TERMINATING,
    )
)
BENCHMARKS.update(on_more_tasks(("TDMPC2",), GYMNASIUM_MUJOCO_FIXED_LENGTH))
#: The other MinAtar games gymnax runs.
BENCHMARKS.update(
    on_more_tasks(
        ("PQN", "PPO-MinAtar", "DQN", "UDRL"),
        ("Asterix-MinAtar", "Freeway-MinAtar", "SpaceInvaders-MinAtar"),
    )
)
#: Agents left out, each for a reason inherent to it.
EXCLUDED = {
    "APG": "learns through the environment's gradients: it needs a"
    " differentiable model-reference task, not a reward benchmark",
    "IQL": "offline: it learns from a dataset, not from interaction",
    "DAgger": "imitation: it needs an expert to query",
    "TDMPC2MultiTask": "multi-task: it trains on a suite of tasks",
}


def build(label: str) -> Any:
    import ajax

    benchmark = BENCHMARKS[label]
    agent = getattr(ajax, label.split("-")[0])
    return agent(env_id=benchmark.env_id, **benchmark.options)


def iteration_steps(agent: Any) -> int:
    """The env steps of one training iteration: every env's rollout for the
    agents that collect n_steps per env, one step of every env otherwise."""
    return agent.env_args.n_envs * getattr(agent, "n_steps", 1)


def warm_up_steps(agent: Any) -> int:
    """The steps before the agent's updates run: its learning_starts, or
    TD-MPC2's seed phase, else WARM_UP_FLOOR."""
    for name in ("learning_starts", "resolved_seed_steps"):
        if hasattr(agent, name):
            return getattr(agent, name)
    return WARM_UP_FLOOR


def round_up(steps: int, unit: int) -> int:
    return -(-steps // unit) * unit


def peak_memory_gb() -> float:
    """The process's peak resident memory (ru_maxrss counts kilobytes on
    Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak * (1 if sys.platform == "darwin" else 1024) / 2**30, 2)


def device_peak_memory_gb() -> Optional[float]:
    """The first device's peak memory in JAX's allocator; None on a CPU, whose
    memory is the process's own."""
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return None if peak is None else round(peak / 2**30, 2)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def machine() -> dict:
    cpu = platform.processor()
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        models = [
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        ]
        cpu = models[0] if models else cpu
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "system": platform.system(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "memory_gb": round(memory, 1),
    }


def manifest() -> dict:
    """The commit, whether the tree was clean (the records aside), the
    machine, JAX, the device it ran on and the XLA flags in force, which
    change what compiles and how fast."""
    import jax

    records = f":!{RESULTS.relative_to(ROOT)}"
    return {
        "commit": git("rev-parse", "HEAD"),
        "clean_tree": not git("status", "--porcelain", "--", ".", records),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "machine": machine(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def append(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def read_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def time_agent(label: str, n_seeds: int, steps: int) -> dict:
    """Train n_seeds one iteration past the agent's warm-up (its compile
    timed), then resume for steps rounded up to whole iterations (timed,
    its compile aside), both over the benchmark's horizon."""
    benchmark, agent = BENCHMARKS[label], build(label)
    unit = iteration_steps(agent)
    warm_up = round_up(warm_up_steps(agent) + unit, unit)
    measured = round_up(steps, unit)
    seeds = list(range(n_seeds))
    spans = compile_spans()
    spans.clear()
    began = time.perf_counter()
    state, _ = agent.train(
        seed=seeds, n_timesteps=warm_up, total_timesteps=benchmark.budget
    )
    warm_up_s = time.perf_counter() - began
    compile_s = covered_seconds(spans)
    spans.clear()
    began = time.perf_counter()
    agent.train(
        seed=seeds,
        n_timesteps=measured,
        total_timesteps=benchmark.budget,
        initial_state=state,
    )
    exec_s = time.perf_counter() - began - covered_seconds(spans)
    return {
        "agent": label,
        "env": benchmark.env_id,
        "n_seeds": n_seeds,
        "warm_up_steps": warm_up,
        "measured_steps": measured,
        "compile_s": round(compile_s, 1),
        "warm_up_s": round(warm_up_s, 1),
        "resume_compile_s": round(covered_seconds(spans), 1),
        "exec_s": round(exec_s, 3),
        # Each seed advances by the run's steps; the env steps count all.
        "run_steps_per_s": round(measured / exec_s, 1),
        "env_steps_per_s": round(n_seeds * measured / exec_s, 1),
        "peak_memory_gb": peak_memory_gb(),
        "device_peak_memory_gb": device_peak_memory_gb(),
        "manifest": manifest(),
    }


class AjaxTrainer:
    """The seeds trained through Ajax, their states kept as checkpoints in
    folder (lost with the machine). A chunk keeps the state it started from
    until the next one starts: a chunk whose record was lost after its
    checkpoint was saved is trained again from it."""

    def __init__(
        self,
        label: str,
        seeds: list[int],
        budget: int,
        eval_every: int,
        folder: pathlib.Path,
    ) -> None:
        self.agent, self.seeds, self.folder = build(label), seeds, folder
        self.budget, self.eval_every = budget, eval_every

    def checkpoint(self, step: int) -> pathlib.Path:
        return self.folder / f"step-{step}.pkl"

    def saved_steps(self) -> set[int]:
        saved = self.folder.glob("step-*.pkl")
        return {int(path.stem.removeprefix("step-")) for path in saved}

    def train(self, start: int, end: int) -> dict:
        import numpy as np

        from ajax.agents.base import shared_counters
        from ajax.checkpoint import load_checkpoint, save_checkpoint
        from ajax.logging.wandb_logging import LoggingConfig

        spans = compile_spans()
        spans.clear()
        horizon = {"total_timesteps": self.budget}
        state = None
        if start:
            state = load_checkpoint(self.agent, str(self.checkpoint(start)), **horizon)
        state, evaluations = self.agent.train(
            seed=self.seeds,
            n_timesteps=end - start,
            initial_state=state,
            logging_config=LoggingConfig(log_frequency=self.eval_every, backends=()),
            **horizon,
        )
        counters = shared_counters(state, self.agent.per_seed_update_counts)
        if int(counters["timestep"]) != end:
            raise RuntimeError(
                f"the chunk ended at {int(counters['timestep'])}, not {end}"
            )
        returns = np.asarray(evaluations[EVALUATION_KEY])
        if np.isnan(returns).any():
            raise RuntimeError("an evaluation slot stayed empty")
        save_checkpoint(state, str(self.checkpoint(end)))
        for path in self.folder.glob("step-*.pkl"):
            if path not in (self.checkpoint(start), self.checkpoint(end)):
                path.unlink()

        def rows(key: str) -> Optional[list]:
            return (
                None
                if key not in evaluations
                else (np.asarray(evaluations[key]).round(3).tolist())
            )

        return {
            "returns": returns.round(3).tolist(),
            "non_finite_steps": rows(NON_FINITE_KEY),
            "eval_lengths": rows(LENGTH_KEY),
            "compile_s": covered_seconds(spans),
        }


def resume_point(records: list[dict], saved_steps: set[int]) -> tuple[int, int]:
    """The attempt and the step the run goes on from: the last record's end,
    when it ended the run or the trainer kept its state; a new attempt from
    step 0 otherwise."""
    if not records:
        return 0, 0
    last = records[-1]
    if last["end"] == last["budget"] or last["end"] in saved_steps:
        return last["attempt"], last["end"]
    return last["attempt"] + 1, 0


def chunk_steps(
    previous: Optional[dict], remaining: int, eval_every: int, seconds: float
) -> int:
    """The next chunk: the whole evaluation intervals that fill seconds at
    the previous chunk's pace (one interval without a previous chunk), at
    least one, at most what remains."""
    intervals = 1
    if previous is not None:
        budget_s = seconds - previous["compile_s"]
        intervals = int(budget_s * previous["run_steps_per_s"]) // eval_every
    return min(max(intervals, 1) * eval_every, remaining)


def run_seeds(
    label: str,
    seeds: list[int],
    budget: int,
    eval_every: int,
    trainer: Any,
    path: pathlib.Path,
    publish: Callable[[pathlib.Path, str], None],
    chunk_s: float,
    stop_after_s: float,
    clock: Callable[[], float] = time.perf_counter,
    describe: Callable[[], dict] = manifest,
    warm_up: int = 0,
) -> bool:
    """Train seeds to budget chunk by chunk, appending each chunk's record to
    path and publishing it. True once the run reached its budget; False when
    the next chunk would end past stop_after_s (call again to go on).

    trainer, as AjaxTrainer: saved_steps(), the steps whose state it keeps;
    train(start, end), which trains from start (a fresh run at 0, else the
    state kept at start) to end, keeps the states at start and end, and
    returns {"returns": the chunk's evaluations, one row per seed,
    "non_finite_steps" and "eval_lengths": the same shape, or None for a
    metric the agent does not log, "compile_s": its seconds compiling}.
    warm_up: the steps before the agent's updates run."""
    records = read_records(path)
    if records and records[-1]["budget"] != budget:
        raise ValueError(
            f"{path} records a budget of {records[-1]['budget']}, not {budget}"
        )
    attempt, step = resume_point(records, trainer.saved_steps())
    began, chunks = clock(), 0
    while step < budget:
        # A chunk begun inside the warm-up ran steps without updates, faster
        # than training: sized at its pace, the next chunk would overrun its
        # time several fold. Only a chunk begun past the warm-up sets the
        # pace; until one has, chunks are one interval.
        paced = [record for record in records if record["start"] >= warm_up]
        previous = paced[-1] if paced else None
        steps = chunk_steps(previous, budget - step, eval_every, chunk_s)
        if chunks and previous is not None:
            predicted = previous["compile_s"] + steps / previous["run_steps_per_s"]
            if clock() - began + predicted > stop_after_s:
                return False
        context, chunk_began = describe(), clock()
        result = trainer.train(step, step + steps)
        wall_s = clock() - chunk_began
        record = {
            "agent": label,
            "seeds": seeds,
            "attempt": attempt,
            "start": step,
            "end": step + steps,
            "budget": budget,
            "eval_steps": list(range(step + eval_every, step + steps + 1, eval_every)),
            "returns": result["returns"],
            "non_finite_steps": result["non_finite_steps"],
            "eval_lengths": result["eval_lengths"],
            "wall_s": round(wall_s, 1),
            "compile_s": round(result["compile_s"], 1),
            "run_steps_per_s": round(
                steps / max(wall_s - result["compile_s"], 1e-3), 3
            ),
            "peak_memory_gb": peak_memory_gb(),
            "device_peak_memory_gb": device_peak_memory_gb(),
            "manifest": context,
        }
        append(path, record)
        publish(
            path, f"Record {label} seeds {seeds[0]}-{seeds[-1]} to step {step + steps}"
        )
        records.append(record)
        step, chunks = step + steps, chunks + 1
    return True


def commit_and_push(path: pathlib.Path, message: str) -> None:
    subprocess.run(["git", "add", str(path)], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--", str(path)], cwd=ROOT, check=True
    )
    if subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=ROOT).returncode:
        print(
            "push failed: the record is committed and goes with the next push",
            flush=True,
        )


def seed_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def range_name(seeds: list[int]) -> str:
    """The name of a seed range's records and checkpoints: one file per
    range, so that ranges run apart merge without conflict."""
    return f"seeds-{seeds[0]}-{seeds[-1]}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser("time", help="time an agent on its benchmark")
    timing.add_argument("label", choices=sorted(BENCHMARKS))
    timing.add_argument("--n-seeds", type=int, nargs="+", default=[1, 10])
    timing.add_argument("--steps", type=int, default=4096, help="run steps timed")
    running = commands.add_parser("run", help="train seeds A-B on the benchmark")
    running.add_argument("label", choices=sorted(BENCHMARKS))
    running.add_argument("--seeds", type=seed_range, required=True, help="A-B")
    running.add_argument("--budget", type=int, help="steps instead of the table's")
    running.add_argument("--chunk-minutes", type=float, default=90)
    running.add_argument("--stop-after-minutes", type=float, default=110)
    running.add_argument("--no-git", action="store_true", help="commit nothing")
    for command in (timing, running):
        command.add_argument("--results", type=pathlib.Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.command == "time":
        for n_seeds in args.n_seeds:
            record = time_agent(args.label, n_seeds, args.steps)
            append(args.results / "timing" / f"{args.label}.jsonl", record)
            print(json.dumps(record), flush=True)
        return 0
    benchmark, seeds = BENCHMARKS[args.label], args.seeds
    budget = args.budget or benchmark.budget
    eval_every = min(benchmark.eval_every, budget)
    if not args.no_git and git("rev-parse", "--abbrev-ref", "HEAD") == "main":
        parser.error("records are committed on a branch of their own, not on main")
    if not args.no_git and not manifest()["clean_tree"]:
        parser.error("commit first: every record names the commit it ran")
    name = range_name(seeds)
    folder = args.results / "checkpoints" / args.label / name
    trainer = AjaxTrainer(args.label, seeds, budget, eval_every, folder)
    finished = run_seeds(
        args.label,
        seeds,
        budget,
        eval_every,
        trainer,
        args.results / args.label / f"{name}.jsonl",
        (lambda *_: None) if args.no_git else commit_and_push,
        args.chunk_minutes * 60,
        args.stop_after_minutes * 60,
        warm_up=warm_up_steps(trainer.agent),
    )
    print(
        "finished" if finished else "unfinished: run the same command again", flush=True
    )
    return 0 if finished else 3


if __name__ == "__main__":
    sys.exit(main())
