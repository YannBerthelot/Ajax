"""Chunked, checkpointed validation runs of the world-model agents (M9).

Shared by :mod:`paper_protocol` (the paper-protocol runs) and
:mod:`multitask_validation` (the multi-task pipeline's source runs and
offline training); ``docs/world_models/VALIDATION.md`` describes them.

A run trains ``k`` seeds of one agent at once (the agents' seed ``vmap``:
one jitted program per seed) and is split into **chunks** through the
agents' own resume path (``train(initial_state=...)``), which continues
every schedule from the absolute tick, so a chunked run computes what the
uninterrupted run computes (the agents' resume tests pin it). After each
chunk a *probe* measures the metric the reference reports and appends one
JSON line to ``curve.jsonl``. The state is saved with Ajax's checkpoint
utility (:mod:`ajax.checkpoint`) when :data:`SAVE_EVERY_S` seconds have
passed since the last save (DreamerV3's own ``run.save_every: 900``,
``29eb964:dreamerv3/configs.yaml:48``): a GPU run that is interrupted
resumes from its last save, drops the curve lines written after it (and a
last line cut short by the kill) and recomputes them. A finished run keeps its state only when asked (the
multi-task source runs export it); otherwise it deletes it (a DreamerV3
state holds its whole replay, ~2 GB per seed at the paper protocol).

Files of a run directory: ``run.json`` (the run's specification, written
once and atomically; a resume with another specification is refused), ``curve.jsonl``
(one line per chunk, each with its provenance: git commit, jax version,
device) and ``state.pkl`` (while the run is in progress).
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REFERENCE_DIR = os.path.join(HERE, "references")

#: Seconds between two state saves (DreamerV3's ``run.save_every``).
SAVE_EVERY_S = 900.0

RUN_FILE = "run.json"
CURVE_FILE = "curve.jsonl"
STATE_FILE = "state.pkl"


# ---------------------------------------------------------------------------
# References and provenance
# ---------------------------------------------------------------------------


def load_reference(name: str) -> dict[str, Any]:
    """A committed reference file (``extract_references.py``):
    ``"dreamerv3_dmc_proprio"`` or ``"tdmpc2_dmc"``."""
    with open(os.path.join(REFERENCE_DIR, f"{name}.json")) as f:
        return json.load(f)


def jsonable(value: Any) -> Any:
    """``value`` as JSON-compatible data (dataclasses as dicts, tuples as
    lists, NumPy scalars as numbers, anything else as its ``str``)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def provenance() -> dict[str, Any]:
    """Git commit (and whether the tree had changes), JAX version, backend
    and device of this process."""
    import jax

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    device = jax.devices()[0]
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": getattr(device, "device_kind", str(device)),
        "device_count": jax.device_count(),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# The generic chunked run
# ---------------------------------------------------------------------------


def read_records(path: str) -> list[dict]:
    """The records of a ``curve.jsonl`` (``[]`` if there is none).

    A last line that does not parse is a write cut short (a killed process,
    a full disk) and is dropped: it came after the last state save, so the
    resume recomputes it. A bad line before the last raises.
    """
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    return records


def write_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file: an interruption
    leaves the old file or the new one, never a partial one."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_records(path: str, records: list[dict]) -> None:
    write_atomic(path, "".join(json.dumps(r) + "\n" for r in records))


def provenance_summary(records: Sequence[dict]) -> dict[str, Any]:
    """The distinct provenances of a run's records: ``git_sha`` (in order of
    appearance), whether any record came from a tree with changes, the JAX
    versions and devices."""

    def distinct(key: str) -> list:
        seen: list = []
        for record in records:
            value = record.get("provenance", {}).get(key)
            if value not in seen:
                seen.append(value)
        return seen

    return {
        "git_sha": distinct("git_sha"),
        "dirty": any(r.get("provenance", {}).get("git_dirty") for r in records),
        "jax": distinct("jax"),
        "device": distinct("device"),
    }


def spec_differences(expected: Any, stored: Any, prefix: str = "") -> list[str]:
    """The keys (dotted paths) where two JSON specifications differ."""
    expected, stored = jsonable(expected), jsonable(stored)
    if isinstance(expected, dict) and isinstance(stored, dict):
        out = []
        for key in sorted(set(expected) | set(stored)):
            path = f"{prefix}{key}"
            if key not in expected or key not in stored:
                out.append(path)
            else:
                out += spec_differences(expected[key], stored[key], f"{path}.")
        return out
    return [] if expected == stored else [prefix.rstrip(".") or "<root>"]


class Probe:
    """Measures a run's metric after each chunk.

    :meth:`start` is called once per :func:`run_chunked` call, with the
    restored state of a resumed run (``None`` for a fresh run), and may keep
    what :meth:`measure` needs from the previous chunk.
    """

    def start(self, state: Any, progress: int) -> None:
        """Before the first chunk of this call."""

    def measure(self, state: Any, progress: int) -> dict[str, Any]:
        """The record's metric fields after a chunk ending at ``progress``."""
        raise NotImplementedError


def check_specification(out_dir: str, spec: dict[str, Any]) -> None:
    """Write ``run.json`` for a new run directory; refuse to continue a run
    of another specification."""
    path = os.path.join(out_dir, RUN_FILE)
    spec = jsonable(spec)
    if os.path.isfile(path):
        with open(path) as f:
            stored = json.load(f)["spec"]
        if stored != spec:
            raise ValueError(
                f"{out_dir} holds a run of another specification: start it in"
                " a new directory (or delete the old one)"
            )
        return
    os.makedirs(out_dir, exist_ok=True)
    write_atomic(path, json.dumps({"spec": spec, "provenance": provenance()}, indent=1))


def run_chunked(
    out_dir: str,
    *,
    spec: dict[str, Any],
    budget: int,
    chunk: int,
    train: Callable[[Any, int], Any],
    skeleton: Callable[[], Any],
    progress_of: Callable[[Any], int],
    probe: Probe,
    save_every_s: float = SAVE_EVERY_S,
    keep_state: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], Any]:
    """Run ``budget`` units in chunks of ``chunk``, resuming from ``out_dir``.

    Args:
        out_dir: the run directory (module docstring).
        spec: the run's specification (``run.json``).
        budget, chunk: the run length and the chunk, in the agent's unit.
        train: ``(state or None, n) -> state``: ``n`` more units from
            ``state`` (``None``: a fresh run).
        skeleton: ``() -> state``: a state of the right structure to restore
            a checkpoint into (the agents' ``train(n_timesteps=0)``).
        progress_of: the units a state has done.
        probe: measures the record's metric fields after each chunk
            (:class:`Probe`).
        save_every_s: seconds between state saves (0: after every chunk).
        keep_state: keep the final state on disk (else it is deleted).
        log: progress messages.

    Returns:
        ``(records, state)``: every chunk's record and the final state
        (``None`` when the run had finished before this call and its state
        was not kept).
    """
    if chunk <= 0 or budget <= 0:
        raise ValueError(f"budget {budget} and chunk {chunk} must be positive")
    check_specification(out_dir, spec)
    curve_path = os.path.join(out_dir, CURVE_FILE)
    state_path = os.path.join(out_dir, STATE_FILE)
    records = read_records(curve_path)

    from ajax.checkpoint import checkpoint_exists, restore_into, save_checkpoint

    state: Any = None
    progress = 0
    if checkpoint_exists(state_path):
        state = restore_into(skeleton(), state_path)
        progress = progress_of(state)
        log(f"{out_dir}: resuming at {progress} of {budget}")
    elif records and records[-1]["progress"] >= budget:
        return records, None
    kept = [r for r in records if r["progress"] <= progress]
    if len(kept) != len(records):
        log(f"{out_dir}: dropping {len(records) - len(kept)} records after the save")
    records = kept
    if os.path.isfile(curve_path):  # also drops a last line cut short
        _write_records(curve_path, records)

    probe.start(state, progress)
    last_save = time.monotonic()
    while progress < budget:
        n = min(chunk, budget - progress)
        start = time.perf_counter()
        state = train(state, n)
        progress += n
        if progress_of(state) != progress:
            raise AssertionError(
                f"the state reports {progress_of(state)} units, expected {progress}"
            )
        fields = probe.measure(state, progress)
        record = {
            "progress": progress,
            **fields,
            "wall_s": round(time.perf_counter() - start, 2),
            "provenance": provenance(),
        }
        records.append(record)
        with open(curve_path, "a") as f:
            f.write(json.dumps(jsonable(record)) + "\n")
        final = progress >= budget
        if (final and keep_state) or (
            not final and time.monotonic() - last_save >= save_every_s
        ):
            save_checkpoint(state, state_path)
            last_save = time.monotonic()
        log(f"{out_dir}: {progress}/{budget} {_summary(fields)}")
    if not keep_state and os.path.isfile(state_path):
        os.remove(state_path)
    return records, state


def _summary(fields: dict[str, Any]) -> str:
    value = fields.get("value")
    if value is None:
        return ""
    return "value per seed " + ", ".join(f"{v:.1f}" for v in value)


# ---------------------------------------------------------------------------
# Single-task runs (DreamerV3, TD-MPC2)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One validation run: an agent on one task, ``seeds`` vmapped.

    Attributes:
        agent: ``"DreamerV3"`` or ``"TDMPC2"`` (classes exported by ``ajax``).
        env_id: the mujoco_playground env id.
        reference_task: the task's name in the reference file.
        description: one line for ``--list``.
        kwargs: constructor arguments besides ``env_id``.
        budget: run length in the agent's unit (DreamerV3: rows, reset
            rows included; TD-MPC2: agent steps), :attr:`unit`.
        chunk: units per chunk (one record each).
        seeds: the seeds, trained together.
        num_eval_episodes: TD-MPC2's evaluation episodes per record.
        smoke: a tiny plumbing run (no verdict).
    """

    agent: str
    env_id: str
    reference_task: str
    description: str
    kwargs: dict
    budget: int
    chunk: int
    seeds: tuple[int, ...]
    num_eval_episodes: int = 10
    smoke: bool = False

    @property
    def action_repeat(self) -> int:
        return int(self.kwargs.get("action_repeat", 1))

    @property
    def unit(self) -> str:
        """The unit of ``budget``, ``chunk`` and the records' ``progress``;
        ``env_frames`` (= units x action repeat) is the references' env-step
        axis."""
        return "rows" if self.agent == "DreamerV3" else "agent_steps"

    @property
    def metric(self) -> str:
        """The reference's metric (``VALIDATION.md``)."""
        return "train_episode_return" if self.agent == "DreamerV3" else "eval_return"

    def env_frames(self, units: int) -> int:
        """Simulator steps after ``units`` (DreamerV3: rows x repeat, its
        reference's clock, DESIGN.md section 2; TD-MPC2: agent steps x
        repeat)."""
        return int(units) * self.action_repeat


def build_agent(spec: RunSpec) -> Any:
    import ajax

    return getattr(ajax, spec.agent)(env_id=spec.env_id, **spec.kwargs)


def _seed_values(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def units_done(state: Any) -> int:
    """The units a (seed-batched) single-task state has done: the
    collector's ``timestep`` (DreamerV3 rows, TD-MPC2 agent steps), equal
    across seeds."""
    done = np.unique(_seed_values(state.collector_state.timestep))
    if done.size != 1:
        raise ValueError(f"the seeds are at different steps {done}")
    return int(done[0])


class TrainEpisodeProbe(Probe):
    """DreamerV3's metric: the returns of the training episodes that ended
    during the chunk (the stochastic policy's, dreamerv3_spec 7.3, 7.7).

    Read from the row collector's per-env rolling window of the last ``W``
    episode returns (``RowCollectorState.episodic_return_state``: ``buffer``
    ``[W, n_envs, 1]``, write position ``index``). On fixed-length,
    non-terminating tasks (DMC) an env's ``k``-th episode ends on tick
    ``k (T + 1) + T - 1`` (``T`` agent steps and the held reset row of the
    dynamic reset mode, DESIGN.md section 5.2), so after ``t`` ticks
    ``(t + 1) // (T + 1)`` episodes have ended; the probe checks the
    window's write position against that count (a termination or another
    episode length raises) and reads the chunk's episodes, at most ``W`` per
    env (:func:`check_run` checks the chunk length).
    """

    def __init__(self, n_envs: int, episode_length: int) -> None:
        self.n_envs = n_envs
        self.period = episode_length + 1
        self._index: Optional[np.ndarray] = None
        self._ended = 0

    def episodes_after(self, rows: int) -> int:
        """Episodes each env has ended after ``rows`` rows (all envs)."""
        return (rows // self.n_envs + 1) // self.period

    @staticmethod
    def _window(state: Any) -> tuple[np.ndarray, np.ndarray]:
        window = state.collector_state.episodic_return_state
        returns = np.asarray(window.buffer)[..., 0]  # [S, W, n_envs]
        index = np.asarray(window.index)[..., 0].astype(np.int64)  # [S, n_envs]
        return returns, index

    def start(self, state: Any, progress: int) -> None:
        if state is None:
            self._index, self._ended = None, 0
        else:
            self._index = self._window(state)[1]
            self._ended = self.episodes_after(progress)

    def measure(self, state: Any, progress: int) -> dict[str, Any]:
        returns, index = self._window(state)
        size = returns.shape[1]
        previous = np.zeros_like(index) if self._index is None else self._index
        ended = self.episodes_after(progress)
        count = ended - self._ended
        if count > size:
            raise ValueError(
                f"{count} episodes per env ended in one chunk, more than the"
                f" {size} the collector's return window keeps: use shorter chunks"
            )
        if np.any((index - previous) % size != count % size):
            raise ValueError(
                "the training episodes do not end every T + 1 rows (a"
                " termination?): the per-chunk returns of fixed-length tasks"
                " cannot be read"
            )
        self._index, self._ended = index, ended
        # The chunk's episodes: the `count` entries before `index`.
        slots = (index[:, None, :] - 1 - np.arange(count)[None, :, None]) % size
        chunk_returns = np.take_along_axis(returns, slots, axis=1)  # [S, c, n]
        total = chunk_returns.reshape(chunk_returns.shape[0], -1).sum(axis=1)
        episodes = count * self.n_envs
        value = total / episodes if episodes else np.full(total.shape, np.nan)
        return {
            "episodes": [episodes] * total.shape[0],
            "return_sum": total.tolist(),
            "value": value.tolist(),
        }


def tdmpc2_evaluator(agent: Any, num_episodes: int) -> Callable[[Any, Any], dict]:
    """``(states, keys) -> metrics``: ``num_episodes`` evaluation episodes of
    the planner in ``eval_mode`` per seed
    (:func:`~ajax.agents.TDMPC2.train_TDMPC2.evaluate_tdmpc2`, the agent's
    own evaluation, tdmpc2_spec 4.21), every seed at once."""
    import jax

    from ajax.agents.TDMPC2.train_TDMPC2 import evaluate_tdmpc2

    def evaluate(state: Any, key: Any) -> dict:
        return evaluate_tdmpc2(
            state,
            key,
            env_args=agent.env_args,
            config=agent.agent_config,
            gamma=agent.gamma,
            num_episodes=num_episodes,
        )

    return jax.jit(jax.vmap(evaluate))


def eval_keys(seeds: Sequence[int], progress: int) -> Any:
    """The evaluation keys after ``progress`` agent steps:
    ``fold_in(PRNGKey(s), progress)`` per seed ``s``, so a resumed run
    evaluates with the uninterrupted run's keys."""
    import jax

    return jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), progress))(
        np.asarray(seeds)
    )


class EvalProbe(Probe):
    """TD-MPC2's metric: ``evaluate(states, keys)``
    (:func:`tdmpc2_evaluator`) with the keys of :func:`eval_keys`."""

    def __init__(
        self,
        evaluate: Callable[[Any, Any], dict],
        seeds: Sequence[int],
        num_episodes: int,
    ) -> None:
        self._evaluate = evaluate
        self.seeds = list(seeds)
        self.num_episodes = num_episodes

    def measure(self, state: Any, progress: int) -> dict[str, Any]:
        import jax

        out = jax.device_get(self._evaluate(state, eval_keys(self.seeds, progress)))
        return {
            "episodes": [self.num_episodes] * len(self.seeds),
            "value": np.asarray(out["Eval/episodic mean reward"]).tolist(),
            "length": np.asarray(out["Eval/mean episodic length"]).tolist(),
        }


class _RecordProbe(Probe):
    """A single-task probe's fields plus the run's progress fields."""

    def __init__(self, inner: Probe, spec: RunSpec, agent: Any) -> None:
        self.inner, self.spec, self.agent = inner, spec, agent

    def start(self, state: Any, progress: int) -> None:
        self.inner.start(state, progress)

    def measure(self, state: Any, progress: int) -> dict[str, Any]:
        spec = self.spec
        return {
            "env_frames": spec.env_frames(progress),
            "unit": spec.unit,
            "metric": spec.metric,
            "seeds": list(spec.seeds),
            **self.inner.measure(state, progress),
            "n_updates": _seed_values(state.n_updates).tolist(),
            "replay_bytes_per_seed": getattr(self.agent, "replay_bytes_per_seed", None),
        }


def check_run(spec: RunSpec, agent: Any) -> None:
    """Raise unless the chunks fit the agent: whole ticks; TD-MPC2 chunks of
    whole episodes; DreamerV3 chunks whose episodes its probe can read."""
    from ajax.environments.utils import agent_episode_length

    n_envs = agent.env_args.n_envs
    if spec.chunk % n_envs or spec.budget % n_envs:
        raise ValueError(f"{spec}: budget and chunk must be multiples of n_envs")
    if spec.agent == "TDMPC2":
        episode = agent.agent_episode_length * n_envs
        if spec.chunk % episode or spec.budget % spec.chunk:
            raise ValueError(
                f"TD-MPC2 chunks are evaluated on episode boundaries: chunk"
                f" {spec.chunk} must be a multiple of T * n_envs = {episode}"
                f" and divide the budget {spec.budget}"
            )
    else:
        env_args = agent.env_args
        period = (
            agent_episode_length(
                env_args.env, env_args.env_params, env_args.action_repeat
            )
            + 1
        )
        most = spec.chunk // n_envs // period + 1
        if most > 10:  # the row collector's return window (init_row_collector_state)
            raise ValueError(
                f"chunks of {spec.chunk // n_envs} rows per env can end {most}"
                " episodes per env, more than the 10 the return window keeps"
            )


def run_single_task(
    spec: RunSpec,
    out_dir: str,
    *,
    save_every_s: float = SAVE_EVERY_S,
    keep_state: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], Any, Optional[Any]]:
    """Run (or resume) ``spec`` in ``out_dir``; ``(records, agent, state)``.

    Each record holds the run's progress (``progress`` in the agent's unit,
    ``env_frames``), the reference's metric per seed (``value``, with
    ``episodes`` and, for DreamerV3, ``return_sum``) and its provenance;
    ``run.json`` holds the specification and the agent's resolved
    configuration.
    """
    from ajax.environments.utils import agent_episode_length

    agent = build_agent(spec)
    check_run(spec, agent)
    seeds = list(spec.seeds)
    if spec.agent == "DreamerV3":
        env_args = agent.env_args
        probe: Probe = TrainEpisodeProbe(
            env_args.n_envs,
            agent_episode_length(
                env_args.env, env_args.env_params, env_args.action_repeat
            ),
        )
    else:
        probe = EvalProbe(
            tdmpc2_evaluator(agent, spec.num_eval_episodes),
            seeds,
            spec.num_eval_episodes,
        )

    def train(state: Any, n: int) -> Any:
        return agent.train(seed=seeds, n_timesteps=n, initial_state=state)[0]

    def skeleton() -> Any:
        return agent.train(seed=seeds, n_timesteps=0)[0]

    run_spec = {
        "run": dataclasses.asdict(spec),
        "resolved": {
            "config": jsonable(agent.config),
            "agent_config": jsonable(getattr(agent, "agent_config", None)),
            "dreamer_config": jsonable(getattr(agent, "dreamer_config", None)),
        },
    }
    records, state = run_chunked(
        out_dir,
        spec=run_spec,
        budget=spec.budget,
        chunk=spec.chunk,
        train=train,
        skeleton=skeleton,
        progress_of=units_done,
        probe=_RecordProbe(probe, spec, agent),
        save_every_s=save_every_s,
        keep_state=keep_state,
        log=log,
    )
    return records, agent, state
