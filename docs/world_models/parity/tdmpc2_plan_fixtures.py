"""Record TD-MPC2 planning parity fixtures from the real paper-era reference code.

This script runs the **unmodified** ``TDMPC2.act()`` (``mpc = true``) of
``nicklashansen/tdmpc2@5f6fade`` (``tdmpc2/tdmpc2.py:70-171``: ``act``,
``_estimate_value``, ``plan``; agent algorithm identical to b67b21c) for a short
sequence of decisions and saves everything a JAX port needs to reproduce them
to ``tests/agents/TDMPC2/fixtures/tdmpc2_plan.npz``:

* the configuration (the update fixture's, plus the paper's planning defaults
  of ``config.yaml``: 6 iterations, 512 samples, 64 elites, 24 policy
  trajectories, std in [0.05, 2], temperature 0.5; asserted equal to
  ``config.yaml``) and the initial ``state_dict`` of every module;
* per decision: the observation, ``t0``, ``eval_mode``, the warm-start mean the
  agent held before the call, and every random draw, recorded by wrapping the
  reference's own random calls without changing what they return:
  ``torch.randn_like`` (the policy samples, ``common/world_model.py:134``),
  ``torch.randn`` (the Gaussian candidates and the exploration noise,
  ``tdmpc2.py:143, 170``), and ``np.random.choice`` (the pair of Q heads,
  ``world_model.py:170``, and the elite draw, ``tdmpc2.py:166``). For the elite
  draw the wrapper saves numpy's global state, reads the one uniform number
  ``choice(p=...)`` consumes, restores the state and calls the original, so
  the original draws exactly what it would have; the recorded uniform is
  checked to give the returned index by numpy's inverse-CDF rule;
* per decision, the outputs: the action, the new ``_prev_mean``, and per
  iteration the candidate values, the elite indices, values and scores, and
  the mean and std after the update. These are read from ``plan()``'s local
  variables by a line tracer (``sys.settrace``) at the ``for`` line of the MPPI
  loop and at the ``topk`` line, which reads values and changes nothing.

Nothing in the reference is patched except, as in
``tdmpc2_update_fixtures.py`` (whose helpers this script reuses), the CUDA
device the agent hard-codes.

Fixture design (``docs/world_models/DESIGN.md`` §10):

* Tiny networks of the update fixture (obs 5, action 2, latent 16, widths 32,
  5 Q heads, 101 bins, horizon 3), discount 0.9, ``dropout = 0`` (torch's
  dropout masks inside ``torch.vmap`` cannot be recorded; Ajax tests the
  dropout of the planner's Q pass on its own). Planning hyperparameters at the
  paper defaults.
* The reference's own initialisation (``torch.manual_seed(0)``) with the
  final weights of the reward head, of the online Q heads and of the policy
  prior overwritten by ``N(0, HEAD_STD^2)`` draws. With the paper's zero
  reward and Q heads every value is tied; with its ``N(0, 0.02^2)`` policy
  head (means near 0, log-std near -4) the 24 policy trajectories are
  near-identical, near-zero sequences that never rank among the elites.
  ``HEAD_STD = 0.3`` gives values of a few units and elite value ranges of
  0.1 to 2, so the temperature-0.5 scores spread over tens of elites and the
  weighted mean and std are exercised, and diverse policy trajectories.
* Decisions: ``t0`` training, two training warm starts, one evaluation-mode
  warm start, then an episode boundary (``t0`` training while the agent
  still holds the previous decision's non-zero ``_prev_mean``, which ``t0``
  must discard). ``_prev_mean`` carries over from one decision to the next.
* Coverage of the policy trajectories: a decision is kept only if at least
  one policy trajectory is an elite in at least one iteration, so the
  fixture checks how columns ``[:24]`` enter the mean / std update. (They
  rank among the elites in about half the decisions, mostly in the first
  iteration; once the Gaussian samples have converged they almost never do,
  so the drawn elite is never a policy trajectory here. Ajax's behaviour
  tests cover that case.) The per-iteration counts are saved.
* Robustness to float32 rounding (torch vs XLA, macOS vs Linux): a candidate
  whose value is within rounding of the 64th elite's could enter or leave the
  elite set, and an elite drawn next to a near-tie or with the uniform next
  to a CDF step could change. A decision is kept only if, in every iteration,
  the gap between the last elite and the first non-elite is at least
  ``MARGIN_REL * max(1, |value|)``, and in the last iteration the drawn elite
  is that far from its neighbours in value and its CDF interval contains the
  uniform with ``MARGIN_CDF`` to spare. Otherwise (or without a policy
  elite) the decision is run again from the same agent and RNG state with
  another observation (the number of attempts is saved). Near-ties *inside*
  the elite set do not matter: the mean, std and draw are invariant to their
  order, and the tests compare the elite sets.
* Ties: when tied values straddle the top-k boundary, torch's CPU ``topk``
  selects a different subset of them than ``lax.top_k`` (lower index first),
  not only a different order (spec 3.15). The script records its result for
  an all-tied vector, and that at the reference's zero-head initialisation
  (spec 3.23) every candidate value of every iteration is tied and the elites
  are the first 64 candidates in order, which ``lax.top_k`` reproduces.

How to run (in the throwaway environment of ``tdmpc2_update_fixtures.py``;
this file is not collected by pytest and is not importable from Ajax):

.. code-block:: bash

    tdmpc2_venv/bin/python docs/world_models/parity/tdmpc2_plan_fixtures.py \\
        --reference tdmpc2_ref/tdmpc2

The output is deterministic for a given torch build.

Reference code: Copyright (c) Nicklas Hansen (2023), MIT License.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import tdmpc2_update_fixtures as update_fixtures
import torch

REFERENCE_COMMIT = update_fixtures.REFERENCE_COMMIT
DEFAULT_OUT = update_fixtures.REPO_ROOT / "tests/agents/TDMPC2/fixtures/tdmpc2_plan.npz"

OBS_DIM, ACTION_DIM = update_fixtures.OBS_DIM, update_fixtures.ACTION_DIM
HEAD_STD = 0.3
SEED = update_fixtures.SEED
MARGIN_REL = 2e-4
MARGIN_CDF = 1e-3
MAX_ATTEMPTS = 500

# (t0, eval_mode) of each decision, in order.
DECISIONS = (
    (True, False),
    (False, False),
    (False, False),
    (False, True),
    (True, False),
)

PLANNING = {
    "mpc": True,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "min_std": 0.05,
    "max_std": 2,
    "temperature": 0.5,
}
CONFIG: dict[str, Any] = {**update_fixtures.CONFIG, **PLANNING}


class DrawRecorder:
    """Wraps the reference's random calls; records what they return.

    ``draws`` holds ``(kind, value)``: ``"eps"`` from ``torch.randn_like``,
    ``"randn"`` from ``torch.randn``, ``"pair"`` from ``np.random.choice``
    without ``p``, and ``"elite"`` (the uniform, then the returned index) from
    ``np.random.choice(p=...)``.
    """

    def __init__(self) -> None:
        self.draws: list[tuple[str, Any]] = []
        self._randn = torch.randn
        self._randn_like = torch.randn_like
        self._choice = np.random.choice

    def install(self) -> None:
        torch.randn = self.randn  # type: ignore[assignment]
        torch.randn_like = self.randn_like  # type: ignore[assignment]
        np.random.choice = self.choice  # type: ignore[assignment]

    def randn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self._randn(*args, **kwargs)
        self.draws.append(("randn", update_fixtures.to_numpy(out)))
        return out

    def randn_like(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self._randn_like(*args, **kwargs)
        self.draws.append(("eps", update_fixtures.to_numpy(out)))
        return out

    def choice(
        self, a: Any, size: Any = None, replace: bool = True, p: Any = None
    ) -> Any:
        if p is None:
            out = self._choice(a, size, replace)
            self.draws.append(("pair", np.asarray(out).copy()))
            return out
        # RandomState.choice with p and size=None draws one random_sample():
        # read it from a copy of the state, then let the original draw it.
        state = np.random.get_state()
        uniform = np.random.random_sample()
        np.random.set_state(state)
        out = self._choice(a, size, replace, p)
        cdf = np.asarray(p, np.float64).cumsum()
        cdf /= cdf[-1]
        assert int(out) == int(cdf.searchsorted(uniform, side="right"))
        self.draws.append(("elite", (float(uniform), int(out))))
        return out


class PlanTracer:
    """Reads ``plan()``'s locals at two lines, through ``sys.settrace``.

    At the MPPI ``for`` line (before the loop and after each iteration) it
    copies ``mean``, ``std`` and, after an iteration, ``elite_idxs``,
    ``elite_value`` and ``score``; at the ``topk`` line it copies ``value``.
    Reading locals does not change them.
    """

    def __init__(self, plan_fn: Callable[..., Any]) -> None:
        fn = inspect.unwrap(plan_fn)  # plan is wrapped by @torch.no_grad()
        self.code = fn.__code__
        lines, start = inspect.getsourcelines(fn)

        def line_of(text: str) -> int:
            hits = [start + i for i, line in enumerate(lines) if text in line]
            assert len(hits) == 1, (text, hits)
            return hits[0]

        self.loop_line = line_of("for _ in range(self.cfg.iterations)")
        self.topk_line = line_of("elite_idxs = torch.topk(")
        self.loop: list[dict[str, np.ndarray]] = []
        self.values: list[np.ndarray] = []
        self.training: list[bool] = []

    def __enter__(self) -> PlanTracer:
        self.loop.clear()
        self.values.clear()
        sys.settrace(self._global)
        return self

    def __exit__(self, *_exc: Any) -> None:
        sys.settrace(None)

    def _global(self, frame: types.FrameType, event: str, _arg: Any) -> Any:
        if event == "call" and frame.f_code is self.code:
            return self._local
        return None

    def _local(self, frame: types.FrameType, event: str, _arg: Any) -> Any:
        if event != "line":
            return self._local
        local = frame.f_locals
        if frame.f_lineno == self.loop_line:
            names = ["mean", "std", "elite_idxs", "elite_value", "score"]
            if not self.loop:
                names += ["pi_actions", "z"]
                self.training.append(bool(local["self"].model.training))
            snap = {n: update_fixtures.to_numpy(local[n]) for n in names if n in local}
            self.loop.append(snap)
        elif frame.f_lineno == self.topk_line:
            self.values.append(update_fixtures.to_numpy(local["value"])[:, 0])
        return self._local


def split_draws(
    draws: list[tuple[str, Any]], iterations: int, eval_mode: bool
) -> dict[str, np.ndarray]:
    """One decision's draws, in the reference's order, as fixture arrays."""
    h, p, n = CONFIG["horizon"], CONFIG["num_pi_trajs"], CONFIG["num_samples"]
    expected = (
        ["eps"] * h
        + ["randn", "eps", "pair"] * iterations
        + ["elite"]
        + ([] if eval_mode else ["randn"])
    )
    assert [kind for kind, _ in draws] == expected, [kind for kind, _ in draws]
    values = [v for _, v in draws]
    pi_eps = np.stack(values[:h])
    per_iter = values[h : h + 3 * iterations]
    candidate_eps = np.stack(per_iter[0::3])
    terminal_eps = np.stack(per_iter[1::3])
    q_pair = np.stack(per_iter[2::3]).astype(np.int32)
    uniform, rank = values[h + 3 * iterations]
    action_eps = (
        np.zeros(ACTION_DIM, np.float32) if eval_mode else values[-1].astype(np.float32)
    )
    assert pi_eps.shape == (h, p, ACTION_DIM)
    assert candidate_eps.shape == (iterations, h, n - p, ACTION_DIM)
    assert terminal_eps.shape == (iterations, n, ACTION_DIM)
    assert action_eps.shape == (ACTION_DIM,)
    return {
        "draws/pi_eps": pi_eps,
        "draws/candidate_eps": candidate_eps,
        "draws/terminal_eps": terminal_eps,
        "draws/q_pair": q_pair,
        "draws/elite_uniform": np.array(uniform, np.float64),
        "draws/action_eps": action_eps,
        "elite_rank": np.array(rank, np.int32),
    }


def trace_outputs(tracer: PlanTracer, iterations: int) -> dict[str, np.ndarray]:
    """Per-iteration statistics from the tracer's snapshots."""
    assert len(tracer.loop) == iterations + 1 and len(tracer.values) == iterations
    after = tracer.loop[1:]
    return {
        "z": tracer.loop[0]["z"][0],
        "init_mean": tracer.loop[0]["mean"],
        "init_std": tracer.loop[0]["std"],
        "pi_actions": tracer.loop[0]["pi_actions"],
        "value": np.stack(tracer.values),
        "elite_idx": np.stack([s["elite_idxs"] for s in after]).astype(np.int32),
        "elite_value": np.stack([s["elite_value"][:, 0] for s in after]),
        "score": np.stack([s["score"][:, 0] for s in after]),
        "mean": np.stack([s["mean"] for s in after]),
        "std": np.stack([s["std"] for s in after]),
    }


def margins(record: dict[str, np.ndarray]) -> tuple[float, float]:
    """``(value margin, cdf margin)`` of a decision (module docstring).

    The value margin is the smallest gap, relative to ``max(1, |value|)``,
    between the last elite and the first non-elite of any iteration and
    between the drawn elite and its neighbours in the last iteration.
    """
    e = CONFIG["num_elites"]
    gaps = []
    for value in record["value"]:
        top = np.sort(value.astype(np.float64))[::-1]
        gaps.append((top[e - 1] - top[e]) / max(1.0, abs(top[e - 1])))
    elite = record["elite_value"][-1].astype(np.float64)
    k = int(record["elite_rank"])
    for j in (k - 1, k + 1):
        if 0 <= j < e:
            gaps.append(abs(elite[k] - elite[j]) / max(1.0, abs(elite[k])))
    cdf = np.cumsum(record["score"][-1].astype(np.float64))
    cdf /= cdf[-1]
    u = float(record["draws/elite_uniform"])
    lower = cdf[k - 1] if k > 0 else 0.0
    return float(min(gaps)), float(min(u - lower, cdf[k] - u))


def tie_records(agent_cls: Any, cfg: types.SimpleNamespace) -> dict[str, np.ndarray]:
    """torch.topk's order on an all-tied vector, and the tie at initialisation.

    Runs one decision of a fresh agent with the paper's zero-initialised
    reward and Q heads and checks that every candidate value of every
    iteration is tied (spec 3.23).
    """
    n, e = CONFIG["num_samples"], CONFIG["num_elites"]
    tied = torch.topk(torch.full((n,), 0.37), e, dim=0).indices.numpy()
    agent = agent_cls(types.SimpleNamespace(**vars(cfg)))
    tracer = PlanTracer(agent_cls.plan)
    with tracer:
        agent.act(torch.zeros(OBS_DIM), t0=True, eval_mode=True)
    values = np.stack(tracer.values)
    assert np.all(values == values[:, :1]), "zero heads: all values tied"
    init_elites = np.stack([s["elite_idxs"] for s in tracer.loop[1:]])
    return {
        "ties/topk_all_tied": tied.astype(np.int32),
        "ties/init_elite_idx": init_elites.astype(np.int32),
        "ties/init_value": values[:, 0],
    }


def run_decision(
    agent: Any,
    recorder: DrawRecorder,
    tracer: PlanTracer,
    obs: np.ndarray,
    t0: bool,
    eval_mode: bool,
) -> dict[str, np.ndarray]:
    """One ``agent.act`` call and everything it drew and computed."""
    iterations = agent.cfg.iterations
    prev = getattr(agent, "_prev_mean", None)
    record: dict[str, np.ndarray] = {
        "obs": obs,
        "t0": np.array(t0),
        "eval_mode": np.array(eval_mode),
        "prev_mean_in": (
            np.zeros((CONFIG["horizon"], ACTION_DIM), np.float32)
            if prev is None
            else update_fixtures.to_numpy(prev)
        ),
    }
    recorder.draws.clear()
    with tracer:
        action = agent.act(torch.from_numpy(obs.copy()), t0=t0, eval_mode=eval_mode)
    record.update(split_draws(recorder.draws, iterations, eval_mode))
    record.update(trace_outputs(tracer, iterations))
    record["action"] = update_fixtures.to_numpy(action)
    record["prev_mean"] = update_fixtures.to_numpy(agent._prev_mean)
    assert np.array_equal(record["prev_mean"], record["mean"][-1])
    return record


def decide_with_margin(
    agent: Any,
    recorder: DrawRecorder,
    tracer: PlanTracer,
    rng: np.random.Generator,
    t0: bool,
    eval_mode: bool,
) -> dict[str, np.ndarray]:
    """One recorded decision whose margins are large enough and with a policy
    trajectory among the elites of some iteration (module docstring).

    Each attempt restarts from the agent's warm start and the global RNG
    states of the first attempt, with a new observation.
    """
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    prev: Optional[torch.Tensor] = getattr(agent, "_prev_mean", None)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        if prev is not None:
            agent._prev_mean = prev.clone()
        elif hasattr(agent, "_prev_mean"):
            del agent._prev_mean
        obs = rng.normal(size=OBS_DIM).astype(np.float32)
        record = run_decision(agent, recorder, tracer, obs, t0, eval_mode)
        value_margin, cdf_margin = margins(record)
        pi_elites = (record["elite_idx"] < CONFIG["num_pi_trajs"]).sum(axis=1)
        if value_margin >= MARGIN_REL and cdf_margin >= MARGIN_CDF and pi_elites.any():
            record["attempts"] = np.array(attempt, np.int32)
            record["value_margin"] = np.array(value_margin)
            record["cdf_margin"] = np.array(cdf_margin)
            record["pi_elites"] = pi_elites.astype(np.int32)
            return record
    raise RuntimeError(
        f"no observation with enough margin and a policy elite in {MAX_ATTEMPTS} tries"
    )


def record_decisions(
    agent: Any, recorder: DrawRecorder, tracer: PlanTracer
) -> dict[str, np.ndarray]:
    """The fixture's decisions, chained through the agent's warm start."""
    rng = np.random.default_rng(SEED + 2)
    out: dict[str, np.ndarray] = {}
    for d, (t0, eval_mode) in enumerate(DECISIONS):
        record = decide_with_margin(agent, recorder, tracer, rng, t0, eval_mode)
        print(
            f"decision {d} (t0={t0}, eval={eval_mode}): attempt "
            f"{int(record['attempts'])}, margins {float(record['value_margin']):.1e}"
            f" / {float(record['cdf_margin']):.1e}, elite rank "
            f"{int(record['elite_rank'])}, elite value range "
            f"{np.ptp(record['elite_value'], axis=1).round(3)}, policy elites "
            f"per iteration {record['pi_elites']}"
        )
        out.update({f"decision{d}/{k}": v for k, v in record.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="the tdmpc2/ directory of a 5f6fade checkout",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    head = subprocess.run(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == REFERENCE_COMMIT, f"reference is at {head}, need {REFERENCE_COMMIT}"

    sys.path.insert(0, str(args.reference.resolve()))
    import common.scale as ref_scale
    import tdmpc2 as ref_agent
    from common import layers as ref_layers
    from common.world_model import WorldModel

    ref_agent.torch = update_fixtures.CpuTorch()
    ref_scale.torch = update_fixtures.CpuTorch()

    paper = update_fixtures.read_config_yaml(args.reference / "config.yaml")
    for key, value in CONFIG.items():
        if key in paper and key not in update_fixtures.FIXTURE_SETTINGS:
            assert value == paper[key], f"{key}: {value} != config.yaml {paper[key]}"
    assert set(PLANNING) - {"mpc"} <= set(paper)

    cfg = types.SimpleNamespace(
        **{k: v for k, v in CONFIG.items() if k != "obs_dim"},
        obs_shape={"state": (OBS_DIM,)},
        bin_size=(CONFIG["vmax"] - CONFIG["vmin"]) / (CONFIG["num_bins"] - 1),
    )
    member = ref_layers.mlp(
        cfg.latent_dim + cfg.action_dim,
        2 * [cfg.mlp_dim],
        cfg.num_bins,
        dropout=cfg.dropout,
    )
    q_param_names = [n for n, _ in member.named_parameters()]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dropout_in_eval = update_fixtures.check_q_dropout_mode(WorldModel, cfg)
    assert dropout_in_eval, "expected the paper-era Q dropout to ignore eval()"
    ties = tie_records(ref_agent.TDMPC2, cfg)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = ref_agent.TDMPC2(cfg)
    assert agent.discount == 0.9 and agent.cfg.iterations == CONFIG["iterations"]
    model = agent.model
    init_state = update_fixtures.state_numpy(model)
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        model._reward[-1].weight.normal_(0.0, HEAD_STD, generator=generator)
        model._Qs.params[-2].normal_(0.0, HEAD_STD, generator=generator)
        model._pi[-1].weight.normal_(0.0, HEAD_STD, generator=generator)
    heads = {
        k: v
        for k, v in update_fixtures.state_numpy(model).items()
        if not np.array_equal(v, init_state[k])
    }
    q_head = f"_Qs.params.{q_param_names.index('2.weight')}"
    expected_heads = [q_head, "_reward.2.weight", "_pi.2.weight"]
    assert sorted(heads) == sorted(expected_heads), sorted(heads)

    recorder = DrawRecorder()
    recorder.install()
    tracer = PlanTracer(ref_agent.TDMPC2.plan)
    decisions = record_decisions(agent, recorder, tracer)
    assert not any(tracer.training), "act() plans with the model in eval mode"

    out: dict[str, Any] = {
        "meta/config": np.array(json.dumps(CONFIG)),
        "meta/paper_config": np.array(json.dumps(paper)),
        "meta/reference_commit": np.array(REFERENCE_COMMIT),
        "meta/torch_version": np.array(torch.__version__),
        "meta/head_std": np.array(HEAD_STD, np.float32),
        # A Python float in the reference (float64 discount powers).
        "meta/discount": np.array(agent.discount, np.float64),
        "meta/q_dropout_active_in_eval_mode": np.array(dropout_in_eval),
        "meta/q_param_names": np.array(q_param_names),
        "meta/margin_rel": np.array(MARGIN_REL),
        "meta/margin_cdf": np.array(MARGIN_CDF),
        "meta/n_decisions": np.array(len(DECISIONS)),
        **ties,
        **{f"init/{k}": v for k, v in init_state.items()},
        **{f"heads/{k}": v for k, v in heads.items()},
        **decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB, {len(out)} arrays)"
    )


if __name__ == "__main__":
    main()
