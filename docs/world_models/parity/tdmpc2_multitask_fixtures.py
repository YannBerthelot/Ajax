"""Record TD-MPC2 multi-task parity fixtures from the real paper-era reference code.

This script runs the **unmodified** ``TDMPC2`` agent of
``nicklashansen/tdmpc2@5f6fade`` with ``cfg.multitask = True`` (the paper's
multi-task machinery: ``common/world_model.py:19-23, 58-91, 122-148``,
``tdmpc2.py:26, 32-34, 94-171, 201-216``; agent algorithm identical to
b67b21c) on three synthetic tasks of different observation and action
dimensions, and saves everything a JAX port needs to reproduce it:

* ``tests/agents/TDMPC2/fixtures/tdmpc2_multitask_update.npz``: four
  consecutive ``agent.update()`` calls on fixed multi-task batches, as
  ``tdmpc2_update_fixtures.py`` records single-task updates (the same
  recorders: every ``torch.randn_like`` and ``np.random.choice`` draw, both
  clip norms, the TD targets, the policy loss' samples and Q values, every
  logged loss), plus the task-embedding table at four points of every update:
  on entry, after the pre-step renorm (read at the ``_td_target`` call, whose
  ``encode(obs[1:], task)`` made the update's first look-up), after the
  world-model Adam step (read on entry to ``update_pi``) and on exit (after
  the post-step renorm of ``update_pi``'s first look-up);
* ``tests/agents/TDMPC2/fixtures/tdmpc2_multitask_plan.npz``: five
  ``act(obs, t0, eval_mode, task)`` decisions (MPPI with the masked
  candidates, means and stds and the per-task discount), recorded as
  ``tdmpc2_plan_fixtures.py`` records single-task decisions (its draw
  recorder and ``plan()`` line tracer), plus the embedding table before and
  after each decision.

``nn.Embedding(max_norm=1)`` renormalises every looked-up row whose norm
exceeds 1 *in place* (``torch.embedding_renorm_``: ``w *= 1 / (|w| +
1e-7)``), at every look-up, outside autograd (tdmpc2_spec 1.22). The update
fixture is built so that this write-back binds at both points of an update
(asserted): row 1 starts at norm 1.6 (renormed at update 0's first look-up),
row 2 at norm 1.3 is absent from update 0's batch (so it is *not* renormed
there, and Adam leaves it unchanged: zero gradient, zero moments) and is
renormed by update 1's first look-up; the embedding directions are redrawn
until, in some update, the Adam step pushes a looked-up row back above norm
1, so that the post-step renorm binds too. Task 0 is absent from update 2's
batch, so its row moves by Adam momentum only. The plan fixture's first
decision looks up a row of norm 1.4: the reference renorms it in place at
``act`` (recorded), Ajax renorms it without persisting (deviation T15).

Nothing in the reference is patched except, as in the single-task
generators, the CUDA device the agent hard-codes (``tdmpc2.py:19, 33``,
``common/scale.py:9-10``): the ``torch`` global of those modules is replaced
by a proxy that maps ``device(...)`` and the ``device=`` of ``tensor(...)``
to the CPU and forwards everything else to torch.

Fixture design (``docs/world_models/DESIGN.md`` §7, §10):

* Three tasks with observation dims (5, 2, 7) and action dims (2, 4, 1):
  observations zero-padded at the end to 7 (``MultitaskWrapper._pad_obs``,
  ``envs/wrappers/multitask.py:44-47``), actions to 4 with prefix masks
  (``world_model.py:21-23``); per-task episode lengths 100, 200, 500 agent
  steps, i.e. the distinct discounts 0.95, 0.975, 0.99 of the paper's
  heuristic at its defaults (``tdmpc2.py:32-49``). ``task_dim = 11``,
  distinct from every other size of the fixtures (asserted by the parity
  test), so that no axis of the embedding table can be confused with
  another by shape. Batches mix the tasks per sample (``task (B,)``,
  ``common/buffer.py:81``); the stored actions are zero on the invalid dims.
* The tiny networks of the single-task fixtures (latent 16, widths 32, 5 Q
  heads, 101 bins, horizon 3, batch 8), ``dropout = 0`` (torch's dropout
  masks inside ``torch.vmap`` cannot be recorded) and every other
  hyperparameter at the paper defaults (``config.yaml``; asserted).
* Initial parameters: the reference's own initialisation
  (``torch.manual_seed(0)``) with, as in the single-task fixtures, the final
  weights of the reward, Q and target Q heads (update fixture, ``N(0,
  0.6^2)``) or of the reward, Q and policy heads (plan fixture, ``N(0,
  0.3^2)``) overwritten, and the embedding rows rescaled to the norms above
  (the paper's ``U(-0.02, 0.02)`` rows have norm ~0.03, so the renorm and
  the conditioning would barely act).
* Plan decisions: task 2 (one valid action dim of 4, its row renormed by
  the look-up) ``t0`` in eval mode, then a warm start with the
  training-mode exploration noise; task 1 (all four dims valid, its row
  below norm 1) ``t0`` (discarding task 2's non-zero warm start) and a warm
  start, in eval mode, the offline trainer's protocol; task 0 (two valid
  dims of 4) ``t0`` in eval mode. Each decision is kept, as in the
  single-task plan fixture, only with large tie margins and a policy
  trajectory among the elites of some iteration (else the decision is rerun
  from the same state with another observation). As for the update
  fixture, the embedding directions are redrawn until every decision finds
  such an observation: with some directions a task's MPPI converges onto
  values within ~1e-5 of each other for almost every observation, closer
  than float32 rounding allows the elite sets to be compared exactly.

How to run (in the throwaway environment of ``tdmpc2_update_fixtures.py``;
this file is not collected by pytest and is not importable from Ajax):

.. code-block:: bash

    tdmpc2_venv/bin/python docs/world_models/parity/tdmpc2_multitask_fixtures.py \\
        --reference tdmpc2_ref/tdmpc2

The output is deterministic for a given torch build.

Reference code: Copyright (c) Nicklas Hansen (2023), MIT License.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tdmpc2_plan_fixtures as plan_fixtures
import tdmpc2_update_fixtures as update_fixtures
import torch

REFERENCE_COMMIT = update_fixtures.REFERENCE_COMMIT
FIXTURES = update_fixtures.REPO_ROOT / "tests/agents/TDMPC2/fixtures"
DEFAULT_UPDATE_OUT = FIXTURES / "tdmpc2_multitask_update.npz"
DEFAULT_PLAN_OUT = FIXTURES / "tdmpc2_multitask_plan.npz"

TASKS = ("synthetic-a", "synthetic-b", "synthetic-c")
OBS_DIMS = (5, 2, 7)
ACTION_DIMS = (2, 4, 1)
EPISODE_LENGTHS = (100, 200, 500)
TASK_DIM = 11
OBS_DIM, ACTION_DIM = max(OBS_DIMS), max(ACTION_DIMS)
HORIZON, BATCH = 3, 8
N_UPDATES = 4
SEED = update_fixtures.SEED
UPDATE_HEAD_STD = update_fixtures.HEAD_STD
PLAN_HEAD_STD = plan_fixtures.HEAD_STD
MAX_ATTEMPTS = 50

# Per-sample task ids of each update's batch (module docstring).
BATCH_TASKS = (
    (0, 1, 0, 1, 1, 0, 0, 1),  # task 2 absent
    (2, 0, 1, 2, 0, 1, 2, 1),
    (1, 2, 2, 1, 2, 1, 1, 2),  # task 0 absent
    (0, 2, 1, 0, 1, 2, 0, 2),
)
UPDATE_EMB_NORMS = (0.5, 1.6, 1.3)
PLAN_EMB_NORMS = (0.7, 0.9, 1.4)
# (task, t0, eval_mode) of each decision, in order.
DECISIONS = (
    (2, True, True),
    (2, False, False),
    (1, True, True),
    (1, False, True),
    (0, True, True),
)

CONFIG: dict[str, Any] = {
    **update_fixtures.CONFIG,
    **plan_fixtures.PLANNING,
    "obs_dim": OBS_DIM,
    "action_dim": ACTION_DIM,
    "task_dim": TASK_DIM,
    "multitask": True,
    "discount_min": 0.95,
    "discount_max": 0.995,
}
# The tiny sizes, dropout off, the batch and the small task_dim (config.yaml:
# 96). The discount bounds are the paper's here (asserted).
FIXTURE_SETTINGS = {
    "latent_dim",
    "enc_dim",
    "mlp_dim",
    "dropout",
    "batch_size",
    "task_dim",
}


class CpuTorch(update_fixtures.CpuTorch):
    """The single-task proxy, plus ``tensor(..., device=...)`` on the CPU
    (``tdmpc2.py:33`` writes ``device='cuda'``)."""

    @staticmethod
    def tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
        if "device" in kwargs:
            kwargs["device"] = "cpu"
        return torch.tensor(*args, **kwargs)


def action_masks() -> np.ndarray:
    masks = np.zeros((len(TASKS), ACTION_DIM), np.float32)
    for i, dim in enumerate(ACTION_DIMS):
        masks[i, :dim] = 1.0
    return masks


def make_config() -> types.SimpleNamespace:
    """The reference's cfg for the multi-task agent (``parse_cfg`` and
    ``make_multitask_env`` write these fields, ``common/parser.py:51-59``,
    ``envs/__init__.py:34-52``)."""
    return types.SimpleNamespace(
        **{k: v for k, v in CONFIG.items() if k != "obs_dim"},
        obs_shape={"state": (OBS_DIM,)},
        bin_size=(CONFIG["vmax"] - CONFIG["vmin"]) / (CONFIG["num_bins"] - 1),
        tasks=list(TASKS),
        obs_shapes=list(OBS_DIMS),
        action_dims=list(ACTION_DIMS),
        episode_lengths=list(EPISODE_LENGTHS),
    )


def make_batches(rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Fixed multi-task batches in the reference's ``buffer.sample()`` layout.

    ``obs (H+1, B, S_max)`` zero-padded at the end per sample, ``action (H,
    B, A_max)`` zero on each sample's invalid dims, ``reward (H, B, 1)``
    (batch 1 with large rewards, two beyond the two-hot range, as in the
    single-task fixture), ``task (B,)`` int64.
    """
    batches = []
    for k, tasks in enumerate(BATCH_TASKS):
        task = np.asarray(tasks, np.int64)
        obs = rng.normal(size=(HORIZON + 1, BATCH, OBS_DIM)).astype(np.float32)
        obs_mask = np.arange(OBS_DIM)[None] < np.asarray(OBS_DIMS)[task][:, None]
        action = rng.uniform(-1, 1, size=(HORIZON, BATCH, ACTION_DIM))
        action = (action * action_masks()[task]).astype(np.float32)
        scale = 30.0 if k == 1 else 1.0
        reward = scale * rng.normal(size=(HORIZON, BATCH, 1))
        if k == 1:
            reward[0, 0, 0], reward[2, 5, 0] = 3.0e4, -5.0e4
        batches.append(
            {
                "obs": obs * obs_mask[None].astype(np.float32),
                "action": action,
                "reward": reward.astype(np.float32),
                "task": task,
            }
        )
    return batches


class FakeBuffer:
    """``common.buffer.Buffer`` stand-in: ``sample()`` returns one batch."""

    def __init__(self, batch: dict[str, np.ndarray]) -> None:
        self.batch = batch

    def sample(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.from_numpy(self.batch[k].copy())
            for k in ("obs", "action", "reward", "task")
        )


def set_embedding_norms(
    model: Any, norms: tuple[float, ...], generator: torch.Generator
) -> None:
    """Rows in random directions with the given norms (module docstring)."""
    with torch.no_grad():
        weight = model._task_emb.weight
        direction = torch.randn(weight.shape, generator=generator)
        direction /= direction.norm(dim=-1, keepdim=True)
        weight.copy_(direction * torch.tensor(norms)[:, None])


def table(model: Any) -> np.ndarray:
    return update_fixtures.to_numpy(model._task_emb.weight)


def uninstall(recorder: update_fixtures.Recorder) -> None:
    """Restore the functions ``Recorder.install`` wrapped."""
    torch.randn_like = recorder._randn_like  # type: ignore[assignment]
    np.random.choice = recorder._choice  # type: ignore[assignment]
    torch.nn.utils.clip_grad_norm_ = recorder._clip


def uninstall_draws(recorder: plan_fixtures.DrawRecorder) -> None:
    """Restore the functions ``DrawRecorder.install`` wrapped."""
    torch.randn = recorder._randn  # type: ignore[assignment]
    torch.randn_like = recorder._randn_like  # type: ignore[assignment]
    np.random.choice = recorder._choice  # type: ignore[assignment]


def renorm_points(
    tables: dict[str, np.ndarray], tasks: np.ndarray
) -> dict[str, list[int]]:
    """Rows the reference renormed before the TD target and after Adam."""
    looked_up = sorted({int(t) for t in tasks})

    def renormed(before: np.ndarray, after: np.ndarray) -> list[int]:
        rows = []
        for r in range(len(TASKS)):
            norm = float(np.linalg.norm(before[r]))
            if r in looked_up and norm > 1.0:
                np.testing.assert_allclose(after[r], before[r] / norm, rtol=1e-6)
                rows.append(r)
            else:
                np.testing.assert_array_equal(after[r], before[r])
        return rows

    return {
        "pre": renormed(tables["in"], tables["pre"]),
        "post": renormed(tables["adam"], tables["out"]),
    }


def record_updates(
    ref_agent: Any, attempt: int
) -> tuple[dict[str, Any], list[dict[str, list[int]]]]:
    """Four updates of a fresh agent; returns the fixture and the renorms."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = ref_agent.TDMPC2(make_config())
    model = agent.model
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        model._reward[-1].weight.normal_(0.0, UPDATE_HEAD_STD, generator=generator)
        model._Qs.params[-2].normal_(0.0, UPDATE_HEAD_STD, generator=generator)
        model._target_Qs.params[-2].normal_(0.0, UPDATE_HEAD_STD, generator=generator)
    set_embedding_norms(
        model, UPDATE_EMB_NORMS, torch.Generator().manual_seed(SEED + 100 + attempt)
    )
    discounts = update_fixtures.to_numpy(agent.discount)
    expected = [agent._get_discount(t) for t in EPISODE_LENGTHS]
    np.testing.assert_allclose(discounts, [0.95, 0.975, 0.99], rtol=1e-7)
    np.testing.assert_array_equal(discounts, np.asarray(expected, np.float32))
    np.testing.assert_array_equal(
        update_fixtures.to_numpy(model._action_masks), action_masks()
    )

    rec = update_fixtures.Recorder()
    rec.pi_params = {id(p) for p in model._pi.parameters()}
    rec.install()
    tables: dict[str, np.ndarray] = {}
    td_log: list[Any] = []
    pi_log: list[Any] = []
    q_log: list[Any] = []
    td_target, update_pi = agent._td_target, agent.update_pi

    def td_spy(next_z: Any, reward: Any, task: Any) -> Any:
        tables["pre"] = table(model)
        out = td_target(next_z, reward, task)
        td_log.append(out)
        return out

    def pi_spy(zs: Any, task: Any) -> Any:
        tables["adam"] = table(model)
        return update_pi(zs, task)

    agent._td_target = td_spy
    agent.update_pi = pi_spy
    model.pi = update_fixtures._record_outputs(model.pi, pi_log)
    model.Q = update_fixtures._record_outputs(model.Q, q_log)

    out: dict[str, Any] = {
        f"init/{k}": v for k, v in update_fixtures.state_numpy(model).items()
    }
    out["meta/discounts"] = discounts
    out["meta/embedding_attempt"] = np.array(attempt, np.int32)
    renorms = []
    batches = make_batches(np.random.default_rng(SEED))
    prev_pi_post_sq = 0.0
    try:
        for k, batch in enumerate(batches):
            for log in (rec.draws, rec.clips, td_log, pi_log, q_log):
                log.clear()
            tables["in"] = table(model)
            stats = agent.update(FakeBuffer(batch))
            tables["out"] = table(model)

            kinds = [kind for kind, _ in rec.draws]
            assert kinds == ["eps", "pair", "eps", "pair"], kinds
            (_, td_eps), (_, td_pair), (_, pi_eps), (_, pi_pair) = rec.draws
            assert td_eps.shape == (HORIZON, BATCH, ACTION_DIM)
            assert pi_eps.shape == (HORIZON + 1, BATCH, ACTION_DIM)
            assert len(rec.clips) == 2 and rec.clips[1]["all_pi"] == 1.0
            wm_clip, pi_clip = rec.clips
            assert len(pi_log) == 2 and len(q_log) == 3 and len(td_log) == 1
            _, pis, log_pis, _ = pi_log[1][2]
            invalid = action_masks()[batch["task"]] == 0
            assert np.all(update_fixtures.to_numpy(pis)[:, invalid] == 0)

            p = f"update{k}/"
            out.update({p + f"batch/{n}": v for n, v in batch.items()})
            out[p + "draws/td_eps"] = td_eps
            out[p + "draws/td_pair"] = td_pair.astype(np.int32)
            out[p + "draws/pi_eps"] = pi_eps
            out[p + "draws/pi_pair"] = pi_pair.astype(np.int32)
            out[p + "td_targets"] = update_fixtures.to_numpy(td_log[0])
            out[p + "pi_actions"] = update_fixtures.to_numpy(pis)
            out[p + "pi_log_pis"] = update_fixtures.to_numpy(log_pis)
            out[p + "pi_q"] = update_fixtures.to_numpy(q_log[2][2])
            for name in (
                "consistency_loss",
                "reward_loss",
                "value_loss",
                "pi_loss",
                "total_loss",
                "grad_norm",
                "pi_scale",
            ):
                out[p + name] = np.array(stats[name], np.float32)
            out[p + "pi_grad_norm"] = np.array(pi_clip["norm"], np.float32)
            out[p + "wm_own_grad_norm"] = np.array(
                np.sqrt(wm_clip["sq_other"]), np.float32
            )
            out[p + "stale_pi_grad_sq_norm"] = np.array(wm_clip["sq_pi"], np.float32)
            for point in ("in", "pre", "adam", "out"):
                out[p + f"task_emb/{point}"] = tables[point]
            # The paper-era clip norm counts the previous policy gradient
            # (as in the single-task fixture).
            assert np.isclose(wm_clip["sq_pi"], prev_pi_post_sq, rtol=1e-6)
            prev_pi_post_sq = pi_clip["post_sq"]
            renorms.append(renorm_points(tables, batch["task"]))
            print(
                f"  update {k}: tasks {sorted(set(batch['task'].tolist()))}, "
                f"renormed rows pre-step {renorms[-1]['pre']}, post-step "
                f"{renorms[-1]['post']}, row norms "
                f"{np.linalg.norm(tables['out'], axis=-1).round(4)}, wm norm "
                f"{wm_clip['norm']:.3f}, pi norm {pi_clip['norm']:.3f}"
            )
    finally:
        uninstall(rec)
    out.update({f"final/{n}": v for n, v in update_fixtures.state_numpy(model).items()})
    return out, renorms


def covered(renorms: list[dict[str, list[int]]], out: dict[str, Any]) -> bool:
    """The update fixture's coverage (module docstring)."""
    pre_ok = renorms[0]["pre"] == [1] and renorms[1]["pre"] == [2]
    post_ok = any(r["post"] for r in renorms)
    moved = not np.array_equal(
        out["update2/task_emb/in"][0], out["update2/task_emb/out"][0]
    )
    return pre_ok and post_ok and moved


# -- planning ---------------------------------------------------------------


class NoMarginError(RuntimeError):
    """No observation gave a decision with the margins and a policy elite."""


def split_draws(
    draws: list[tuple[str, Any]], iterations: int, eval_mode: bool
) -> dict[str, np.ndarray]:
    """One decision's draws in the reference's order (``A = A_max``)."""
    h, p, n = HORIZON, CONFIG["num_pi_trajs"], CONFIG["num_samples"]
    expected = (
        ["eps"] * h
        + ["randn", "eps", "pair"] * iterations
        + ["elite"]
        + ([] if eval_mode else ["randn"])
    )
    assert [kind for kind, _ in draws] == expected, [kind for kind, _ in draws]
    values = [v for _, v in draws]
    per_iter = values[h : h + 3 * iterations]
    uniform, rank = values[h + 3 * iterations]
    record = {
        "draws/pi_eps": np.stack(values[:h]),
        "draws/candidate_eps": np.stack(per_iter[0::3]),
        "draws/terminal_eps": np.stack(per_iter[1::3]),
        "draws/q_pair": np.stack(per_iter[2::3]).astype(np.int32),
        "draws/elite_uniform": np.array(uniform, np.float64),
        "draws/action_eps": (
            np.zeros(ACTION_DIM, np.float32)
            if eval_mode
            else values[-1].astype(np.float32)
        ),
        "elite_rank": np.array(rank, np.int32),
    }
    assert record["draws/pi_eps"].shape == (h, p, ACTION_DIM)
    assert record["draws/candidate_eps"].shape == (iterations, h, n - p, ACTION_DIM)
    assert record["draws/terminal_eps"].shape == (iterations, n, ACTION_DIM)
    return record


def run_decision(
    agent: Any,
    recorder: plan_fixtures.DrawRecorder,
    tracer: plan_fixtures.PlanTracer,
    obs: np.ndarray,
    task: int,
    t0: bool,
    eval_mode: bool,
) -> dict[str, np.ndarray]:
    """One ``agent.act(obs, t0, eval_mode, task)`` and what it drew and computed."""
    prev = getattr(agent, "_prev_mean", None)
    record: dict[str, np.ndarray] = {
        "obs": obs,
        "task": np.array(task, np.int32),
        "t0": np.array(t0),
        "eval_mode": np.array(eval_mode),
        "prev_mean_in": (
            np.zeros((HORIZON, ACTION_DIM), np.float32)
            if prev is None
            else update_fixtures.to_numpy(prev)
        ),
        "task_emb/in": table(agent.model),
    }
    recorder.draws.clear()
    with tracer:
        action = agent.act(
            torch.from_numpy(obs.copy()), t0=t0, eval_mode=eval_mode, task=task
        )
    record.update(split_draws(recorder.draws, agent.cfg.iterations, eval_mode))
    record.update(plan_fixtures.trace_outputs(tracer, agent.cfg.iterations))
    record["action"] = update_fixtures.to_numpy(action)
    record["prev_mean"] = update_fixtures.to_numpy(agent._prev_mean)
    record["task_emb/out"] = table(agent.model)
    assert np.array_equal(record["prev_mean"], record["mean"][-1])
    # Prefix masks: invalid dims are exactly 0 in the policy trajectories,
    # the means, the stds and the executed action (tdmpc2_spec 3.9).
    invalid = action_masks()[task] == 0
    for name in ("pi_actions", "mean", "std", "action"):
        assert np.all(record[name][..., invalid] == 0), name
    return record


def decide_with_margin(
    agent: Any,
    recorder: plan_fixtures.DrawRecorder,
    tracer: plan_fixtures.PlanTracer,
    rng: np.random.Generator,
    task: int,
    t0: bool,
    eval_mode: bool,
) -> dict[str, np.ndarray]:
    """``tdmpc2_plan_fixtures.decide_with_margin`` for a task: each attempt
    also restores the embedding table, which ``act`` renorms in place."""
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    prev: Optional[torch.Tensor] = getattr(agent, "_prev_mean", None)
    weight = agent.model._task_emb.weight.detach().clone()
    for attempt in range(1, plan_fixtures.MAX_ATTEMPTS + 1):
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        with torch.no_grad():
            agent.model._task_emb.weight.copy_(weight)
        if prev is not None:
            agent._prev_mean = prev.clone()
        elif hasattr(agent, "_prev_mean"):
            del agent._prev_mean
        obs = np.zeros(OBS_DIM, np.float32)
        obs[: OBS_DIMS[task]] = rng.normal(size=OBS_DIMS[task])
        record = run_decision(agent, recorder, tracer, obs, task, t0, eval_mode)
        value_margin, cdf_margin = plan_fixtures.margins(record)
        pi_elites = (record["elite_idx"] < CONFIG["num_pi_trajs"]).sum(axis=1)
        if (
            value_margin >= plan_fixtures.MARGIN_REL
            and cdf_margin >= plan_fixtures.MARGIN_CDF
            and pi_elites.any()
        ):
            record["attempts"] = np.array(attempt, np.int32)
            record["value_margin"] = np.array(value_margin)
            record["cdf_margin"] = np.array(cdf_margin)
            record["pi_elites"] = pi_elites.astype(np.int32)
            return record
    raise NoMarginError(
        f"task {task}: no observation with enough margin and a policy elite"
    )


def record_decisions(ref_agent: Any, attempt: int) -> dict[str, Any]:
    """The plan fixture: a fresh agent, its heads and the decisions, the
    embedding rows in the directions of attempt ``attempt`` (raises
    :class:`NoMarginError` when a decision finds no observation)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = ref_agent.TDMPC2(make_config())
    model = agent.model
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        model._reward[-1].weight.normal_(0.0, PLAN_HEAD_STD, generator=generator)
        model._Qs.params[-2].normal_(0.0, PLAN_HEAD_STD, generator=generator)
        model._pi[-1].weight.normal_(0.0, PLAN_HEAD_STD, generator=generator)
    set_embedding_norms(
        model, PLAN_EMB_NORMS, torch.Generator().manual_seed(SEED + 2 + attempt)
    )
    assert agent.cfg.iterations == CONFIG["iterations"]  # A_max = 4 < 20

    out: dict[str, Any] = {
        f"init/{k}": v for k, v in update_fixtures.state_numpy(model).items()
    }
    out["meta/discounts"] = update_fixtures.to_numpy(agent.discount)
    out["meta/embedding_attempt"] = np.array(attempt, np.int32)
    recorder = plan_fixtures.DrawRecorder()
    recorder.install()
    tracer = plan_fixtures.PlanTracer(ref_agent.TDMPC2.plan)
    rng = np.random.default_rng(SEED + 2)
    try:
        for d, (task, t0, eval_mode) in enumerate(DECISIONS):
            record = decide_with_margin(
                agent, recorder, tracer, rng, task, t0, eval_mode
            )
            print(
                f"  decision {d} (task {task}, t0={t0}, eval={eval_mode}): attempt "
                f"{int(record['attempts'])}, margins "
                f"{float(record['value_margin']):.1e} / "
                f"{float(record['cdf_margin']):.1e}, elite rank "
                f"{int(record['elite_rank'])}, policy elites per iteration "
                f"{record['pi_elites']}"
            )
            out.update({f"decision{d}/{k}": v for k, v in record.items()})
    finally:
        uninstall_draws(recorder)
    assert not any(tracer.training), "act() plans with the model in eval mode"
    # The first decision's look-up renormed task 2's row in place (T15).
    before, after = out["decision0/task_emb/in"], out["decision0/task_emb/out"]
    assert np.linalg.norm(before[2]) > 1.0
    np.testing.assert_allclose(
        after[2], before[2] / np.linalg.norm(before[2]), rtol=1e-6
    )
    np.testing.assert_array_equal(after[:2], before[:2])
    out["meta/n_decisions"] = np.array(len(DECISIONS))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="the tdmpc2/ directory of a 5f6fade checkout",
    )
    parser.add_argument("--update-out", type=Path, default=DEFAULT_UPDATE_OUT)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_OUT)
    args = parser.parse_args()
    # torch.tensor(tensor) in tdmpc2.py:102 warns; the result is a copy.
    warnings.filterwarnings("ignore", message="To copy construct", category=UserWarning)

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

    ref_agent.torch = CpuTorch()
    ref_scale.torch = CpuTorch()

    paper = update_fixtures.read_config_yaml(args.reference / "config.yaml")
    for key, value in CONFIG.items():
        if key in paper and key not in FIXTURE_SETTINGS:
            assert value == paper[key], f"{key}: {value} != config.yaml {paper[key]}"

    cfg = make_config()
    member = ref_layers.mlp(
        cfg.latent_dim + cfg.action_dim + cfg.task_dim,
        2 * [cfg.mlp_dim],
        cfg.num_bins,
        dropout=cfg.dropout,
    )
    meta: dict[str, Any] = {
        "meta/config": np.array(json.dumps(CONFIG)),
        "meta/paper_config": np.array(json.dumps(paper)),
        "meta/reference_commit": np.array(REFERENCE_COMMIT),
        "meta/torch_version": np.array(torch.__version__),
        "meta/q_param_names": np.array([n for n, _ in member.named_parameters()]),
        "meta/tasks": np.array(TASKS),
        "meta/obs_dims": np.array(OBS_DIMS, np.int32),
        "meta/action_dims": np.array(ACTION_DIMS, np.int32),
        "meta/episode_lengths": np.array(EPISODE_LENGTHS, np.int32),
        "meta/action_masks": action_masks(),
    }

    print("updates:")
    for attempt in range(MAX_ATTEMPTS):
        updates, renorms = record_updates(ref_agent, attempt)
        if covered(renorms, updates):
            break
        print(f"  attempt {attempt}: renorm coverage incomplete, redrawing")
    else:
        raise RuntimeError(
            f"no embedding directions with full coverage in {MAX_ATTEMPTS}"
        )
    updates["meta/head_std"] = np.array(UPDATE_HEAD_STD, np.float32)
    updates["meta/pre_step_renormed_rows"] = np.array(
        [len(r["pre"]) for r in renorms], np.int32
    )
    updates["meta/post_step_renormed_rows"] = np.array(
        [len(r["post"]) for r in renorms], np.int32
    )

    print("decisions:")
    for attempt in range(MAX_ATTEMPTS):
        try:
            decisions = record_decisions(ref_agent, attempt)
            break
        except NoMarginError as err:
            print(f"  attempt {attempt}: {err}, redrawing the embedding directions")
    else:
        raise RuntimeError(
            f"no embedding directions with decisions of enough margin in"
            f" {MAX_ATTEMPTS}"
        )
    decisions["meta/head_std"] = np.array(PLAN_HEAD_STD, np.float32)
    decisions["meta/margin_rel"] = np.array(plan_fixtures.MARGIN_REL)
    decisions["meta/margin_cdf"] = np.array(plan_fixtures.MARGIN_CDF)

    for path, data in ((args.update_out, updates), (args.plan_out, decisions)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **meta, **data)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB, {len(data)} arrays)")


if __name__ == "__main__":
    main()
