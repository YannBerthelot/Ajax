"""Record TD-MPC2 update parity fixtures from the real paper-era reference code.

This script runs the **unmodified** ``TDMPC2`` agent of
``nicklashansen/tdmpc2@5f6fade`` (``tdmpc2/tdmpc2.py`` and
``tdmpc2/common/*``; its agent algorithm is identical to b67b21c, the commit
behind the paper's curves) for a few consecutive ``agent.update()`` calls on
fixed synthetic batches, and saves everything a JAX port needs to reproduce
those updates exactly to ``tests/agents/TDMPC2/fixtures/tdmpc2_update.npz``:

* the configuration, the numeric defaults of the reference's ``config.yaml``
  (the paper defaults; every non-fixture entry of the configuration is
  asserted equal to them) and the initial ``state_dict`` of every module
  (encoder, dynamics, reward, policy prior, online and target Q ensembles);
* the batches, fed through a fake buffer whose ``sample()`` returns them;
* every random draw of every update, recorded by wrapping the reference's own
  random calls without changing what they return: ``torch.randn_like`` (the
  policy's reparameterisation noise, ``common/world_model.py:134``) and
  ``np.random.choice`` (the random pair of Q heads, ``world_model.py:170``);
* per update, every quantity the reference computes and logs
  (``tdmpc2.py:282-290``), the gradient norms returned by both
  ``clip_grad_norm_`` calls (``tdmpc2.py:195, :271``), the TD targets, the
  policy loss' actions, log-probabilities and Q values, and the RunningScale
  value; and all parameters (world model, policy prior, target Q) after the
  last update.

Nothing in the reference is patched except, for a CPU-only torch, the
``torch.device('cuda')`` the agent hard-codes (``tdmpc2.py:19``,
``common/scale.py:9-10``): the ``torch`` global of those two modules is
replaced by a proxy whose ``device()`` returns the CPU device and which
forwards everything else to torch.

Fixture design (``docs/world_models/DESIGN.md`` §10):

* Tiny sizes: obs 5, action 2, latent 16, enc/mlp width 32, two encoder
  layers, 5 Q heads, 101 bins in [-10, 10], horizon 3, batch 8, discount fixed
  at 0.9 (``discount_min = discount_max = 0.9``); every other hyperparameter
  at the paper defaults (``config.yaml``).
* ``dropout = 0``: torch's dropout masks drawn inside ``torch.vmap`` cannot be
  recorded or injected, so dropout is tested separately on the Ajax side. The
  script does record one fact about it (``check_q_dropout_mode``): at 5f6fade
  the Q ensemble's dropout is active in **eval mode too**, because the
  functional module that ``combine_state_for_ensemble`` builds is hidden in
  the ``torch.vmap`` closure and never sees ``model.train()`` /
  ``model.eval()`` (``common/layers.py:12-21``). TD targets (target Q) and
  planning therefore run with dropout at the paper-era commit.
* The initial parameters are the reference's own initialisation
  (``torch.manual_seed(0)``) with the final weights of the reward head, of the
  online Q heads and of the target Q heads then overwritten with independent
  ``N(0, 0.6^2)`` draws. With the paper's zero-initialised heads the losses and
  the policy gradient are nearly degenerate for the first few updates and
  neither clip binds; with these heads the TD target depends on which ensemble
  (online or target) is used, the RunningScale moves, and the four updates
  cover the gradient-clipping cases: world-model clip not binding / binding,
  policy clip binding / not binding, and the paper-era clip-norm quirk
  (deviations.md §2) with a clipped and with an unclipped previous policy
  gradient. The script asserts these cases.
* Parameters are saved after the last update only (about 0.5 MB in all; one
  snapshot of every module is ~250 kB, most of it the two Q ensembles'
  101-bin output layers). A parameter error in an earlier update still shows
  in that update's recorded intermediates: the policy loss' Q values use its
  post-step Q, and the next update's TD targets, losses and policy samples use
  all its post-update parameters.

How to run (in a throwaway environment; this file is not collected by pytest
and is not importable from Ajax):

.. code-block:: bash

    git clone https://github.com/nicklashansen/tdmpc2 tdmpc2_ref
    git -C tdmpc2_ref checkout 5f6fadec0fec78304b4b53e8171d348b58cac486
    uv venv --python 3.11 tdmpc2_venv  # or: python3.11 -m venv tdmpc2_venv
    uv pip install --python tdmpc2_venv/bin/python torch==2.2.2 numpy==1.26.4
    tdmpc2_venv/bin/python docs/world_models/parity/tdmpc2_update_fixtures.py \
        --reference tdmpc2_ref/tdmpc2

(the CPU wheel of torch 2.2.2 suffices; functorch ships with it). The output
is deterministic for a given torch build.

Reference code: Copyright (c) Nicklas Hansen (2023), MIT License.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REFERENCE_COMMIT = "5f6fadec0fec78304b4b53e8171d348b58cac486"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "tests/agents/TDMPC2/fixtures/tdmpc2_update.npz"

OBS_DIM, ACTION_DIM, HORIZON, BATCH = 5, 2, 3, 8
N_UPDATES = 4
HEAD_STD = 0.6
SEED = 0

# The reference's config keys (config.yaml + the parser-derived fields the
# agent reads). Integers stay integers where config.yaml writes integers.
# Entries outside FIXTURE_SETTINGS equal config.yaml's (asserted in main()).
CONFIG: dict[str, Any] = {
    "obs": "state",
    "obs_dim": OBS_DIM,
    "action_dim": ACTION_DIM,
    "latent_dim": 16,
    "enc_dim": 32,
    "mlp_dim": 32,
    "num_enc_layers": 2,
    "num_q": 5,
    "dropout": 0.0,
    "simnorm_dim": 8,
    "num_bins": 101,
    "vmin": -10,
    "vmax": 10,
    "task_dim": 0,
    "multitask": False,
    "log_std_min": -10,
    "log_std_max": 2,
    "entropy_coef": 1e-4,
    "horizon": HORIZON,
    "batch_size": BATCH,
    "rho": 0.5,
    "lr": 3e-4,
    "enc_lr_scale": 0.3,
    "grad_clip_norm": 20,
    "tau": 0.01,
    "consistency_coef": 20,
    "reward_coef": 0.1,
    "value_coef": 0.1,
    "discount_denom": 5,
    "discount_min": 0.9,
    "discount_max": 0.9,
    "episode_length": 100,
    "iterations": 6,
}
# The fixture's tiny sizes, dropout off, batch and fixed discount; task_dim is
# the parser's single-task value (``common/parser.py:57-58``).
FIXTURE_SETTINGS = {
    "latent_dim",
    "enc_dim",
    "mlp_dim",
    "dropout",
    "batch_size",
    "discount_min",
    "discount_max",
    "task_dim",
}


class _CpuTorch(types.ModuleType):
    """``torch`` with ``device(...)`` mapped to the CPU; everything else forwarded."""

    def __init__(self) -> None:
        super().__init__("torch")

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def device(*_args: Any, **_kwargs: Any) -> torch.device:
        return torch.device("cpu")


class Recorder:
    """Wraps the reference's random calls and clip calls; records their results."""

    def __init__(self) -> None:
        self.draws: list[tuple[str, np.ndarray]] = []
        self.clips: list[dict[str, float]] = []
        self.pi_params: set[int] = set()
        self._randn_like = torch.randn_like
        self._choice = np.random.choice
        self._clip = torch.nn.utils.clip_grad_norm_

    def install(self) -> None:
        torch.randn_like = self.randn_like  # type: ignore[assignment]
        np.random.choice = self.choice  # type: ignore[assignment]
        torch.nn.utils.clip_grad_norm_ = self.clip_grad_norm_

    def randn_like(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self._randn_like(*args, **kwargs)
        self.draws.append(("eps", out.detach().numpy().copy()))
        return out

    def choice(self, *args: Any, **kwargs: Any) -> Any:
        out = self._choice(*args, **kwargs)
        self.draws.append(("pair", np.asarray(out).copy()))
        return out

    def clip_grad_norm_(self, parameters: Any, max_norm: float, *args: Any) -> Any:
        params = [p for p in parameters if p.grad is not None]
        is_pi = [id(p) in self.pi_params for p in params]
        sq_other = sum(
            float((p.grad.double() ** 2).sum()) for p, s in zip(params, is_pi) if not s
        )
        sq_pi = sum(
            float((p.grad.double() ** 2).sum()) for p, s in zip(params, is_pi) if s
        )
        norm = self._clip(params, max_norm, *args)
        post_sq = sum(float((p.grad.double() ** 2).sum()) for p in params)
        self.clips.append(
            {
                "norm": float(norm),
                "max_norm": float(max_norm),
                "sq_other": sq_other,
                "sq_pi": sq_pi,
                "all_pi": float(all(is_pi)),
                "post_sq": post_sq,
            }
        )
        return norm


def read_config_yaml(path: Path) -> dict[str, Any]:
    """The numeric top-level entries of the reference's flat ``config.yaml``."""
    entries: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line or line[0] in " #-" or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        try:
            number = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue  # strings, booleans, ``???``
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            entries[key] = number
    return entries


def _record_outputs(fn: Callable[..., Any], log: list[Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        log.append((args, kwargs, out))
        return out

    return wrapped


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def _state(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: _np(v) for k, v in model.state_dict().items()}


def make_batches(rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Fixed synthetic batches in the reference's ``buffer.sample()`` layout.

    ``obs (H+1, B, S)``, ``action (H, B, A)``, ``reward (H, B, 1)``, float32
    (``common/buffer.py`` returns time-major slices). Batch 1 has large rewards,
    two of them beyond the two-hot range ``|symlog(r)| > 10``, so the target
    clipping to the edge bins is exercised.
    """
    batches = []
    for k in range(N_UPDATES):
        scale = 30.0 if k == 1 else 1.0
        reward = scale * rng.normal(size=(HORIZON, BATCH, 1))
        if k == 1:
            reward[0, 0, 0], reward[2, 5, 0] = 3.0e4, -5.0e4
        batches.append(
            {
                "obs": rng.normal(size=(HORIZON + 1, BATCH, OBS_DIM)).astype(
                    np.float32
                ),
                "action": rng.uniform(-1, 1, size=(HORIZON, BATCH, ACTION_DIM)).astype(
                    np.float32
                ),
                "reward": reward.astype(np.float32),
            }
        )
    return batches


class FakeBuffer:
    """Stands in for ``common.buffer.Buffer``: ``sample()`` returns one batch."""

    def __init__(self, batch: dict[str, np.ndarray]) -> None:
        self.batch = batch

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        b = self.batch
        return (
            torch.from_numpy(b["obs"].copy()),
            torch.from_numpy(b["action"].copy()),
            torch.from_numpy(b["reward"].copy()),
            None,
        )


def check_q_dropout_mode(world_model_cls: Any, cfg: types.SimpleNamespace) -> bool:
    """True iff the 5f6fade Q ensemble applies dropout in eval mode.

    Builds a world model with dropout 0.5 and non-zero Q heads, switches it to
    eval mode and compares two forward passes of the online and of the target
    ensemble on the same input.
    """
    probe = types.SimpleNamespace(**{**vars(cfg), "dropout": 0.5})
    model = world_model_cls(probe)
    with torch.no_grad():
        model._Qs.params[-2].normal_()
        model._target_Qs.params[-2].copy_(model._Qs.params[-2])
    model.eval()
    assert not model._Qs.training and not model._target_Qs.training
    x = torch.randn(4, cfg.latent_dim + cfg.action_dim)
    with torch.no_grad():
        online = [model._Qs(x) for _ in range(2)]
        target = [model._target_Qs(x) for _ in range(2)]
    differs_online = bool((online[0] != online[1]).any())
    differs_target = bool((target[0] != target[1]).any())
    assert differs_online == differs_target
    return differs_online


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

    ref_agent.torch = _CpuTorch()
    ref_scale.torch = _CpuTorch()

    paper = read_config_yaml(args.reference / "config.yaml")
    for key, value in CONFIG.items():
        if key in paper and key not in FIXTURE_SETTINGS:
            assert value == paper[key], f"{key}: {value} != config.yaml {paper[key]}"

    cfg = types.SimpleNamespace(
        **{k: v for k, v in CONFIG.items() if k != "obs_dim"},
        obs_shape={"state": (OBS_DIM,)},
        bin_size=(CONFIG["vmax"] - CONFIG["vmin"]) / (CONFIG["num_bins"] - 1),
    )

    # The order of the stacked Q-ensemble tensors ``_Qs.params.<i>``: the
    # parameter order of one member (combine_state_for_ensemble, layers.py:15).
    member = ref_layers.mlp(
        cfg.latent_dim + cfg.action_dim,
        2 * [cfg.mlp_dim],
        cfg.num_bins,
        dropout=cfg.dropout,
    )
    q_param_names = [n for n, _ in member.named_parameters()]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dropout_in_eval = check_q_dropout_mode(WorldModel, cfg)
    assert dropout_in_eval, "expected the paper-era Q dropout to ignore eval()"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = ref_agent.TDMPC2(cfg)
    assert agent.discount == 0.9
    model = agent.model
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        model._reward[-1].weight.normal_(0.0, HEAD_STD, generator=generator)
        model._Qs.params[-2].normal_(0.0, HEAD_STD, generator=generator)
        model._target_Qs.params[-2].normal_(0.0, HEAD_STD, generator=generator)

    rec = Recorder()
    rec.pi_params = {id(p) for p in model._pi.parameters()}
    rec.install()
    td_log: list[Any] = []
    pi_log: list[Any] = []
    q_log: list[Any] = []
    agent._td_target = _record_outputs(agent._td_target, td_log)
    model.pi = _record_outputs(model.pi, pi_log)
    model.Q = _record_outputs(model.Q, q_log)

    out: dict[str, Any] = {
        "meta/config": np.array(json.dumps(CONFIG)),
        "meta/paper_config": np.array(json.dumps(paper)),
        "meta/reference_commit": np.array(REFERENCE_COMMIT),
        "meta/torch_version": np.array(torch.__version__),
        "meta/head_std": np.array(HEAD_STD, np.float32),
        "meta/discount": np.array(agent.discount, np.float32),
        "meta/q_dropout_active_in_eval_mode": np.array(dropout_in_eval),
        "meta/q_param_names": np.array(q_param_names),
    }
    out.update({f"init/{k}": v for k, v in _state(model).items()})

    batches = make_batches(np.random.default_rng(SEED))
    prev_pi_post_sq = 0.0
    for k, batch in enumerate(batches):
        rec.draws.clear()
        rec.clips.clear()
        td_log.clear()
        pi_log.clear()
        q_log.clear()
        stats = agent.update(FakeBuffer(batch))

        kinds = [kind for kind, _ in rec.draws]
        assert kinds == ["eps", "pair", "eps", "pair"], kinds
        (_, td_eps), (_, td_pair), (_, pi_eps), (_, pi_pair) = rec.draws
        assert td_eps.shape == (HORIZON, BATCH, ACTION_DIM)
        assert pi_eps.shape == (HORIZON + 1, BATCH, ACTION_DIM)
        assert len(rec.clips) == 2 and rec.clips[1]["all_pi"] == 1.0
        wm_clip, pi_clip = rec.clips
        assert len(pi_log) == 2 and len(q_log) == 3 and len(td_log) == 1
        _, pis, log_pis, _ = pi_log[1][2]

        prefix = f"update{k}/"
        out.update({prefix + f"batch/{n}": v for n, v in batch.items()})
        out[prefix + "draws/td_eps"] = td_eps
        out[prefix + "draws/td_pair"] = td_pair.astype(np.int32)
        out[prefix + "draws/pi_eps"] = pi_eps
        out[prefix + "draws/pi_pair"] = pi_pair.astype(np.int32)
        out[prefix + "td_targets"] = _np(td_log[0][2])
        out[prefix + "pi_actions"] = _np(pis)
        out[prefix + "pi_log_pis"] = _np(log_pis)
        out[prefix + "pi_q"] = _np(q_log[2][2])
        for name in ("consistency_loss", "reward_loss", "value_loss", "pi_loss"):
            out[prefix + name] = np.array(stats[name], np.float32)
        out[prefix + "total_loss"] = np.array(stats["total_loss"], np.float32)
        out[prefix + "grad_norm"] = np.array(stats["grad_norm"], np.float32)
        out[prefix + "pi_grad_norm"] = np.array(pi_clip["norm"], np.float32)
        out[prefix + "pi_scale"] = np.array(stats["pi_scale"], np.float32)
        out[prefix + "wm_own_grad_norm"] = np.array(
            np.sqrt(wm_clip["sq_other"]), np.float32
        )
        out[prefix + "stale_pi_grad_sq_norm"] = np.array(wm_clip["sq_pi"], np.float32)

        # The paper-era quirk: the world-model clip norm includes the previous
        # update's post-clip policy gradient (zero_grad of pi runs at the start
        # of update_pi, tdmpc2.py:184, and model.parameters() contains _pi).
        expected_sq = prev_pi_post_sq
        assert np.isclose(wm_clip["sq_pi"], expected_sq, rtol=1e-6)
        assert np.isclose(
            wm_clip["norm"] ** 2, wm_clip["sq_other"] + wm_clip["sq_pi"], rtol=1e-5
        )
        prev_pi_post_sq = pi_clip["post_sq"]
        print(
            f"update {k}: wm norm {wm_clip['norm']:.4f} "
            f"(own {np.sqrt(wm_clip['sq_other']):.4f}, stale pi "
            f"{np.sqrt(wm_clip['sq_pi']):.4f}), pi norm {pi_clip['norm']:.4f}, "
            f"scale {stats['pi_scale']:.5f}"
        )

    # Coverage of the clipping cases (see the module docstring).
    wm = [float(out[f"update{k}/grad_norm"]) for k in range(N_UPDATES)]
    own = [float(out[f"update{k}/wm_own_grad_norm"]) for k in range(N_UPDATES)]
    pi = [float(out[f"update{k}/pi_grad_norm"]) for k in range(N_UPDATES)]
    clip = CONFIG["grad_clip_norm"]
    assert wm[0] < clip, "update 0: the world-model clip must not bind"
    assert any(
        w > clip and o < clip for w, o in zip(wm, own)
    ), "some update must clip the world model only because of the stale pi grads"
    assert pi[0] > clip and pi[1] > clip, "updates 0-1: the pi clip must bind"
    assert any(p < clip for p in pi[1:-1]), "an unclipped pi gradient must follow"
    assert any(w < clip for w in wm[1:]), "a later non-binding world-model clip"
    out.update({f"final/{n}": v for n, v in _state(model).items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB, {len(out)} arrays)"
    )


if __name__ == "__main__":
    main()
