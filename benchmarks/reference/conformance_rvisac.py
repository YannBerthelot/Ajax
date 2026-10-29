"""yhisaki/average-reward-drl's RVI-SAC update, recorded for the conformance check
(tests/conformance/test_rvisac_original.py).

usage, from the repository root:
    uv run --project benchmarks/reference/envs/torch --locked python benchmarks/reference/conformance_rvisac.py CLONE OUT_DIR

CLONE is a checkout of yhisaki/average-reward-drl at COMMIT (the paper-era 07d0635 has the same
algorithm; the later commits reformat and move to gymnasium v5). The repository has no licence, so it
is not vendored: its records are committed instead (tests/conformance/records). Imports nothing from
Ajax. Their RVI_SAC is built with the hyperparameters of their rvi_sac.yaml (the published runs':
fq_update_tau 0.005, not the constructor's 0.01) at small widths, and makes the four calls of its
update_if_dataset_is_ready (update_critic, update_actor, update_reset_cost, update_target_networks)
on fixed random batches, a quarter of whose rows terminate, instead of on its replay's samples.
Recorder seams: their package's __init__ (which imports dm_control and wandb, never read by an
update) is skipped by loading the modules under an empty package; Normal.sample and Normal.rsample
draw loc + eps * scale with eps standard normal, kept in the float32 run and replayed in the float64
one (the next actions' draw, then the actor's). The updates in RECORDED run aside on a copy, in
float32 and then in float64 with the same draws. One .npz per recorded update: in.* (every network
and its target, the five Adams, fq and fq_reset), batch.*, draw.*, f32.* and f64.*; the first and
the last also hold their deterministic (evaluation) action on fixed states, act.state and
f32.act / f64.act.
"""

import copy
import importlib
import json
import pathlib
import platform
import sys
import types

import numpy as np
import torch

COMMIT = "b5b0d31bcd7e1778f11254490538e8662b01c2ed"
SEED, BATCH, OBS, ACTION = 0, 64, 3, 2
#: rvi_sac.yaml's settings, at widths small enough for the records to be committed.
SETTINGS = {
    "critic_hidden_dim": 16,
    "critic_reset_hidden_dim": 8,
    "actor_hidden_dim": 16,
    "fq_update_tau": 0.005,
    "target_reset_prob": 0.001,
    "fq_gain": 1.0,
    "fq_reset_gain": 1.0,
    "lr": 3e-4,
    "tau": 0.005,
    "batch_size": BATCH,
    "device": "cpu",
}
#: The updates recorded: the first two from the initialisation, and two after a warm-up long
#: enough for the Adams to be warm and the reset cost above zero.
RECORDED = (1, 2, 401, 402)
WITH_ACTION = (1, 402)


def load_rvi_sac(clone: pathlib.Path) -> type:
    package = types.ModuleType("average_reward_drl")
    package.__path__ = [str(clone / "average_reward_drl")]
    sys.modules["average_reward_drl"] = package
    return importlib.import_module("average_reward_drl.algorithms.rvi_sac").RVI_SAC


class Tape:
    """The standard-normal draws of Normal.sample and Normal.rsample: drawn and kept, or replayed."""

    def __init__(self):
        self.given: list = []
        self.drawn: list = []

    def eps(self, dist, sample_shape) -> torch.Tensor:
        shape = dist._extended_shape(torch.Size(sample_shape))
        if self.given:
            value = torch.as_tensor(self.given.pop(0), dtype=dist.loc.dtype)
        else:
            value = torch.randn(shape, dtype=dist.loc.dtype)
        self.drawn.append(value.detach().numpy().copy())
        return value

    def rsample(self, dist, sample_shape=()):
        return dist.loc + self.eps(dist, sample_shape) * dist.scale

    def sample(self, dist, sample_shape=()):
        with torch.no_grad():
            return self.rsample(dist, sample_shape)


TAPE = Tape()


def networks(agent) -> dict:
    """Each network and the optimizer stepping it (None for a target)."""
    return {
        "actor": (agent.actor, agent.policy_optimizer),
        "critic": (agent.critic, agent.critic_optimizer),
        "critic_target": (agent.critic_target, None),
        "critic_reset": (agent.critic_reset, agent.critic_reset_optimizer),
        "critic_reset_target": (agent.critic_reset_target, None),
        "temperature": (agent.temperature, agent.temperature_optimizer),
        "reset_cost": (agent.reset_cost, agent.reset_cost_optimizer),
    }


def state(agent) -> dict:
    out = {}
    for prefix, (net, optimizer) in networks(agent).items():
        for name, parameter in net.named_parameters():
            out[f"{prefix}.{name}"] = parameter.detach().numpy().copy()
            if optimizer is None:
                continue
            moments = optimizer.state.get(parameter, {})
            for moment in ("exp_avg", "exp_avg_sq"):
                value = moments[moment] if moments else torch.zeros_like(parameter)
                out[f"opt.{prefix}.{name}.{moment}"] = value.detach().numpy().copy()
            step = moments["step"] if moments else torch.zeros(())
            out[f"opt.{prefix}.{name}.step"] = step.detach().numpy().copy()
    for name in ("fq", "fq_reset"):
        out[name] = np.asarray(float(getattr(agent, name)), np.float64)
    return out


def update(agent, batch) -> None:
    """update_if_dataset_is_ready's calls, in its order, on batch."""
    agent.update_critic(batch)
    agent.update_actor(batch)
    agent.update_reset_cost(batch)
    agent.update_target_networks()


def as_batch(arrays: dict, dtype) -> types.SimpleNamespace:
    """Their Batch's fields (it casts to float32; this keeps dtype for the float64 run)."""
    fields = {k: torch.as_tensor(v, dtype=dtype) for k, v in arrays.items()}
    fields["terminated"] = torch.as_tensor(arrays["terminated"], dtype=torch.bool)
    fields["truncated"] = torch.as_tensor(arrays["truncated"], dtype=torch.bool)
    return types.SimpleNamespace(**fields)


def aside(agent, arrays: dict, draws: list, double: bool) -> tuple[dict, list]:
    """One update on a copy, in float64 when double, draws replayed when given."""
    a = copy.deepcopy(agent)
    dtype = torch.float64 if double else torch.float32
    if double:
        for net, optimizer in networks(a).values():
            net.double()
            for moments in [] if optimizer is None else optimizer.state.values():
                for moment in ("exp_avg", "exp_avg_sq"):
                    moments[moment] = moments[moment].double()
        for name in ("fq", "fq_reset"):
            setattr(a, name, torch.tensor(float(getattr(a, name)), dtype=dtype))
    TAPE.given, TAPE.drawn = list(draws), []
    update(a, as_batch(arrays, dtype))
    assert not TAPE.given, "a replayed draw was not used"
    return state(a), TAPE.drawn


def deterministic_action(agent, states: np.ndarray, double: bool) -> np.ndarray:
    """Their act in evaluation mode, on a batch of states: the actor's deterministic head."""
    a = copy.deepcopy(agent)
    dtype = torch.float64 if double else torch.float32
    a.actor.to(dtype)
    a.actor_gaussian_head.deterministic = True
    with torch.no_grad():
        return a.actor(torch.as_tensor(states, dtype=dtype)).numpy().copy()


def make_batch(rng: np.random.Generator) -> dict:
    """A quarter of the rows terminate (their s' is the reset state, here a draw like any other),
    and one in sixteen of the others is truncated, which the update never reads."""
    rows = np.arange(BATCH)
    return {
        "state": rng.normal(size=(BATCH, OBS)),
        "action": rng.uniform(-1.0, 1.0, (BATCH, ACTION)),
        "reward": rng.normal(size=BATCH),
        "next_state": rng.normal(size=(BATCH, OBS)),
        "terminated": (rows % 4 == 3).astype(np.float64),
        "truncated": (rows % 16 == 5).astype(np.float64),
    }


def record(clone: pathlib.Path, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rvi_sac = load_rvi_sac(clone)
    normal = torch.distributions.Normal
    normal.rsample = lambda dist, sample_shape=(): TAPE.rsample(dist, sample_shape)
    normal.sample = lambda dist, sample_shape=(): TAPE.sample(dist, sample_shape)
    agent = rvi_sac(OBS, ACTION, **SETTINGS)
    out.mkdir(parents=True, exist_ok=True)
    states = rng.normal(size=(BATCH, OBS))
    for step in range(1, max(RECORDED) + 1):
        arrays = make_batch(rng)
        if step in RECORDED:
            f32, draws = aside(agent, arrays, [], double=False)
            f64, _ = aside(agent, arrays, draws, double=True)
            record_arrays = {f"in.{k}": v for k, v in state(agent).items()}
            record_arrays.update({f"batch.{k}": v for k, v in arrays.items()})
            record_arrays.update(
                {"draw.next_action": draws[0], "draw.action": draws[1]}
            )
            record_arrays.update({f"f32.{k}": v for k, v in f32.items()})
            record_arrays.update({f"f64.{k}": v for k, v in f64.items()})
            if step in WITH_ACTION:
                record_arrays["act.state"] = states
                record_arrays["f32.act"] = deterministic_action(agent, states, False)
                record_arrays["f64.act"] = deterministic_action(agent, states, True)
            np.savez_compressed(out / f"rvisac_{step}.npz", **record_arrays)
        TAPE.drawn = []
        update(agent, as_batch(arrays, torch.float32))
    assert float(agent.reset_cost().detach()) > 0.0, (
        "the reset cost never left zero: warm up longer"
    )
    versions = {"torch": torch.__version__, "numpy": np.__version__}
    meta = versions | {
        "python": platform.python_version(),
        "yhisaki_average_reward_drl": COMMIT,
    }
    (out / "versions.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    record(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
