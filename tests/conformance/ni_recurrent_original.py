"""Ajax's side of the recurrent SAC and TD3 check against Ni et al.'s pomdp-baselines
(test_ni_recurrent_original.py): one update from each recorded state
(benchmarks/reference/conformance_ni_recurrent.py), written in the record's names.

The state maps one to one, kernels transposed, but for the GRU's biases. Ni et al.'s actor and
critic (policies/models/recurrent_actor.py, recurrent_critic.py) map onto Ajax's: the embedders of
the previous action, the previous reward and the observation onto the input embedding's parts, each
RNN onto its memory cell, the shortcut embedders onto the shortcuts, the policy's Mlp onto the
actor's layers after the memory, qf1 and qf2 onto the critic ensemble's members; then the targets,
both Adams, and SAC's log alpha with its Adam.

torch's GRU stacks its gates (r, z, n) in weight_ih_l0 and weight_hh_l0 and gives each gate two
biases, b_ih and b_hh. flax's GRUCell has a kernel per gate and input, one bias for each of r and
z (ir, iz), and b_in and b_hn (in, hn) for n, where r multiplies b_hn. Ajax's r and z biases start
as Ni et al.'s sums b_ih + b_hh, and their Adam moments as b_ih's: one gradient reaches both of
torch's pair, so their moments agree (to rounding, the recorder's arrays show it). Each of the pair
then takes the step Ajax's one bias takes, so the pair's sum moves twice as far (GATE_BIAS). The
actor's step reads the stepped critic, so the difference reaches every actor leaf in the same
update, by about 1e-3 of TD3's actor step and 2e-5 of SAC's here (float64: the critic's leaves
and the temperature match to 1e-13). The recorder therefore also records the update, from the
same state, with each GRU given flax's one bias per gate (ONE_GATE_BIAS: b_hh's r and z rows
held); Ajax's update is checked leaf by leaf against that one, and against Ni et al.'s own on
the leaves upstream of the gate biases' step, the gate biases predicted (gate_bias_reports).

The windows are injected where Ajax samples its replay (recurrent.get_sequence_batch_from_buffer),
as the replay stores them: each row's observation with the action and reward before it, the
window's first row too, whose previous action and reward Ajax's sampler zeroes as Ni et al.'s
update does (a[0] = r[0] = 0, policy_rnn.py:247-257), and the row after the window's last, which
only flags the episode starts past it. The update's draws are taped from Ni et al.'s (NI_DRAWS).
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
from typing import Any, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.recurrent as recurrent
import ajax.agents.SAC.SAC as sac_module
from ajax.networks.memory import MemoryConfig
from tests.conformance import kit, submaps

ENV_ID = "Pendulum-v1"  # the recorder's: 3 observations, 1 action
WIDTH = 128  # rnn_hidden_size
GATE_ROWS = {"r": 0, "z": 1, "n": 2}  # torch's order of the gates in its stacked arrays
#: rnn.yml's policy and replay settings, in Ajax's names. Most are Ajax's recurrent defaults;
#: given so that a changed default fails here as a setting, not as a leaf.
COMMON: dict[str, Any] = {
    "memory": MemoryConfig("gru"),
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "sequence_length": 64,
    "burn_in": 0,
    "learning_starts": 1_000,
    "buffer_size": 10_000,  # never sampled: the windows are injected
}
SETTINGS = {
    "sac": dict(COMMON, alpha_learning_rate=3e-4, target_entropy_per_dim=-1.0),
    "td3": dict(COMMON, target_policy_noise=0.2, target_noise_clip=0.5),
}
AGENTS = {"sac": ajax.SAC, "td3": ajax.TD3}
#: Which of Ni et al.'s draws (the record's noise, in their order) each of Ajax's standard-normal
#: draws reads, by the function that draws it. Ni et al. sample at every observation of the
#: window, o[0] to o[T], and keep o[1:] for the next actions (q_target[1:], sac.py:127) and o[:-1]
#: for the actor's loss (policy_loss[:-1], sac.py:177); Ajax samples the rows' next observations
#: and the rows themselves. SAC: the next actions', then the actor's (sac.py:104-108, :158-160);
#: TD3: the target noise (td3.py:57).
NI_DRAWS = {
    "sac": {
        "sample_next_actions": (0, slice(1, None)),
        "compute_soft_policy_objective": (1, slice(None, -1)),
    },
    "td3": {"compute_td3_td_target": (0, slice(1, None))},
}
#: Ajax's layers in the record's modules. The input embedding's parts:
EMBEDDINGS = {
    "previous_action": "action_embedder.fc",
    "previous_reward": "reward_embedder.fc",
    "observation": "observ_embedder.fc",
}
SHORTCUTS = {
    "actor": "current_observ_embedder.fc",
    "critic": "current_shortcut_embedder.fc",
}
#: The actor's layers after the memory: SAC's Actor calls them head, mean and log_std, TD3's
#: deterministic actor hidden and head.
ACTOR_LAYERS = {
    ("head", "network", "layers_0"): "policy.fc0",
    ("head", "network", "layers_2"): "policy.fc1",
    ("mean",): "policy.last_fc",
    ("log_std",): "policy.last_fc_log_std",
    ("hidden", "layers_0"): "policy.fc0",
    ("hidden", "layers_2"): "policy.fc1",
    ("head",): "policy.last_fc",
}
MEMBER_LAYERS = {
    ("head", "network", "layers_0"): "fc0",
    ("head", "network", "layers_2"): "fc1",
    ("model",): "last_fc",
}
LEAF = re.compile(
    r"\.(actor_state|critic_state|alpha)\.(params|target_params|step|opt_state\[0\]\.(mu|nu|count))(.*)"
)
#: The names of the r and z gates' biases, torch's pair summed (the module docstring).
GATE_BIAS = re.compile(r".*rnn\.bias\[[rz]\]")
#: The record's update with torch's GRUs given flax's one bias per r and z gate.
ONE_GATE_BIAS = "one_gate_bias."


@dataclasses.dataclass(frozen=True)
class Source:
    """Where an Ajax leaf lies in a record: the sum of these arrays, their rows ``rows`` when
    set, transposed for a kernel; ``count``: an Adam's step count."""

    keys: tuple[str, ...]
    rows: Optional[tuple[int, int]] = None
    kernel: bool = False
    count: bool = False

    def read(self, arrays: dict, tag: str) -> np.ndarray:
        """The value under ``tag`` ("in.", "f32." or "f64.")."""
        if self.count:
            return np.asarray(
                int(np.asarray(arrays[tag + self.keys[0]]).reshape(-1)[0])
            )
        value = np.zeros(())
        for key in self.keys:
            part = np.asarray(arrays[tag + key], np.float64)
            if self.rows is not None:
                part = part[self.rows[0] : self.rows[1]]
            value = value + part
        return value.T if self.kernel else value


def sources(path: Any) -> Optional[list[tuple[str, Source]]]:
    """An Ajax leaf's names and sources in the record (two for an ensemble leaf); None for the
    leaves Ni et al.'s update has no counterpart of."""
    match = LEAF.fullmatch(jax.tree_util.keystr(path))
    if match is None:
        return None
    owner, kind, moment, rest = match.groups()
    keys = re.findall(r"\['([^']+)'\]", rest)
    tail = {None: "", "mu": ".exp_avg", "nu": ".exp_avg_sq", "count": ""}[moment]
    net = {"actor_state": "actor", "critic_state": "critic", "alpha": "alpha"}[owner]
    if kind == "step" or moment == "count":
        # The train state's step and its Adam's count: both torch's step, which every parameter
        # of one optimiser shares.
        first = "opt.alpha" if net == "alpha" else f"opt.{net}.rnn.weight_ih_l0"
        name = f"opt.{net}.step" if moment else f"{net}.step"
        return [(name, Source((f"{first}.step",), count=True))]
    if net == "alpha":
        name = "opt.alpha" + tail if moment else "log_alpha"
        return [(name, Source((name,)))]
    if moment is not None:
        prefix = f"opt.{net}."
    else:
        prefix = f"{net}_target." if kind == "target_params" else f"{net}."
    *modules, leaf = keys[1:]  # past "params"
    if modules[0] == "memory_cell":
        return [gru_source(prefix, modules[-1], leaf, tail)]
    if modules[0] == "encoder":
        layers = [EMBEDDINGS[modules[1]]]
    elif modules == ["shortcut"]:
        layers = [SHORTCUTS[net]]
    elif modules[0] == "ensemble":
        layer = MEMBER_LAYERS[tuple(modules[1:])]
        layers = [f"qf{member}.{layer}" for member in (1, 2)]
    else:
        layers = [ACTOR_LAYERS[tuple(modules)]]
    torch_leaf = "weight" if leaf == "kernel" else "bias"
    names = [f"{prefix}{layer}.{torch_leaf}{tail}" for layer in layers]
    return [(n, Source((n,), kernel=leaf == "kernel")) for n in names]


def gru_source(prefix: str, gate: str, leaf: str, tail: str) -> tuple[str, Source]:
    """A flax GRUCell leaf's source: ``gate`` its dense layer, ir to hn (input or hidden, then the
    gate), as rows of torch's stacked arrays."""
    side, row = gate[0] + "h", GATE_ROWS[gate[1]]
    rows = (row * WIDTH, (row + 1) * WIDTH)
    if leaf == "kernel":
        key = f"{prefix}rnn.weight_{side}_l0{tail}"
        return f"{key}[{gate[1]}]", Source((key,), rows, kernel=True)
    if gate[1] == "n" or tail:  # one bias each side, or the r and z moments: b_ih's
        key = f"{prefix}rnn.bias_{side}_l0{tail}"
        return f"{key}[{gate[1]}]", Source((key,), rows)
    pair = (f"{prefix}rnn.bias_ih_l0", f"{prefix}rnn.bias_hh_l0")
    return f"{prefix}rnn.bias[{gate[1]}]", Source(pair, rows)


def to_ajax(template: Any, arrays: dict, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        found = sources(path)
        if found is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        values = [source.read(arrays, "in.") for _, source in found]
        if found[0][1].count:
            return jnp.asarray(values[0], x.dtype)
        value = np.stack(values) if len(values) > 1 else values[0]
        return jnp.asarray(value.reshape(x.shape), dtype)

    return kit.map_with_path(put, template)


def to_reference(state: Any) -> dict:
    out = {}
    for path, x in kit.leaves_with_path(state):
        found = sources(path)
        if found is None:
            continue
        value = np.asarray(x) if found[0][1].count else np.asarray(x, np.float64)
        values = list(value) if len(found) > 1 else [value]
        for (name, _), v in zip(found, values, strict=True):
            out[name] = v
    return out


def window(arrays: dict, parts: Any) -> dict:
    """The record's windows as Ajax's replay holds them, time-major: T + 1 rows, each
    observation joined with the action and reward before it in the columns ``parts`` (the
    networks' input embedding) gives them."""

    def joined(obs: np.ndarray, action: np.ndarray, reward: np.ndarray) -> np.ndarray:
        columns = {
            "observation": obs,
            "previous_action": action,
            "previous_reward": reward,
        }
        width = max(stop for _, ranges, _ in parts for _, stop in ranges)
        out = np.zeros(obs.shape[:-1] + (width,), np.float64)
        for name, ranges, _ in parts:
            ((start, stop),) = ranges
            out[..., start:stop] = columns[name]
        return out

    def batch(name: str) -> np.ndarray:
        return np.asarray(arrays[f"batch.{name}"], np.float64)

    action, reward, terminated = batch("act"), batch("rew"), batch("term")
    obs, next_obs = batch("obs"), batch("obs2")
    past = np.zeros_like(action[:1])  # the row past the window: never trained on
    return {
        "obs": joined(
            np.concatenate([obs, next_obs[-1:]]),
            np.concatenate([batch("prev_act")[None], action]),
            np.concatenate([batch("prev_rew")[None], reward]),
        ),
        "next_obs": joined(
            np.concatenate([next_obs, np.zeros_like(next_obs[:1])]),
            np.concatenate([action, past]),
            np.concatenate([reward, past]),
        ),
        "action": np.concatenate([action, past]),
        "reward": np.concatenate([reward, past]),
        "terminated": np.concatenate([terminated, past]),
        "truncated": np.zeros_like(np.concatenate([terminated, past])),
    }


@dataclasses.dataclass(frozen=True)
class Run:
    """Ajax's state after one update from each record, by precision then record name, in the
    record's names; and each name's source in the records."""

    states: dict
    sources: dict


@contextlib.contextmanager
def planted(algo: str) -> Iterator[dict]:
    """The settings of a planted fault, with the code it changes changed while it runs: SAC's
    temperature stepping log alpha (feedforward SAC's loss) where Ni et al.'s steps alpha; TD3's
    actor stepping every second update (sfujim/TD3's delay) where Ni et al.'s steps every one."""
    if algo == "td3":
        yield {"policy_delay": 2}
        return
    alpha_loss = sac_module.compute_alpha_temperature_loss
    sac_module.compute_alpha_temperature_loss = sac_module.compute_temperature_loss
    try:
        yield {}
    finally:
        sac_module.compute_alpha_temperature_loss = alpha_loss


def run(
    records: Any,
    algo: str,
    precisions: tuple[str, ...] = ("f32", "f64"),
    fault: bool = False,
) -> Run:
    """Ajax's update from each of ``algo``'s records in each precision; with ``fault``, the
    planted one."""
    faulty: contextlib.AbstractContextManager[dict] = (
        planted(algo) if fault else contextlib.nullcontext({})
    )
    with faulty as fault_settings:
        agent = AGENTS[algo](ENV_ID, **dict(SETTINGS[algo], **fault_settings))
        step = kit.capture_step(agent, 2_000)
        template, update = step.fresh(jax.random.PRNGKey(0)), step.update
        assert update is not None, "an off-policy agent passes its update to its loop"
        given: dict = {}  # the windows, set while the update is traced (its arguments)
        sample = recurrent.get_sequence_batch_from_buffer

        def injected(buffer: Any, buffer_state: Any, key: Any) -> dict:
            sampled = sample(buffer, buffer_state, key)
            replaced = {
                f: jnp.asarray(v, sampled[f].dtype).reshape(sampled[f].shape)
                for f, v in given.items()
            }
            return dict(sampled, **replaced)

        recurrent.get_sequence_batch_from_buffer = injected
        try:
            return _updates(records, algo, precisions, template, update, given)
        finally:
            recurrent.get_sequence_batch_from_buffer = sample


def _updates(
    records: Any,
    algo: str,
    precisions: tuple[str, ...],
    template: Any,
    update: Any,
    given: dict,
) -> Run:
    parts = template.critic_state.apply_fn.__self__.input_embedding
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{algo}_*.npz"))}

    def from_window(state: Any, windows: dict) -> Any:
        given.clear()
        given.update(windows)
        return update(state, None)[0]

    # The noises' keys, record by record, in float32 (keys do not depend on the precision; the
    # tape logs on a callback thread, outside jax.enable_x64). The replay's own draws (its
    # windows' starts) are left to draw: the injected windows replace what they pick.
    draws: dict = {}
    for name, a in arrays.items():
        windows = {f: np.asarray(v, np.float32) for f, v in window(a, parts).items()}
        with kit.recording() as recorded:
            state = to_ajax(template, a, np.float32)
            jax.block_until_ready(jax.jit(from_window)(state, windows))
        draws[name] = [
            (role, key) for role, key, _ in recorded.log if role[0] == "normal"
        ]
        # Every noise that yields values is one of Ni et al.'s, once (shape-only traces log
        # nothing).
        functions = sorted(role[1].rpartition(":")[2] for role, _ in draws[name])
        assert functions == sorted(NI_DRAWS[algo]), functions
    table: dict = {}

    def with_noise(state: Any, windows: dict, tape: dict) -> Any:
        table.update(tape)
        return from_window(state, windows)

    states: dict = {}
    for precision in precisions:
        dtype = np.float64 if precision == "f64" else np.float32
        states[precision] = {}
        with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
            replayed = jax.jit(with_noise)
            for name, a in arrays.items():
                tape: dict = {}
                for role, key in draws[name]:
                    index, rows = NI_DRAWS[algo][role[1].rpartition(":")[2]]
                    value = np.asarray(a["noise"][index][rows], np.float32)
                    tape[role] = (key[None], value.reshape((1, *role[2])))
                windows = {f: np.asarray(v, dtype) for f, v in window(a, parts).items()}
                state = to_ajax(template, a, dtype)
                states[precision][name] = to_reference(replayed(state, windows, tape))
        assert not replay.misses
    catalogue = {}
    for path, _ in kit.leaves_with_path(template):
        for name, source in sources(path) or []:
            catalogue[name] = source
    return Run(states, catalogue)


def reports(
    records: Any,
    ours: Run,
    algo: str,
    update: Run,
    leaves: Any = lambda name: True,
    original: str = ONE_GATE_BIAS,
) -> list[submaps.Report]:
    """Per record of ``algo``, ``update``'s state against Ni et al.'s float32 one on the leaves
    ``leaves`` keeps, on the change from ``in.``, within 10 times the two sides' float32 rounding
    (``ours``: Ajax's float32 and float64 runs, submaps.rounding_null). ``original``: the update
    recorded with one bias per gate (ONE_GATE_BIAS), or "" for Ni et al.'s own."""
    out = []
    names = [name for name in ours.sources if leaves(name)]
    for path in sorted(records.glob(f"{algo}_*.npz")):
        arrays = dict(np.load(path))
        reference = {
            tag: {n: ours.sources[n].read(arrays, f"{prefix}{tag}.") for n in names}
            for tag, prefix in (("in", ""), ("f32", original), ("f64", original))
        }
        a32, a64, new = (
            {n: run.states[p][path.stem][n] for n in names}
            for run, p in ((ours, "f32"), (ours, "f64"), (update, "f32"))
        )
        nulls = submaps.rounding_null(
            (reference["f32"], reference["f64"]), (a32, a64), old=reference["in"]
        )
        compared = submaps.compare(
            new, reference["f32"], nulls, old=reference["in"], context=path.stem
        )
        out.append(compared)
    assert out, f"no record for {algo}"
    return out


def gate_bias_reports(records: Any, ours: Run, algo: str) -> list[submaps.Report]:
    """Per record of ``algo``, the critic's r and z gate biases (torch's pairs, summed) after Ni
    et al.'s own update against what Ajax's one bias predicts for them: twice Ajax's step, each
    of the pair taking it; the target's, Ajax's target step plus tau times Ajax's step, the online
    bias it averages having moved that much more. The actor's step reads the stepped critic, so
    its own gate biases are past the difference's reach of a prediction."""
    paths = sorted(records.glob(f"{algo}_*.npz"))
    tau = SETTINGS[algo]["tau"]
    return [_gate_bias_report(dict(np.load(p)), p.stem, ours, tau) for p in paths]


def _gate_bias_report(
    arrays: dict, record: str, ours: Run, tau: float
) -> submaps.Report:
    def ajaxs(name: str, precision: str) -> np.ndarray:
        old = ours.sources[name].read(arrays, "in.")
        return ours.states[precision][record][name] - old

    def torchs(name: str, precision: str) -> np.ndarray:
        source = ours.sources[name]
        return source.read(arrays, f"{precision}.") - source.read(arrays, "in.")

    predicted: dict = {"f32": {}, "f64": {}}
    measured: dict = {"f32": {}, "f64": {}}
    for name in ours.sources:
        if GATE_BIAS.fullmatch(name) is None or not name.startswith("critic."):
            continue
        target = name.replace("critic.", "critic_target.", 1)
        for precision in ("f32", "f64"):
            step = ajaxs(name, precision)
            predicted[precision][name] = 2 * step
            predicted[precision][target] = ajaxs(target, precision) + tau * step
            measured[precision][name] = torchs(name, precision)
            measured[precision][target] = torchs(target, precision)
    nulls = submaps.rounding_null(
        (measured["f32"], measured["f64"]), (predicted["f32"], predicted["f64"])
    )
    return submaps.compare(predicted["f32"], measured["f32"], nulls, context=record)
