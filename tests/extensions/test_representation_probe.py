"""RepresentationProbe on real tiny runs.

Contract (the extension's docstring): off unless listed; on, it leaves
training as it was, and writes one file per seed at step 0 and at the first
update on or past each multiple of ``probe_interval``, holding what that
seed's networks compute on the probe set, and logs the latest metrics at each
evaluation. Each expected value comes from outside the extension: the steps
worked out from the loop's documented cadence, activations recomputed with
NumPy from the saved kernels, values from the agent's own ``predict_value``,
metrics from other formulas of the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, SAC
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.environments.obs_norm import apply_obs_norm, normalized_obs
from ajax.extensions.capabilities import MissingCapability, NotWired
from ajax.extensions.observe import ExpertActionInput
from ajax.extensions.representation import RepresentationProbe
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import action_value_input, predict_value
from ajax.representation.probe_set import ProbeSet, make_probe_set
from ajax.representation.run_data import load_replay_observations
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.test_agent_contract import CASES

ENV = "Pendulum-v1"
NET = ("8", "relu", "8", "relu")
TINY: dict[str, Any] = {
    "actor_architecture": NET,
    "critic_architecture": NET,
    "n_envs": 2,
}
OFF_POLICY: dict[str, Any] = {
    **TINY,
    "learning_starts": 8,
    "batch_size": 8,
    "buffer_size": 128,
    "policy_update_start": 0,
}
ON_POLICY: dict[str, Any] = {**TINY, "n_steps": 8, "batch_size": 8, "n_epochs": 1}
ACTOR = "actor/encoder/network/layers_"
CRITIC = "critic/ensemble/encoder/network/layers_"
N_STATES = 32


def probe(directory: Path, interval: int = 16, **kwargs) -> RepresentationProbe:
    return RepresentationProbe(directory, interval, n_states=N_STATES, **kwargs)


def probes_of(directory: Path, seed: int = 0) -> list:
    return [np.load(path) for path in sorted(directory.glob(f"seed{seed:03d}_*.npz"))]


def probe_set(directory: Path) -> ProbeSet:
    return ProbeSet.load(directory / "probe_set.npz")


def seed_of(state, seed: int = 0):
    return jax.tree.map(lambda leaf: leaf[seed], state)


def rectified_layers(probe_file, network: str, x: np.ndarray, member=None) -> list:
    """The two hidden ReLU layers of a saved network on ``x``, with NumPy."""
    prefix = f"parameters/{network}/params/"
    prefix += "ensemble/" if member is not None else ""
    outputs = []
    for index in (0, 2):
        dense = f"{prefix}encoder/network/layers_{index}/"
        kernel, bias = probe_file[dense + "kernel"], probe_file[dense + "bias"]
        if member is not None:
            kernel, bias = kernel[member], bias[member]
        x = np.maximum(x @ kernel + bias, 0.0)
        outputs.append(x)
    return outputs


@pytest.fixture(scope="module")
def sac_run(tmp_path_factory):
    """Two seeds of SAC; 8 updates of probe 16 reach step 64 (see cadence)."""
    directory = tmp_path_factory.mktemp("sac")
    agent = SAC(ENV, extensions=(probe(directory),), **OFF_POLICY)
    logging = LoggingConfig(config={}, log_frequency=16, backends=())
    state, evaluations = agent.train(
        seed=[0, 1], n_timesteps=64, logging_config=logging
    )
    return directory, state, evaluations


def test_off_policy_probes_come_at_step_zero_then_each_multiple(sac_run):
    """Catches a missing untrained probe, or a gate firing at every update:
    the two envs step 2 at a time, and updates start at 8, so 16 to 64."""
    directory, _, _ = sac_run

    steps = [int(found["step"]) for found in probes_of(directory)]

    assert steps == [0, 16, 32, 48, 64]


def test_a_rollout_landing_past_a_multiple_still_probes(tmp_path):
    """Catches a gate asking for an exact multiple, which rollouts of 16 steps
    never land on: they reach 16, 32, 48, 64, and only 32 and 48 pass a new
    multiple of 24."""
    agent = PPO(ENV, extensions=(probe(tmp_path, 24),), **ON_POLICY)

    agent.train(seed=[0], n_timesteps=64)

    assert [int(found["step"]) for found in probes_of(tmp_path)] == [0, 32, 48]


def test_each_seed_gets_its_own_files_with_its_own_networks(sac_run):
    """Catches a wrong seed index: seeds overwriting each other's files, or
    one seed's network written under another's name."""
    directory, state, _ = sac_run

    for seed in (0, 1):
        found = probes_of(directory, seed)
        assert len(found) == 5
        last = found[-1]
        kernels = seed_of(state, seed).actor_state.params["params"]["encoder"]
        saved = last["parameters/actor/params/encoder/network/layers_0/kernel"]
        np.testing.assert_array_equal(saved, kernels["network"]["layers_0"]["kernel"])


def test_seeds_share_the_probe_set(sac_run):
    """Catches a seed's states of its own: the same index must be the same
    state across seeds, or nothing compares between them."""
    directory, _, _ = sac_run

    assert len(list(directory.glob("probe_set*"))) == 1
    assert probe_set(directory).observations.shape[0] == N_STATES


def test_files_hold_what_the_saved_parameters_compute(sac_run):
    """Catches activations of another layer, member or input: each saved
    layer is recomputed with NumPy from the saved kernels, on the observations
    for the actor and the observations with the stored actions for the critic."""
    directory, _, _ = sac_run
    last = probes_of(directory, 1)[-1]
    states = probe_set(directory)
    critic_input = np.concatenate([states.observations, states.actions], axis=-1)

    actor = rectified_layers(last, "actor", states.observations)
    critics = [rectified_layers(last, "critic", critic_input, m) for m in (0, 1)]

    assert last[f"layers/{ACTOR}1"].shape == (N_STATES, 8)
    assert last[f"layers/{CRITIC}3"].shape == (2, N_STATES, 8)
    np.testing.assert_allclose(last[f"layers/{ACTOR}1"], actor[0], atol=1e-4)
    np.testing.assert_allclose(last[f"layers/{ACTOR}3"], actor[1], atol=1e-4)
    for member, layers in enumerate(critics):
        saved = last[f"layers/{CRITIC}3"][member]
        np.testing.assert_allclose(saved, layers[1], atol=1e-4)


def effective_rank_by_definition(activations: np.ndarray) -> float:
    singular_values = np.linalg.svd(activations, compute_uv=False)
    shares = singular_values[singular_values > 0] / singular_values.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))


def cka_in_sample_space(first: np.ndarray, second: np.ndarray) -> float:
    """HSIC form of linear CKA (Kornblith et al. 2019): kernels over states,
    where the code works over units."""
    n = len(first)
    centring = np.eye(n) - 1.0 / n
    first_kernel = centring @ first @ first.T @ centring
    second_kernel = centring @ second @ second.T @ centring
    return float(
        np.sum(first_kernel * second_kernel)
        / (np.linalg.norm(first_kernel) * np.linalg.norm(second_kernel))
    )


def test_metrics_measure_the_saved_activations(sac_run):
    """Catches metrics of a stale or other layer, or an ensemble's members
    pooled into one matrix: each is recomputed from the file."""
    directory, _, _ = sac_run
    last = probes_of(directory)[-1]
    actor = f"metrics/Representation/{ACTOR}1/"
    critic = f"metrics/Representation/{CRITIC}3/"

    members = last[f"layers/{CRITIC}3"]
    by_member = [effective_rank_by_definition(member) for member in members]

    assert last[actor + "effective_rank"] == pytest.approx(
        effective_rank_by_definition(last[f"layers/{ACTOR}1"]), rel=1e-3
    )
    assert last[critic + "effective_rank"] == pytest.approx(
        np.mean(by_member), rel=1e-3
    )


def test_cka_compares_a_probe_with_the_one_before(sac_run):
    """Catches a carried layer never updated, which compares every probe with
    the first, or a first probe compared with zeros reading as a number. The
    critic's last layer moves enough for the two comparisons to differ."""
    directory, _, _ = sac_run
    found = probes_of(directory)
    name = f"Representation/{CRITIC}3/cka_previous"

    def mean_over_members(first: int, second: int) -> float:
        members = zip(
            found[first][f"layers/{CRITIC}3"],
            found[second][f"layers/{CRITIC}3"],
            strict=True,
        )
        return float(np.mean([cka_in_sample_space(a, b) for a, b in members]))

    expected = mean_over_members(3, 4)

    assert np.isnan(found[0][f"metrics/{name}"])
    assert abs(expected - mean_over_members(0, 4)) > 1e-3
    assert found[4][f"metrics/{name}"] == pytest.approx(expected, abs=1e-4)


def test_the_evaluations_log_the_latest_probes_metrics(sac_run):
    """Catches metrics stuck on the disk or a stale row: evaluations every 16
    steps meet the probes at 16, 32, 48, 64."""
    directory, _, evaluations = sac_run
    name = f"Representation/{CRITIC}3/activation_norm"

    logged = evaluations[name]

    for seed in (0, 1):
        saved = [found[f"metrics/{name}"] for found in probes_of(directory, seed)]
        np.testing.assert_array_equal(logged[seed], saved[1:])


@pytest.mark.parametrize(
    "cls, kwargs",
    [(SAC, OFF_POLICY), (PPO, ON_POLICY)],
    ids=["SAC", "PPO"],
)
def test_probing_leaves_training_as_it_was(tmp_path, cls, kwargs):
    """Catches a probe drawing on the agent's keys or touching its state. The
    compiled programs differ, so the bits may: rounding, 1e-5 relative over
    these few updates of float32 (CLAUDE.md: bitwise is not the goal)."""
    plain, _ = cls(ENV, **kwargs).train(seed=[0], n_timesteps=64)
    probed, _ = cls(ENV, extensions=(probe(tmp_path),), **kwargs).train(
        seed=[0], n_timesteps=64
    )

    for network in ("actor_state", "critic_state"):
        chex.assert_trees_all_close(
            getattr(plain, network).params, getattr(probed, network).params, rtol=1e-5
        )
    assert int(plain.collector_state.timestep[0]) == 64


@pytest.mark.parametrize(
    "cls, kwargs",
    [(SAC, OFF_POLICY), (PPO, ON_POLICY)],
    ids=["SAC", "PPO"],
)
def test_a_normalising_agents_probe_reads_what_its_networks_read(tmp_path, cls, kwargs):
    """Catches raw observations fed to a network trained on normalised ones:
    the saved values are the agent's own predict_value on its input, and the
    actor's layer what its saved kernels compute on the agent's normalised
    observations."""
    kwargs = {**kwargs, "normalize_observations": True}
    agent = cls(ENV, extensions=(probe(tmp_path),), **kwargs)

    state, _ = agent.train(seed=[0], n_timesteps=64)

    state = seed_of(state)
    critic = state.critic_state
    states = probe_set(tmp_path)
    observations = jnp.asarray(states.observations)
    if cls is SAC:
        x = action_value_input(observations, jnp.asarray(states.actions))
        actor_input = apply_obs_norm(observations, state.actor_state.obs_norm_info)
    else:
        x = normalized_obs(observations, state.collector_state.obs_norm_info)
        actor_input = x
    expected = predict_value(critic, critic.params, x).mean(0).squeeze(-1)
    last = probes_of(tmp_path)[-1]
    assert int(last["step"]) == 64
    np.testing.assert_allclose(last["values"], expected, atol=1e-4)
    first_layer = rectified_layers(last, "actor", np.asarray(actor_input))[0]
    np.testing.assert_allclose(last[f"layers/{ACTOR}1"], first_layer, atol=1e-4)


def test_a_resumed_run_goes_on_from_its_last_probe(tmp_path):
    """Catches a resumed run restarting its numbers over the first files, or
    its gate writing at every update: its counters are per seed, so the cond
    is a select, and the updates after step 48 would leave a next file."""
    agent = SAC(ENV, extensions=(probe(tmp_path),), **OFF_POLICY)
    state, _ = agent.train(seed=[0, 1], n_timesteps=32)

    agent.train(seed=[0, 1], n_timesteps=24, initial_state=state)

    for seed in (0, 1):
        steps = [int(found["step"]) for found in probes_of(tmp_path, seed)]
        assert steps == [0, 16, 32, 48]


def test_layers_keep_the_named_ones(tmp_path):
    """Catches a selection ignored, which fills the disk with every layer."""
    extension = probe(tmp_path, layers=(f"{ACTOR}3",))
    agent = SAC(ENV, extensions=(extension,), **OFF_POLICY)

    agent.train(seed=[0], n_timesteps=16)

    stored = {k for k in probes_of(tmp_path)[0].files if k.startswith("layers/")}
    assert stored == {f"layers/{ACTOR}3"}


def test_a_layer_matching_nothing_is_refused(tmp_path):
    """Catches a typo that records nothing for a whole run."""
    extension = probe(tmp_path, layers=("no/such/layer",))
    agent = SAC(ENV, extensions=(extension,), **OFF_POLICY)

    with pytest.raises(ValueError, match="no layer"):
        agent.train(seed=[0], n_timesteps=16)


def test_float16_halves_activations_alone(tmp_path):
    """Catches rounded parameters, which would swamp the small steps a weight
    trajectory is made of."""
    agent = SAC(ENV, extensions=(probe(tmp_path, float16=True),), **OFF_POLICY)

    agent.train(seed=[0], n_timesteps=16)

    found = probes_of(tmp_path)[0]
    assert found[f"layers/{ACTOR}1"].dtype == np.float16
    assert (
        found["parameters/actor/params/encoder/network/layers_0/kernel"].dtype
        == np.float32
    )


def test_a_given_probe_set_is_used_as_it_is(tmp_path):
    """Catches a run ignoring the states it was given: comparing two
    directories' runs needs one set."""
    env_args = SAC(ENV, **OFF_POLICY).env_args
    given = make_probe_set(env_args, 16, seed=9)
    agent = SAC(ENV, extensions=(probe(tmp_path, probe_set=given),), **OFF_POLICY)

    agent.train(seed=[0], n_timesteps=16)

    assert probes_of(tmp_path)[0][f"layers/{ACTOR}1"].shape == (16, 8)


def test_a_probe_keeps_observations_the_agent_trains_on(tmp_path):
    """Catches a sample from an empty or wrong source: every kept row is a row
    of this seed's replay, none at step 0, whose replay holds nothing yet."""
    extension = probe(tmp_path, replay_states=4)
    agent = SAC(ENV, extensions=(extension,), **OFF_POLICY)

    state, _ = agent.train(seed=[0], n_timesteps=64)

    found = probes_of(tmp_path)
    assert "replay/observations" not in found[0].files
    kept = load_replay_observations(tmp_path)
    assert kept.shape == (4, 4, 3)
    replay = seed_of(state).collector_state.buffer_state.experience
    stored = np.asarray(replay["obs"]).reshape(-1, 3)
    for row in kept.reshape(-1, 3):
        assert (np.abs(stored - row).max(axis=1) < 1e-6).any()


def test_replay_states_on_an_agent_without_a_replay_is_refused(tmp_path):
    """Catches a run recording nothing it asked for: PPO keeps no replay."""
    agent = PPO(ENV, extensions=(probe(tmp_path, replay_states=4),), **ON_POLICY)

    with pytest.raises(ValueError, match="replay"):
        agent.train(seed=[0], n_timesteps=16)


TRACES: list[int] = []


@dataclass(frozen=True)
class TraceCounting(RepresentationProbe):
    def read_networks(self, *args):
        TRACES.append(1)
        return super().read_networks(*args)


def test_the_number_of_probes_does_not_change_the_compilation(tmp_path):
    """Catches a probe leaving the compiled program (work done in the callback,
    a retrace per probe): 7 probes must trace as often as 2."""

    def traces(interval: int) -> int:
        TRACES.clear()
        extension = TraceCounting(tmp_path / str(interval), interval, n_states=8)
        SAC(ENV, extensions=(extension,), **OFF_POLICY).train(seed=[0], n_timesteps=64)
        return len(TRACES)

    assert traces(64) == traces(10)


def test_a_world_model_is_refused(tmp_path):
    """Catches a silent run reading nothing: its policy reads a latent."""
    with pytest.raises(MissingCapability, match="obs_networks"):
        DreamerV3(ENV, extensions=(probe(tmp_path),), **DREAMER_TINY)


def test_observation_features_are_refused(tmp_path):
    """Catches a shape error deep in the trace: the env alone cannot give the
    columns a feature appends."""

    def expert(obs):
        return jnp.zeros((*obs.shape[:-1], 1))

    agent = SAC(
        ENV,
        expert_policy=expert,
        extensions=(ExpertActionInput(), probe(tmp_path)),
        **OFF_POLICY,
    )

    with pytest.raises(ValueError, match="come from the env alone"):
        agent.train(seed=[0], n_timesteps=16)


def test_a_non_positive_interval_is_refused(tmp_path):
    """Catches a zero interval dividing by zero inside the trace."""
    with pytest.raises(ValueError, match="probe_interval"):
        RepresentationProbe(tmp_path, 0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_agent_reading_observations_can_be_probed(tmp_path, name):
    """Catches an agent's networks or state the probe cannot read. A new agent
    joins through CASES; one without obs_networks must be refused instead, as
    UDRL is until a probe set can hold its command."""
    case = CASES[name]
    extension = probe(tmp_path, 8)
    kwargs = dict(case.kwargs)
    try:
        agent = case.cls(case.envs[-1], extensions=(extension,), **kwargs)
    except MissingCapability as refusal:
        assert "obs_networks" in str(refusal)
        return
    except NotWired:
        assert name == "UDRL"
        return

    agent.train(seed=[0], n_timesteps=case.n_timesteps)

    found = probes_of(tmp_path)
    assert len(found) >= 1
    assert any(k.startswith("layers/actor/") for k in found[0].files)
