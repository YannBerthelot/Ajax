"""Compute precision: ``train(precision=...)`` and ``ajax.precision``.

Contract (the bf16 study's design, owner's decisions of 2026-11-20; not read
off the implementation):
* ``"auto"`` resolves to bfloat16 exactly when every device of the run has
  bfloat16 matrix units -- a TPU, or an NVIDIA GPU of compute capability 8.0
  or more -- and to float32 elsewhere (older GPUs, CPUs, devices reporting
  no capability). ``"float32"`` and ``"bfloat16"`` are taken as asked; any
  other value is refused before anything trains.
* In bfloat16 every network family keeps its float32 parameters (the same
  initial weights as in float32) and returns float32, within bfloat16
  accuracy of its float32 output, and not equal to it (the request reached
  its layers). Gradients reach the float32 parameters in float32.
* ``train`` records the resolved precision in the run config. DreamerV3
  stores its latents in the compute dtype, and a run resumed in another
  precision converts them.
"""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, DreamerV3
from ajax.agents.APG.networks import Controller
from ajax.agents.DreamerV3.networks import WorldModel
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.TDMPC2.networks import make_policy_prior, make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.networks.deterministic import DeterministicActor
from ajax.networks.networks import Actor, Encoder, MultiCritic
from ajax.networks.q_networks import QNetwork
from ajax.precision import resolve_precision


def device(platform, capability=None):
    return SimpleNamespace(
        platform=platform, compute_capability=capability, device_kind="fake"
    )


@pytest.mark.parametrize(
    "fake, expected",
    [
        (device("cuda", "6.1"), "float32"),  # GTX 1080 Ti (Pascal)
        (device("cuda", "7.5"), "float32"),  # RTX 2080 Ti (Turing)
        (device("cuda", "8.0"), "bfloat16"),  # A100: the boundary itself
        (device("cuda", "8.9"), "bfloat16"),  # L40S
        (device("cuda", "10.0"), "bfloat16"),  # "10.0" < "8.0" as text
        (device("gpu", "9.0"), "bfloat16"),
        (device("tpu"), "bfloat16"),
        (device("cpu"), "float32"),
        (device("cuda", None), "float32"),  # no capability reported
        (device("rocm", "gfx90a"), "float32"),  # unchecked vendors fall back
    ],
)
def test_auto_picks_bfloat16_only_on_hardware_with_bfloat16_units(fake, expected):
    assert resolve_precision("auto", [fake]) == expected


def test_auto_on_mixed_devices_follows_the_weakest():
    """Catches members of one run computing in different precisions."""
    assert resolve_precision(
        "auto", [device("cuda", "8.9"), device("cuda", "7.5")]
    ) == ("float32")


@pytest.mark.parametrize("asked", ["float32", "bfloat16"])
def test_an_explicit_precision_overrides_the_hardware(asked):
    assert resolve_precision(asked, [device("cuda", "6.1")]) == asked
    assert resolve_precision(asked, [device("cuda", "8.9")]) == asked


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError):
        resolve_precision("float16", [device("cuda", "8.9")])


ARCHITECTURE = ("64", "relu", "64", "relu")
OBS = 5


def _obs(key):
    return jax.random.normal(key, (8, OBS))


def _network_cases():
    """Each network family, built twice: (float32 module, bfloat16 module,
    inputs, how to read a float array from its output)."""
    tdmpc2 = TDMPC2Config(latent_dim=16, enc_dim=32, mlp_dim=32, num_q=2)
    dreamer = DreamerV3Config(units=16, hidden=12, deter=64, stoch=4, classes=3)
    return {
        "encoder": (
            lambda dtype: Encoder(input_architecture=ARCHITECTURE, dtype=dtype),
            lambda key: (_obs(key),),
            lambda out: out,
        ),
        "actor": (
            lambda dtype: Actor(
                input_architecture=ARCHITECTURE,
                action_dim=2,
                continuous=True,
                compute_dtype=dtype,
                mean_kernel_init="lecun_normal",
            ),
            lambda key: (_obs(key),),
            lambda out: out.mean(),
        ),
        "critics": (
            lambda dtype: MultiCritic(
                input_architecture=ARCHITECTURE, num=2, compute_dtype=dtype
            ),
            lambda key: (_obs(key),),
            lambda out: out,
        ),
        "deterministic actor": (
            lambda dtype: DeterministicActor(
                input_architecture=ARCHITECTURE,
                action_dim=2,
                compute_dtype=dtype,
                output_kernel_init="lecun_normal",
            ),
            lambda key: (_obs(key),),
            lambda out: out.mode(),
        ),
        "q network": (
            lambda dtype: QNetwork(
                input_architecture=ARCHITECTURE, n_actions=3, compute_dtype=dtype
            ),
            lambda key: (_obs(key),),
            lambda out: out.q_values,
        ),
        "pqn network": (
            lambda dtype: PQNNetwork(
                input_architecture=ARCHITECTURE, n_actions=3, compute_dtype=dtype
            ),
            lambda key: (_obs(key),),
            lambda out: out.q_values,
        ),
        "apg controller": (
            lambda dtype: Controller(
                input_architecture=ARCHITECTURE, action_dim=2, compute_dtype=dtype
            ),
            lambda key: (_obs(key),),
            lambda out: out.mode(),
        ),
        "tdmpc2 world model": (
            lambda dtype: make_world_model(
                tdmpc2.replace(compute_dtype=dtype or "float32")
            ),
            lambda key: (_obs(key), jax.random.uniform(key, (8, 2), minval=-1)),
            lambda out: out,
        ),
        "tdmpc2 policy prior": (
            lambda dtype: make_policy_prior(
                tdmpc2.replace(compute_dtype=dtype or "float32"), 2
            ),
            lambda key: (jax.random.normal(key, (8, 16)),),
            lambda out: out[0],
        ),
        "dreamerv3 world model": (
            lambda dtype: WorldModel(
                dataclasses.replace(dreamer, compute_dtype=dtype or "float32"), OBS
            ),
            lambda key: (_obs(key), jax.random.uniform(key, (8, 2), minval=-1)),
            lambda out: out[0],  # posterior logits, after a bf16 RSSM step
        ),
    }


CASES = _network_cases()


@pytest.mark.parametrize("case", sorted(CASES))
def test_bfloat16_networks_keep_float32_weights_and_return_float32(case):
    """Catches parameters stored or initialised in bfloat16, outputs left in
    bfloat16 (what follows a network -- distributions, losses -- would then
    run in it), a cast that corrupts values instead of rounding them, and a
    request that never reaches the layers."""
    build, make_inputs, read = CASES[case]
    key = jax.random.PRNGKey(0)
    inputs = make_inputs(key)
    full, mixed = build(None), build("bfloat16")
    variables = full.init(key, *inputs)
    mixed_variables = mixed.init(key, *inputs)
    assert jax.tree.structure(variables) == jax.tree.structure(mixed_variables)
    for ours, theirs in zip(
        jax.tree.leaves(variables), jax.tree.leaves(mixed_variables), strict=True
    ):
        assert theirs.dtype == ours.dtype == jnp.float32
        np.testing.assert_array_equal(ours, theirs)
    expected = read(full.apply(variables, *inputs))
    got = read(mixed.apply(variables, *inputs))
    assert got.dtype == jnp.float32
    # bfloat16 keeps 8 significant bits (~0.4% per rounding), a few layers
    # deep: 5% of the output's scale bounds it with room.
    scale = float(jnp.max(jnp.abs(expected))) + 1e-6
    np.testing.assert_allclose(got, expected, atol=0.05 * scale)
    assert not np.array_equal(np.asarray(got), np.asarray(expected))


def test_bfloat16_gradients_reach_float32_parameters():
    """Catches gradients taken in bfloat16 (the optimiser would update
    float32 weights from bfloat16 moments) or cut at the cast."""
    key = jax.random.PRNGKey(2)
    x = _obs(key)
    variables = Encoder(input_architecture=ARCHITECTURE).init(key, x)
    mixed = Encoder(input_architecture=ARCHITECTURE, dtype="bfloat16")
    gradients = jax.grad(lambda v: jnp.sum(mixed.apply(v, x) ** 2))(variables)
    for leaf in jax.tree.leaves(gradients):
        assert leaf.dtype == jnp.float32
        assert bool(jnp.any(leaf != 0))


def test_train_refuses_an_unknown_precision_before_training():
    agent = PPO("Pendulum-v1", n_envs=1)
    with pytest.raises(ValueError, match="precision"):
        agent.train(seed=0, n_timesteps=64, precision="float16")


def test_train_records_the_precision_it_computes_in():
    """Catches a run config that logs a request ("auto") instead of what the
    run computed in. On a CPU "auto" is float32."""
    agent = PPO("Pendulum-v1", n_envs=1, n_steps=32, batch_size=32, n_epochs=1)
    agent.train(seed=0, n_timesteps=32, precision="auto")
    assert agent.config["precision"] == "float32"


TINY_DREAMER = {"units": 16, "hidden": 12, "deter": 64, "stoch": 4, "classes": 3, "blocks": 8}


@pytest.fixture(scope="module")
def dreamer_bfloat16():
    agent = DreamerV3("Pendulum-v1", n_envs=2, train_ratio=0.0625, **TINY_DREAMER)
    state, _ = agent.train(seed=0, n_timesteps=400, precision="bfloat16")
    return agent, state


def test_dreamerv3_in_bfloat16_stores_bfloat16_latents_and_float32_weights(
    dreamer_bfloat16,
):
    """Catches latents stored in float32 (twice the replay for nothing) and
    weights or optimiser moments drifting to bfloat16 after an update."""
    agent, state = dreamer_bfloat16
    assert agent.config["precision"] == "bfloat16"
    assert state.replay_state.deter.dtype == jnp.bfloat16
    trained = (state.world_model_state, state.actor_state, state.critic_state)
    for leaf in jax.tree.leaves(trained):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_dreamerv3_resumed_in_float32_converts_its_latents(dreamer_bfloat16):
    agent, state = dreamer_bfloat16
    # A copy: the resumed program takes over the buffers it is given.
    resumed, _ = agent.train(
        seed=0,
        n_timesteps=100,
        initial_state=jax.tree.map(jnp.copy, state),
        precision="float32",
    )
    assert resumed.replay_state.deter.dtype == jnp.float32
