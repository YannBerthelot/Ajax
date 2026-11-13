"""Recurrent PPO's networks against its reference code's, PureJaxRL's recurrent PPO (vendored at
31756b1, benchmarks/reference/vendor/purejaxrl), parameter for parameter.

ppo_rnn.py's ActorCriticRNN (lines 46-79) reads the observation through one Dense(128) and ReLU and
one GRU of 128 units (ScannedRNN, line 35), which both heads read, and gives each head a Dense(128)
and ReLU before its output layer (lines 61-64, 71-74). With share_encoder=True and the same widths,
Ajax's recurrent PPO must hold the same parameters: the encoder and the GRU once, the critic's
(share_encoder), and each head's layers on its own side. A topology difference (a layer one side
lacks, an encoder kept twice) leaves a parameter without its counterpart.

PureJaxRL's parameters are read off its source, with shapes on CartPole (4 features, 2 actions):
the file is not run (its GRUCell API predates flax 0.12, see the vendor README). A one-update
comparison, as test_ppo2_original.py makes with ppo2, needs these to match first. They differ on
each head's hidden layer, which Ajax lacks: the check is expected to fail (strict) until it lands,
then the mark goes and the one-update comparison follows.
"""

from typing import Any

import jax
import pytest

from ajax import PPO
from ajax.networks.memory import MemoryConfig

GRU = {
    "ir": {"kernel": (128, 128), "bias": (128,)},
    "iz": {"kernel": (128, 128), "bias": (128,)},
    "in": {"kernel": (128, 128), "bias": (128,)},
    "hr": {"kernel": (128, 128)},
    "hz": {"kernel": (128, 128)},
    "hn": {"kernel": (128, 128), "bias": (128,)},
}
#: ActorCriticRNN's modules (flax names, in the order ppo_rnn.py builds them): the shapes of their
#: parameters, and Ajax's module under share_encoder (network, path below params), None for none.
REFERENCE: dict[str, tuple[dict, tuple[str, str] | None]] = {
    # :53-55, the embedding before the GRU
    "Dense_0": (
        {"kernel": (4, 128), "bias": (128,)},
        ("critic_state", "encoder/network/layers_0"),
    ),
    # :35, :59, the GRU both heads read
    "ScannedRNN_0/GRUCell_0": (GRU, ("critic_state", "memory_cell/core/gru_0")),
    # :61-64, the actor's hidden layer
    "Dense_1": ({"kernel": (128, 128), "bias": (128,)}, None),
    # :65-67, the logits
    "Dense_2": (
        {"kernel": (128, 2), "bias": (2,)},
        ("actor_state", "model/layers_0"),
    ),
    # :71-74, the critic's hidden layer
    "Dense_3": ({"kernel": (128, 128), "bias": (128,)}, None),
    # :75-77, the value
    "Dense_4": ({"kernel": (128, 1), "bias": (1,)}, ("critic_state", "model")),
}
#: ppo_rnn.py's config (lines 343-359), in Ajax's names, and the networks' widths.
SETTINGS: dict[str, Any] = {
    "n_envs": 4,
    "n_steps": 128,
    "num_minibatches": 4,
    "n_epochs": 4,
    "memory": MemoryConfig("gru", 128),
    "actor_architecture": ("128", "relu"),
    "critic_architecture": ("128", "relu"),
}


def reference_leaves() -> dict[str, tuple]:
    """PureJaxRL's parameters by Ajax's name (network/path), shapes; unmatched ones by their own."""
    out = {}
    for module, (params, ajax_module) in REFERENCE.items():
        for leaf, shape in _leaves(params):
            if ajax_module is None:
                out[f"purejaxrl/{module}/{leaf}"] = shape
            else:
                network, path = ajax_module
                out[f"{network}/{path}/{leaf}"] = shape
    return out


def _leaves(tree: dict, prefix: str = "") -> list:
    out = []
    for key, value in tree.items():
        if isinstance(value, dict):
            out += _leaves(value, f"{prefix}{key}/")
        else:
            out.append((prefix + key, value))
    return out


def ajax_leaves(share_encoder: bool) -> dict[str, tuple]:
    """Ajax's trainable parameters by network/path, the critic's ensemble axis left out."""
    agent = PPO("CartPole-v1", share_encoder=share_encoder, **SETTINGS)
    _, context = agent.make_context(4 * agent.n_steps * agent.env_args.n_envs)
    state = agent.init(jax.random.PRNGKey(0), context)
    out = {}
    for network in ("actor_state", "critic_state"):
        params = getattr(state, network).params["params"]
        for path, x in jax.tree_util.tree_leaves_with_path(params):
            keys = [k.key for k in path]
            shape = x.shape
            if keys[0] == "ensemble":
                keys, shape = keys[1:], shape[1:]
            out[f"{network}/{'/'.join(keys)}"] = tuple(shape)
    return out


def differences(ours: dict, reference: dict) -> list[str]:
    missing = sorted(set(reference) - set(ours))
    extra = sorted(set(ours) - set(reference))
    shapes = sorted(k for k in set(ours) & set(reference) if ours[k] != reference[k])
    return (
        [f"PureJaxRL's {k} {reference[k]}: none in Ajax" for k in missing]
        + [f"Ajax's {k} {ours[k]}: none in PureJaxRL" for k in extra]
        + [f"{k}: Ajax {ours[k]}, PureJaxRL {reference[k]}" for k in shapes]
    )


@pytest.mark.xfail(
    strict=True,
    reason="Ajax's heads lack PureJaxRL's hidden layer per head after the GRU (Dense_1,"
    " Dense_3; owner, 2026-11-13: expected to fail until they land)",
)
def test_shared_recurrent_networks_hold_purejaxrls_parameters():
    found = differences(ajax_leaves(share_encoder=True), reference_leaves())
    assert not found, "\n".join(found)


@pytest.mark.parametrize("share_encoder", [False])
def test_networks_kept_apart_fail_the_check(share_encoder):
    """A planted topology fault: each network with its own encoder and GRU."""
    found = differences(ajax_leaves(share_encoder), reference_leaves())
    assert any("none in PureJaxRL" in line for line in found), found
