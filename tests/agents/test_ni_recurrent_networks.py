"""The off-policy agents' networks with memory are Ni et al. (2022)'s.

Contract (owner, 2026-11-13: SAC, TD3, REDQ and ASAC with memory= match
twni2016/pomdp-baselines @e7c19c3; TQC and RVI-SAC by analogy), read from
that code, not from Ajax's:

- The actor's memory reads the previous action, the previous reward and the
  observation, each through its own Linear and ReLU of 16, 16 and 32 units
  (configs/pomdp/*/rnn.yml:33-35; policies/models/recurrent_actor.py:36-46,
  :101-109), a GRU of 128 units (rnn.yml:36); a shortcut Linear and ReLU of
  32 units reads the observation (:76-80); the 256-256 policy MLP reads
  both (:82-87, rnn.yml:39); the mean and log-std layers read its output.
- The critic is one module: the same embedders and a GRU of 128 units,
  then a shortcut Linear and ReLU of 64 units (the GRU's input width) over
  the observation and the action asked about (recurrent_critic.py:70-76);
  two 256-256 Q-networks read both (:94-99), one RNN for both.
- Initialisation: every RNN weight a block of an orthogonal matrix over the
  stacked gates, biases 0 (recurrent_actor.py:67-74); embedders torch's
  nn.Linear, uniform in +-1/sqrt(fan_in) (utils/helpers.py:217-218); MLP
  hidden layers rlkit's fanin_init, uniform in +-1/sqrt(fan_out), biases 0.1
  (torchkit/networks.py:53-57, pytorch_utils.py:39-48); output layers
  uniform in +-3e-3 (Q) or +-1e-3 (actor), biases too (networks.py:66-68,
  actor.py:22, :118-121).
- A run with memory reads PreviousAction and PreviousReward (added where
  absent), at Ni et al.'s width and initialisation unless the config sets
  them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import ASAC, REDQ, SAC, TD3
from ajax.extensions.observe import (
    PreviousAction,
    PreviousReward,
    TrainingProgressInput,
)
from ajax.networks.memory import MemoryConfig
from ajax.networks.utils import RLKIT_FANIN, dense

OBS, ACTION = 3, 1  # Pendulum
EMBEDDED = 16 + 16 + 32


def _states(cls: Any = SAC, **kwargs: Any) -> Any:
    agent = cls("Pendulum-v1", n_envs=2, memory=MemoryConfig("gru"), **kwargs)
    state, _ = agent.train(seed=0, n_timesteps=0)
    return agent, jax.tree.map(lambda leaf: leaf[0], state)  # the one seed


def _shapes(params: Any) -> dict[str, tuple]:
    leaves = jax.tree_util.tree_leaves_with_path(params["params"])
    return {jax.tree_util.keystr(path): leaf.shape for path, leaf in leaves}


def _kernel(params: Any, *names: str) -> tuple[np.ndarray, np.ndarray]:
    node = params["params"]
    for name in names:
        node = node[name]
    return np.asarray(node["kernel"]), np.asarray(node["bias"])


def test_the_actor_is_ni_et_als():
    _, state = _states()
    shapes = _shapes(state.actor_state.params)
    expected = {
        "['encoder']['previous_action']['kernel']": (ACTION, 16),
        "['encoder']['previous_reward']['kernel']": (1, 16),
        "['encoder']['observation']['kernel']": (OBS, 32),
        "['memory_cell']['core']['gru_0']['ir']['kernel']": (EMBEDDED, 128),
        "['memory_cell']['core']['gru_0']['hr']['kernel']": (128, 128),
        "['shortcut']['kernel']": (OBS, 32),
        "['head']['network']['layers_0']['kernel']": (128 + 32, 256),
        "['head']['network']['layers_2']['kernel']": (256, 256),
        "['mean']['kernel']": (256, ACTION),
        "['log_std']['kernel']": (256, ACTION),
    }
    for path, shape in expected.items():
        assert shapes[path] == shape, path


def test_the_two_q_networks_read_one_memory():
    _, state = _states()
    shapes = _shapes(state.critic_state.params)
    # One embedder, GRU and shortcut, outside the two members' ensemble.
    assert shapes["['encoder']['observation']['kernel']"] == (OBS, 32)
    assert shapes["['memory_cell']['core']['gru_0']['iz']['kernel']"] == (EMBEDDED, 128)
    assert shapes["['shortcut']['kernel']"] == (OBS + ACTION, EMBEDDED)
    member = "['ensemble']['head']['network']['layers_0']['kernel']"
    assert shapes[member] == (2, 128 + EMBEDDED, 256)
    assert shapes["['ensemble']['model']['kernel']"] == (2, 256, 1)
    # One carry: the memory's, (members of one, envs, units).
    assert jax.tree.leaves(state.critic_state.hidden_state)[0].shape == (1, 2, 128)


def test_td3s_actor_is_ni_et_als_deterministic_one():
    _, state = _states(TD3)
    shapes = _shapes(state.actor_state.params)
    assert shapes["['hidden']['layers_0']['kernel']"] == (128 + 32, 256)
    assert shapes["['head']['kernel']"] == (256, ACTION)
    head, bias = _kernel(state.actor_state.params, "head")
    assert np.abs(head).max() <= 1e-3 and np.abs(bias).max() <= 1e-3
    assert np.abs(bias).max() > 0  # drawn, not zero


@pytest.mark.parametrize("cls", [SAC, REDQ, ASAC])
def test_layers_start_as_ni_et_als(cls):
    _, state = _states(cls)
    actor, critic = state.actor_state.params, state.critic_state.params
    # Embedders: torch's nn.Linear, kernel and bias in +-1/sqrt(fan_in).
    kernel, bias = _kernel(actor, "encoder", "observation")
    bound = 1 / np.sqrt(OBS)
    assert np.abs(kernel).max() <= bound and np.abs(bias).max() <= bound
    assert np.abs(kernel).max() > 0.5 * bound and np.abs(bias).max() > 0
    # rlkit's hidden layers: +-1/sqrt(fan_out), biases 0.1.
    kernel, bias = _kernel(actor, "head", "network", "layers_0")
    assert np.abs(kernel).max() <= 1 / np.sqrt(256)
    assert np.abs(kernel).max() > 0.9 / np.sqrt(256)
    np.testing.assert_array_equal(bias, np.float32(0.1))
    # Output layers: actor +-1e-3, Q +-3e-3, kernel and bias.
    for name in ("mean", "log_std"):
        kernel, bias = _kernel(actor, name)
        assert max(np.abs(kernel).max(), np.abs(bias).max()) <= 1e-3
        assert np.abs(bias).max() > 0
    kernel, bias = _kernel(critic, "ensemble", "model")
    assert max(np.abs(kernel).max(), np.abs(bias).max()) <= 3e-3
    assert np.abs(kernel).max() > 1e-3


def test_the_gru_starts_as_torchs_orthogonal_over_the_stacked_gates():
    """Each gate's (64, 128) block of an orthogonal (64, 384) matrix has
    entries of std 1/sqrt(384); the recurrent (128, 128) blocks of an
    orthogonal (128, 384) one too. Biases are zero."""
    _, state = _states()
    gru = state.actor_state.params["params"]["memory_cell"]["core"]["gru_0"]
    for gate in ("ir", "iz", "in", "hr", "hz", "hn"):
        std = float(np.asarray(gru[gate]["kernel"]).std())
        np.testing.assert_allclose(std, 1 / np.sqrt(3 * 128), rtol=0.1)
    for gate in ("ir", "iz", "in", "hn"):
        np.testing.assert_array_equal(gru[gate]["bias"], 0.0)


def test_the_observation_part_is_every_column_but_the_actions_and_rewards():
    """A training-progress column is the observation's (normalised, so before
    the raw previous action and reward), embedded with the env's."""
    _, state = _states(extensions=(TrainingProgressInput(),))
    shapes = _shapes(state.actor_state.params)
    assert shapes["['encoder']['observation']['kernel']"] == (OBS + 1, 32)
    assert shapes["['shortcut']['kernel']"] == (OBS + 1, 32)


def test_a_run_with_memory_reads_ni_et_als_inputs_at_their_width():
    agent, _ = _states(extensions=(PreviousAction(),))
    kinds = [type(extension) for extension in agent.extensions]
    assert kinds.count(PreviousAction) == 1 and kinds.count(PreviousReward) == 1
    assert agent.memory.hidden_size == 128
    assert agent.memory.init == "stacked_orthogonal"


def test_a_set_width_and_init_are_kept():
    memory = MemoryConfig("lstm", 32, init="flax")
    agent = SAC("Pendulum-v1", memory=memory)
    assert agent.memory == memory
    transformer = SAC("Pendulum-v1", memory=MemoryConfig("transformer"))
    assert transformer.memory.init == "gpt2"


def test_a_run_without_memory_is_unchanged():
    agent = SAC("Pendulum-v1")
    assert agent.extensions == ()
    assert agent.network_args.input_embedding == ()


def test_an_embedding_without_memory_is_refused():
    agent = SAC("Pendulum-v1", extensions=(PreviousAction(), PreviousReward()))
    agent.network_args = dataclasses.replace(
        agent.network_args, input_embedding=(("observation", 8),)
    )
    with pytest.raises(ValueError, match="needs memory"):
        agent.train(seed=0, n_timesteps=0)


def test_rlkits_fanin_reads_the_fan_out_and_sets_biases_to_one_tenth():
    """torch's (out, in) weight, size[0] = out: bound 1/sqrt(out)."""
    layer = dense(400, RLKIT_FANIN, None, jax.nn.initializers.zeros)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))["params"]
    kernel = np.asarray(params["kernel"])
    assert kernel.shape == (4, 400)
    assert 0.9 / np.sqrt(400) < np.abs(kernel).max() <= 1 / np.sqrt(400)
    np.testing.assert_array_equal(params["bias"], np.float32(0.1))
