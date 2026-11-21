"""``agent.save(state, path)`` and ``load``: an agent and its state from one file.

Contract (docs/guide/checkpoints.md, ``ajax.checkpoint.load``):
  - ``load`` returns ``(agent, state)``: an agent of the saved class built
    with the saved plain arguments, and the saved state, which acts as the
    saved one did;
  - ``ajax.load(path)`` reads the class from the file, ``PPO.load(path)``
    refuses another class's file;
  - an argument that is an object (extensions, a prebuilt environment) is
    not stored: ``load`` refuses, naming it, until it is passed again;
  - an argument given to ``load`` replaces the recorded one;
  - the file is still a checkpoint (``load_checkpoint`` reads it), and a
    file ``save_checkpoint`` wrote without an agent is refused by ``load``.
"""

import gymnax
import jax
import numpy as np
import pytest

import ajax
from ajax import PPO, SAC
from ajax.checkpoint import agent_record, load, load_checkpoint, save_checkpoint
from ajax.extensions.instrumentation import ConditioningMetrics
from ajax.networks.memory import MemoryConfig
from tests.probing import readouts


def tiny_ppo(**arguments) -> PPO:
    return PPO("CartPole-v1", n_envs=2, n_steps=8, n_epochs=1, **arguments)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A PPO trained for two seeds and saved: (agent, state, path)."""
    agent = tiny_ppo(gamma=0.9)
    state, _ = agent.train(seed=[0, 1], n_timesteps=32, logging_config=None)
    path = str(tmp_path_factory.mktemp("save") / "ppo.pkl")
    agent.save(state, path)
    return agent, jax.device_get(state), path


@pytest.mark.parametrize("loader", [ajax.load, PPO.load], ids=["ajax.load", "PPO.load"])
def test_a_saved_agent_and_its_state_load_from_the_file(saved, loader):
    agent, state, path = saved
    loaded_agent, loaded_state = loader(path)
    assert type(loaded_agent) is PPO
    assert (loaded_agent.n_envs, loaded_agent.n_steps, loaded_agent.gamma) == (
        2,
        8,
        0.9,
    )
    assert not readouts.differing_leaves(state, loaded_state, rtol=0, atol=0)
    obs, _ = gymnax.make("CartPole-v1")[0].reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(
        loaded_agent.predict(loaded_state, obs), agent.predict(state, obs)
    )


def test_an_argument_given_to_load_replaces_the_recorded_one(saved):
    _, _, path = saved
    loaded_agent, _ = PPO.load(path, vf_coef=0.25)
    assert loaded_agent.vf_coef == 0.25
    assert loaded_agent.gamma == 0.9


def test_another_class_s_file_is_refused(saved):
    with pytest.raises(ValueError, match="holds a PPO, not a SAC"):
        SAC.load(saved[2])


def test_an_object_argument_is_passed_again(tmp_path):
    """Extensions are objects: the file names them, load refuses without
    them and builds the agent with those passed."""
    agent = tiny_ppo(extensions=(ConditioningMetrics(),))
    state, _ = agent.train(seed=0, n_timesteps=16, logging_config=None)
    path = str(tmp_path / "ppo.pkl")
    agent.save(state, path)
    with pytest.raises(ValueError, match="extensions"):
        ajax.load(path)
    loaded_agent, _ = ajax.load(path, extensions=(ConditioningMetrics(),))
    assert loaded_agent.extensions == (ConditioningMetrics(),)


def test_the_record_keeps_plain_arguments_and_names_the_objects():
    environment = gymnax.make("Pendulum-v1")[0]
    record = agent_record(SAC(environment, memory=MemoryConfig("gru", 8)))
    assert record["class"] == "SAC"
    assert "env_id" in record["to_pass_again"]
    assert "env_id" not in record["arguments"]
    # A memory configuration is stored as the dict every agent also takes.
    assert record["arguments"]["memory"]["kind"] == "gru"
    assert record["arguments"]["memory"]["hidden_size"] == 8
    assert record["arguments"]["gamma"] == 0.99


def test_a_saved_file_is_still_a_checkpoint_and_a_bare_checkpoint_is_no_agent(
    saved, tmp_path
):
    agent, state, path = saved
    restored = load_checkpoint(tiny_ppo(gamma=0.9), path)
    assert not readouts.differing_leaves(state, restored, rtol=0, atol=0)
    bare = str(tmp_path / "bare.pkl")
    save_checkpoint(state, bare)
    with pytest.raises(ValueError, match="records no agent"):
        load(bare)
