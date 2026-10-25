"""ExpertWarmup: the agent's expert takes a share of the warm-up steps, a
controller phase (choose_warmup_action) with no state and no key in the
learner. SAC's expert_fraction builds it and warns, its None default
standing for 0.7 with an expert until B7."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO, SAC, TD3
from ajax.extensions.capabilities import MissingCapability, NotWired
from ajax.extensions.expert import ExpertGains, ExpertWarmup, ResidualPolicy
from ajax.extensions.pretrain import PriorData
from tests.agents.test_gain_mode import TwoGains
from tests.agents.test_offline import episodes

SMALL: dict[str, Any] = {"actor_architecture": ("16", "relu")}
SMALL.update({"critic_architecture": ("16", "relu"), "n_envs": 2})
SMALL.update({"learning_starts": 8, "batch_size": 8, "buffer_size": 64})
SMALL.update({"expert_buffer_n_steps": 0})


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def sac(**kwargs: Any) -> SAC:
    """SAC on Pendulum with push as its expert, its warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return SAC("Pendulum-v1", **{**SMALL, "expert_policy": push, **kwargs})


def train(agent: SAC) -> Any:
    """Six steps per env, four of them the warm-up: one seed's state."""
    state, _ = agent.train(seed=0, n_timesteps=12)
    return jax.tree.map(lambda leaf: leaf[0], state)


@pytest.mark.parametrize("fraction", [1.0, 0.0])
def test_the_expert_takes_its_share_of_the_warmup_and_is_flagged(fraction):
    """At 1 every warm-up row holds the expert's action (push: 0.25 in
    agent units) and is flagged; at 0 none is. No row after the warm-up is
    flagged."""
    state = train(sac(extensions=(ExpertWarmup(fraction=fraction),)))
    rows = state.collector_state.buffer_state.experience
    warmup = {key: np.asarray(value)[:, :4] for key, value in rows.items()}
    took = warmup["is_expert"][..., 0] == 1
    assert took.all() if fraction == 1.0 else not took.any()
    expert_rows = np.isclose(warmup["action"][..., 0], 0.25)
    np.testing.assert_array_equal(expert_rows, took)
    assert (np.asarray(rows["is_expert"])[:, 4:6] == 0).all()


def test_the_keyword_and_the_extension_train_the_same():
    """expert_fraction, and its None default with an expert, build the
    extension: bitwise the same runs."""
    pairs = [
        (sac(expert_fraction=0.5), sac(extensions=(ExpertWarmup(fraction=0.5),))),
        (sac(), sac(extensions=(ExpertWarmup(fraction=0.7),))),
    ]
    for keyword, extension in pairs:
        by_keyword = jax.tree.leaves(train(keyword))
        by_extension = jax.tree.leaves(train(extension))
        for old, new in zip(by_keyword, by_extension, strict=True):
            np.testing.assert_array_equal(old, new)


def test_the_keyword_warns_and_raises_beside_the_extension():
    with pytest.warns(
        FutureWarning, match="adds ExpertWarmup.fraction=0.7. by default"
    ):
        SAC("Pendulum-v1", **SMALL, expert_policy=push)
    with pytest.warns(FutureWarning, match="expert_fraction will be removed"):
        SAC("Pendulum-v1", **SMALL, expert_policy=push, expert_fraction=0.3)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", ".*(ExpertWarmup|expert_fraction)", FutureWarning
        )
        warmup = (ExpertWarmup(fraction=0.3),)
        agent = SAC("Pendulum-v1", **SMALL, expert_policy=push, extensions=warmup)
    assert agent.env_args.warmups == warmup and not agent.extension_stack
    both = {"expert_fraction": 0.3, "extensions": warmup}
    with pytest.raises(ValueError, match="not both"):
        SAC("Pendulum-v1", **SMALL, expert_policy=push, **both)


def test_sac_builds_no_default_warmup_where_it_never_mixed():
    """Without warm-up steps, under an action transform or without an
    expert there is no mix, and nothing is built."""
    assert sac(learning_starts=0).env_args.warmups == ()
    assert sac(extensions=(ResidualPolicy(),)).env_args.warmups == ()
    assert sac(expert_policy=None).env_args.warmups == ()


def test_sac_with_an_expert_and_no_warmup_steps_builds_and_trains():
    """The default mix needs warm-up steps; without them SAC builds none,
    also beside PriorData (whose rows need learning_starts=0). A replay with
    an expert's columns takes no dataset yet, so that one only builds."""
    train(sac(learning_starts=0))
    data = PriorData(episodes(3, discrete=False))
    assert sac(learning_starts=0, extensions=(data,)).env_args.warmups == ()


def test_the_warmup_is_refused_where_it_cannot_act():
    warmup = ExpertWarmup(fraction=0.5)
    with pytest.raises(MissingCapability, match="warmup_steps"):
        sac(learning_starts=0, extensions=(warmup,))
    with pytest.raises(MissingCapability, match="agent_expert"):
        sac(expert_policy=None, extensions=(warmup,))
    with pytest.raises(MissingCapability, match="warmup"):
        PPO("Pendulum-v1", extensions=(warmup,))
    # TD3 has a uniform warm-up; its collector does not run the phase yet.
    with pytest.raises(NotWired, match="choose_warmup_action"):
        TD3("Pendulum-v1", expert_policy=push, extensions=(warmup,))
    with pytest.raises(ValueError, match="one warm-up"):
        sac(extensions=(warmup, ExpertWarmup(fraction=0.2)))
    # SAC's gain mode acts through the shared pipeline, without a chooser yet.
    with pytest.raises(ValueError, match="expert_warmup"):
        sac(expert_policy=TwoGains(), extensions=(ExpertGains(), warmup))
