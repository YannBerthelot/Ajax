"""Adversarial imitation as a composable :class:`Extension`.

* :class:`AdversarialReward` — GAIL's discriminator reward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ajax.buffers.dataset import TransitionDataset
from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.utils import TORCH_LINEAR, dense, parse_architecture


class Discriminator(nn.Module):
    """Logits of "an expert's" for an (observation, action) row."""

    architecture: tuple[str, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.Sequential(parse_architecture(self.architecture, TORCH_LINEAR))(x)
        # torch_linear draws the kernel and the bias; the default is unused.
        return dense(1, TORCH_LINEAR, None, nn.initializers.lecun_normal())(h)


@dataclass(frozen=True)
class AdversarialReward(Extension):
    """GAIL (Ho and Ermon, NeurIPS 2016): the policy learns from a reward a
    discriminator gives, trained to tell its rollouts from the expert's
    demonstrations. Following HumanCompatibleAI/imitation@e5ef188
    (algorithms/adversarial/gail.py and common.py): each rollout's rewards
    become ``-log(1 - D(s, a))`` from the current discriminator (``-
    logsigmoid(-logits)``), then the discriminator takes ``n_updates``
    steps of binary cross-entropy, the demonstrations labelled 1, on the
    rollout's rows and as many demonstration rows drawn uniformly. The
    environment's own reward is not used. The discriminator reads the action
    the env executes, clipped to ``[-1, 1]`` in agent units
    (:func:`~ajax.environments.utils.agent_action_to_env`), as the reference
    records it (its ``BufferingWrapper`` sees the action SB3 steps the env
    with, clipped to the box). Its 32-32 ReLU network starts as
    torch's ``nn.Linear`` and steps by torch's Adam (learning rate 1e-3),
    the reference's ``BasicRewardNet`` and optimiser. Deviations: the
    discriminator reads ``(s, a)`` unnormalised, without the reference
    examples' running normalisation; its generator batch is the whole rollout, the
    reference's ``demo_batch_size`` rows of its last one; PPO stands in for
    the paper's TRPO, as in the reference. Logs ``gail/reward``,
    ``gail/disc_loss`` and the shares of expert and rollout rows it
    classifies right, ``gail/expert_acc`` and ``gail/policy_acc``.
    """

    demonstrations: TransitionDataset
    n_updates: int = 2
    learning_rate: float = 1e-3
    architecture: tuple[str, ...] = ("32", "relu", "32", "relu")
    name: str = "gail"
    requires: ClassVar[frozenset[str]] = frozenset({"continuous_actions"})
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "reward": frozenset({"actions", "raw_observations"})
    }

    def rows(self) -> jax.Array:
        """The demonstrations' (observation, executed action) rows."""
        data = self.demonstrations
        executed = np.clip(data.action, -1.0, 1.0)
        return jnp.asarray(np.concatenate([data.obs, executed], -1))

    def init_state(self, agent_state: Any, rng: jax.Array) -> TrainState:
        del agent_state
        net = Discriminator(self.architecture)
        params = net.init(rng, self.rows()[:1])
        return TrainState.create(
            apply_fn=net.apply, params=params, tx=optax.adam(self.learning_rate)
        )

    def reward(
        self,
        agent_state: Any,
        ext_state: TrainState,
        batch: Any,
        rewards: jax.Array,
        context: ExtensionContext,
    ) -> tuple[jax.Array, TrainState, dict]:
        del agent_state
        # A Gaussian sample outside [-1, 1] runs as the bound: as the expert's.
        executed = jnp.clip(batch["actions"], -1.0, 1.0)
        x = jnp.concatenate([batch["raw_observations"], executed], -1)
        logits = ext_state.apply_fn(ext_state.params, x)
        new_rewards = jax.lax.stop_gradient(-jax.nn.log_sigmoid(-logits))
        policy = x.reshape(-1, x.shape[-1])
        demos = self.rows()

        def step(state: TrainState, key: jax.Array) -> tuple[TrainState, tuple]:
            expert = demos[jax.random.randint(key, (len(policy),), 0, len(demos))]

            def loss(params: Any) -> tuple[jax.Array, tuple]:
                le, lp = state.apply_fn(params, expert), state.apply_fn(params, policy)
                bce = -jax.nn.log_sigmoid(le).mean() - jax.nn.log_sigmoid(-lp).mean()
                return bce / 2, (jnp.mean(le > 0), jnp.mean(lp < 0))

            (value, accs), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), (value, *accs)

        keys = jax.random.split(context.rng, self.n_updates)
        ext_state, (losses, expert_acc, policy_acc) = jax.lax.scan(
            step, ext_state, keys
        )
        metrics = {
            "gail/reward": jnp.mean(new_rewards),
            "gail/disc_loss": losses[-1],
            "gail/expert_acc": expert_acc[-1],
            "gail/policy_acc": policy_acc[-1],
        }
        return new_rewards.reshape(rewards.shape), ext_state, metrics


__all__ = ["AdversarialReward"]
