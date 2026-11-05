"""Recording how an agent's internal representations change during training.

Neuroscience embeds a population's activity into a low-dimensional manifold;
the ML analogue tracks a network's hidden activations on a fixed set of states
across training. :class:`RepresentationProbe` does the part that belongs in the
compiled loop, reading the networks at chosen steps; embedding what it wrote
takes minutes on a CPU, offline.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.traverse_util import flatten_dict

from ajax.agents.update_steps import run_if
from ajax.environments.obs_norm import apply_obs_norm, normalized_obs
from ajax.environments.utils import raw_obs_columns
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.instrumentation import _is_ensemble_critic, critic_takes_action
from ajax.log import gated_log_callback, unevaluated
from ajax.networks.networks import action_value_input, critic_input
from ajax.networks.utils import Activation
from ajax.representation.metrics import (
    dead_unit_fraction,
    effective_rank,
    linear_cka,
    mean_activation_norm,
)
from ajax.representation.probe_set import ProbeSet, probe_set_for
from ajax.representation.run_data import PROBE_FILE
from ajax.state import with_batch_stats
from ajax.types import Replaceable


def probe_metrics(layer: jax.Array, previous: jax.Array) -> dict[str, jax.Array]:
    """One network member's layer on the probe set, ``(n_states, n_units)``,
    against its activations at the previous probe."""
    return {
        "effective_rank": effective_rank(layer),
        "cka_previous": linear_cka(layer, previous),
        "activation_norm": mean_activation_norm(layer),
        "dead_fraction": dead_unit_fraction(layer),
    }


def layer_metrics(layer: jax.Array, previous: jax.Array) -> dict[str, jax.Array]:
    """:func:`probe_metrics`, averaged over an ensemble's members (its layers
    have a leading member axis). An ensemble's members are not a batch of one
    representation: each is compared with its own past."""
    if layer.ndim == 2:
        return probe_metrics(layer, previous)
    return jax.tree.map(jnp.mean, jax.vmap(probe_metrics)(layer, previous))


def named_metrics(layers: dict, previous: dict) -> dict[str, jax.Array]:
    """Every layer's metrics, by the name the logger shows them under."""
    named = {}
    for layer, activations in layers.items():
        measured = layer_metrics(
            activations, previous[layer].reshape(activations.shape)
        )
        named.update({f"Representation/{layer}/{k}": v for k, v in measured.items()})
    return named


def read_activations(train_state: Any, x: jax.Array) -> tuple[Any, dict]:
    """The network's output at ``x`` as it reads it, and each of its
    activation layers' outputs by path. A network without such a layer
    (a CNN encoder's activations are plain functions) reports none."""
    output, captured = train_state.apply_fn(
        with_batch_stats(train_state, train_state.params),
        x,
        capture_intermediates=lambda module, _: isinstance(module, Activation),
        mutable=["intermediates"],
    )
    return output, {
        "/".join(path[:-1]): outputs[0]
        for path, outputs in flatten_dict(captured["intermediates"]).items()
    }


@struct.dataclass
class ProbeState(Replaceable):
    """The probe set, carried rather than closed over: it stays out of the
    compiled program. A resumed run compares its first probe with the last."""

    observations: jax.Array
    actions: jax.Array
    seed_index: jax.Array
    #: The next probe's number and the step of the last one.
    probe_index: jax.Array
    last_step: jax.Array
    #: Per layer, its activations at the last probe, flat.
    previous: dict
    #: Per name, the last probe's metrics, which the evaluations log.
    metrics: dict


@dataclass(frozen=True)
class RepresentationProbe(Extension):
    """Reads the networks on a fixed probe set during training.

    At step 0 and then at the first update on or past each multiple of
    ``probe_interval`` env steps, it writes one file per seed to
    ``run_directory``, and logs at each evaluation the metrics of the latest
    probe as ``Representation/<layer>/<metric>``. The file ``seed000_probe00001.npz``
    holds ``step``, ``metrics/<name>``, ``parameters/<network>/...``,
    ``layers/<layer>`` (``(n_states, n_units)``, an ensemble's with a leading
    member axis) and ``values``, the critic's ensemble mean, when the agent
    has one.

    The probe set is the run directory's ``probe_set.npz``, made there from
    the seed ``probe_seed`` unless it exists: runs pointed at one directory
    read the same states. Give ``probe_set`` to bring your own.

    ``layers`` keeps the layers whose name (``actor/encoder/network/layers_1``)
    holds one of the strings; empty, every one. ``float16`` halves the
    activations' size, and turns values above 65504 infinite. Parameters stay
    float32: rounding them would swamp the small steps between two probes.

    A resumed run goes on from its last probe, but the state it resumes from
    has each seed's own counters, so the probe's work runs at every update.
    """

    run_directory: str | os.PathLike
    probe_interval: int
    n_states: int = 2048
    probe_seed: int = 0
    layers: tuple[str, ...] = ()
    float16: bool = False
    probe_set: Optional[ProbeSet] = field(default=None, compare=False, repr=False)
    env_args: Any = field(default=None, compare=False, repr=False)
    name: str = "representation_probe"
    requires: ClassVar[frozenset[str]] = frozenset({"obs_networks"})
    context: ClassVar[frozenset[str]] = frozenset({"env_args"})
    pretrain_context: ClassVar[frozenset[str]] = frozenset({"seed_index"})

    def __post_init__(self) -> None:
        if self.probe_interval <= 0:
            raise ValueError(
                f"probe_interval is in env steps: got {self.probe_interval}"
            )

    def bind_to_agent(self, **agent_context: Any) -> "RepresentationProbe":
        """Copies the agent's ``env_args`` and fetches the probe set.

        Raises ``ValueError`` on observation columns or actions the env's
        own random rollouts cannot give the networks."""
        env_args = agent_context["env_args"]
        if env_args.observers or env_args.policy_action_dim is not None:
            raise ValueError(
                "RepresentationProbe's states come from the env alone: it cannot"
                " read a network on observation features or a gain policy's"
                " actions."
            )
        probe_set = self.probe_set
        if probe_set is None:
            probe_set = probe_set_for(
                self.run_directory, env_args, self.n_states, self.probe_seed
            )
        return dataclasses.replace(self, env_args=env_args, probe_set=probe_set)

    def read_networks(
        self,
        agent_state: Any,
        observations: jax.Array,
        actions: jax.Array,
        rng: jax.Array,
    ) -> tuple[dict, Optional[jax.Array], dict]:
        """The chosen layers' activations, the critic's values and the
        parameters of each network read."""
        collector = agent_state.collector_state
        if self.env_args.store_normalized_obs:
            observations = normalized_obs(
                observations,
                collector.obs_norm_info,
                raw_obs_columns(self.env_args.env),
            )
        actor = agent_state.actor_state
        actor_input = apply_obs_norm(observations, actor.obs_norm_info)
        _, actor_layers = read_activations(actor, actor_input)
        layers = {f"actor/{k}": v for k, v in actor_layers.items()}
        parameters = {"actor": actor.params}
        values = None
        critic: Any = getattr(agent_state, "critic_state", None)
        if _is_ensemble_critic(critic):
            takes_action = critic_takes_action(critic, observations, actions, rng)
            x = (
                action_value_input(observations, actions)
                if takes_action
                else observations
            )
            output, critic_layers = read_activations(critic, critic_input(critic, x))
            layers.update({f"critic/{k}": v for k, v in critic_layers.items()})
            parameters["critic"] = critic.params
            values = output.mean(axis=0).squeeze(-1)
        kept = {
            name: activations
            for name, activations in layers.items()
            if not self.layers or any(part in name for part in self.layers)
        }
        if not kept:
            raise ValueError(
                f"RepresentationProbe found no layer among {sorted(layers)}"
                f" matching layers={self.layers}."
            )
        return kept, values, parameters

    def init_state(self, agent_state: Any, rng: jax.Array) -> ProbeState:
        assert self.probe_set is not None  # bind_to_agent fills it
        observations = jnp.asarray(self.probe_set.observations, jnp.float32)
        actions = jnp.asarray(self.probe_set.actions, jnp.float32)
        layers = jax.eval_shape(
            lambda: self.read_networks(agent_state, observations, actions, rng)[0]
        )
        metrics = jax.eval_shape(named_metrics, layers, layers)
        previous = jax.tree.map(
            lambda layer: jnp.zeros(layer.size, layer.dtype), layers
        )
        return ProbeState(
            observations=observations,
            actions=actions,
            seed_index=jnp.asarray(0),
            probe_index=jnp.asarray(0),
            last_step=jnp.asarray(0),
            previous=previous,
            metrics=unevaluated(metrics),
        )

    def measure(
        self, agent_state: Any, state: ProbeState, step: Any, rng: jax.Array, due: Any
    ) -> ProbeState:
        """Reads the networks, writes the file and keeps the metrics."""
        layers, values, parameters = self.read_networks(
            agent_state, state.observations, state.actions, rng
        )
        metrics = named_metrics(layers, state.previous)
        record = {
            "step": step,
            "probe_index": state.probe_index,
            "layers": layers,
            "values": values,
            "parameters": parameters,
            "metrics": metrics,
        }
        # Gated inside the callback: under a batched gate (a resumed run) the
        # cond is a select, and this branch runs at every update.
        gated_log_callback(self.write, due, record, state.seed_index)
        return state.replace(
            probe_index=state.probe_index + 1, last_step=step, metrics=metrics
        )

    def pretrain(
        self, agent_state: Any, ext_state: ProbeState, context: ExtensionContext
    ) -> tuple[Any, ProbeState]:
        """Takes the seed's place in the batch and the untrained network's
        probe: off-policy, the first update comes after the warm-up."""
        index = 0 if context.seed_index is None else context.seed_index
        ext_state = ext_state.replace(seed_index=jnp.asarray(index))
        first = jnp.asarray(True)
        ext_state = self.measure(
            agent_state, ext_state, context.step, context.rng, first
        )
        layers, _, _ = self.read_networks(
            agent_state, ext_state.observations, ext_state.actions, context.rng
        )
        return agent_state, ext_state.replace(previous=jax.tree.map(jnp.ravel, layers))

    def post_update(
        self, agent_state: Any, ext_state: ProbeState, context: ExtensionContext
    ) -> tuple[Any, ProbeState]:
        interval = self.probe_interval
        due = context.step // interval > ext_state.last_step // interval

        def measured(state: ProbeState) -> ProbeState:
            return self.measure(agent_state, state, context.step, context.rng, due)

        def remembered(previous: dict) -> dict:
            del previous
            layers = self.read_networks(
                agent_state, ext_state.observations, ext_state.actions, context.rng
            )[0]
            return jax.tree.map(jnp.ravel, layers)

        # A deliberate workaround for XLA's buffers, so that a skipped probe
        # costs nothing. One cond reading the old activations and writing the
        # new ones keeps both in memory, so that each skip copies the old ones
        # through, at every step: 19 MB on SAC's defaults. Hence two conds and
        # the forward pass twice at a probe; the second's gate reads the
        # first's result, which tells XLA that nothing reads the old ones by
        # then and the new ones can take their place. They are carried flat,
        # as XLA can give a matrix another memory layout in one branch, which
        # the other would then copy into.
        ext_state = run_if(due, measured, ext_state)
        just_measured = ext_state.last_step == context.step
        previous = run_if(just_measured, remembered, ext_state.previous)
        return agent_state, ext_state.replace(previous=previous)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: ProbeState,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del agent_state, rng, context
        return ext_state.metrics

    def write(self, record: dict, seed_index: Any) -> None:
        """One probe's file, whole or not at all: a run stopped while writing
        leaves no file an analysis would stumble on."""
        activation_type = np.float16 if self.float16 else np.float32
        arrays = {"step": record["step"]}
        if record["values"] is not None:
            arrays["values"] = record["values"]
        arrays.update({f"metrics/{k}": v for k, v in record["metrics"].items()})
        for layer, activations in record["layers"].items():
            arrays[f"layers/{layer}"] = np.asarray(activations, activation_type)
        for network, tree in record["parameters"].items():
            flat = flatten_dict(tree, sep="/")
            arrays.update({f"parameters/{network}/{k}": v for k, v in flat.items()})
        path = Path(self.run_directory) / PROBE_FILE.format(
            seed=int(seed_index), probe=int(record["probe_index"])
        )
        temporary = path.with_suffix(".tmp")
        with open(temporary, "wb") as file:
            np.savez(file, **arrays)
        os.replace(temporary, path)
