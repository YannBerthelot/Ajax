"""Exact before/after pins for a behaviour-neutral restructure (a tool run by
hand, outside CI): tiny runs whose every final-state leaf and logged
evaluation is saved, with the text of their training program's jaxpr. On one
machine, a restructure that changes nothing reproduces both exactly::

    JAX_PLATFORMS=cpu uv run python -m tests.exact_pins record DIR [CASE ...]
    JAX_PLATFORMS=cpu uv run python -m tests.exact_pins compare DIR [CASE ...]

``compare`` prints per case BITWISE or each differing leaf's largest relative
gap, and whether the program's text is the same. test_agent_pins'
fingerprints hold to 1e-3 across platforms; these hold to the bit on one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from optax.tree_utils import tree_norm

from ajax import APO, ASAC, AVG, DQN, IQL, PPO, PQN, REDQ, SAC, TD3, TQC, CrossQ, DAgger
from ajax.agents.TD3.TD3 import make_default_action_pipeline
from ajax.extensions.base import Extension
from ajax.extensions.expert import ExpertGains, ImitationLoss, ResidualPolicy
from ajax.extensions.observe import ExpertActionInput, TrainingProgressInput
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDPolicy
from ajax.networks.memory import MemoryConfig
from tests.agents.test_offline import episodes
from tests.modules import setpoint_env

SEEDS = [0, 1]


@dataclass(frozen=True)
class Keyed(Extension):
    """Draws on ``ctx.rng`` in every phase: a moved key moves the numbers."""

    name: str = "keyed"

    def init_state(self, agent_state, rng):
        return jax.random.normal(rng)

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target + 0.01 * jax.random.normal(ctx.rng)

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        params = batch.get("critic_params", batch.get("q_state"))
        if params is None:
            return 0.0
        params = getattr(params, "params", params)
        return 1e-3 * (1 + jax.random.normal(ctx.rng)) * tree_norm(params, squared=True)

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        draw = jax.random.normal(ctx.rng)
        loss = 1e-3 * (1 + draw) * tree_norm(batch["actor_params"], squared=True)
        return loss, {"keyed/draw": draw}

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 0.01 * jax.random.normal(ctx.rng)

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"keyed": ext_state}


@dataclass(frozen=True)
class ValueKeyed(Keyed):
    actor_loss = Extension.actor_loss


@dataclass(frozen=True)
class ActorKeyed(Keyed):
    on_target = Extension.on_target
    critic_loss = Extension.critic_loss


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def choose_one(obs: jax.Array) -> jax.Array:
    return jnp.ones(obs.shape[:-1] + (1,), jnp.int32)


class GainExpert:
    """A stateful expert with a learnable gain, in Pendulum's torque units."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args: Any) -> Any:
        state, obs = args[-2:]
        out = self.step_with_gains(state, obs, self.anchor_gains)
        return out if len(args) == 2 else out[0]

    def step_with_gains(self, state: jax.Array, obs: jax.Array, gains: Any) -> tuple:
        return jnp.clip(-gains[0] * obs[..., :1], -2.0, 2.0), state + obs[..., :1]


S: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2}
S |= {"actor_architecture": ("32", "relu"), "critic_architecture": ("32", "relu")}
R = {"buffer_size": 1000, "batch_size": 16, "learning_starts": 20}
SAC40 = {**S, **R, "policy_update_start": 40, "alpha_update_start": 40}
ON = {"env_id": "Pendulum-v1", "n_envs": 2, "n_steps": 32, "batch_size": 16}
ON |= {"actor_architecture": ("16", "tanh"), "critic_architecture": ("16", "tanh")}
CART = {"env_id": "CartPole-v1", "n_envs": 2, **R, "architecture": ("16", "relu")}
PQN_KW = {"env_id": "CartPole-v1", "n_envs": 4, "n_steps": 16, "num_minibatches": 2}
PQN_KW |= {"n_epochs": 2, "architecture": ("16", "relu")}
SETPOINT = dict(zip(("env_id", "env_params"), setpoint_env.make(), strict=True))
PID = {**SETPOINT, "n_envs": 2, "extensions": (setpoint_env.TERMS,)}
PID |= {"actor_architecture": PIDPolicy(), "critic_architecture": ("16", "relu")}
NORM = {"normalize_observations": True}
GRU = {"learning_starts": 60, "memory": MemoryConfig(kind="gru", hidden_size=8)}
DAGGER = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_architecture": ("32", "relu")}
DAGGER |= {"buffer_size": 1000, "batch_size": 16, "expert_policy": push}
CROSSQ = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_width": 32, "critic_width": 64}
CROSSQ |= {**R, "learning_starts": 40}
# The paths a replay agent's options take: a residual policy (its executed
# and chosen actions), an expert's stored actions and imitation, gain mode,
# observation features (one reading the progress), the normalisers.
RESIDUAL = {"expert_policy": push, "extensions": (ResidualPolicy(), Keyed())}
BC = {"expert_policy": push, "extensions": (ImitationLoss(push, coef=1.0),)}
GAIN = {"expert_policy": GainExpert(), "extensions": (ExpertGains(),)}
OBSERVE: dict[str, Any] = {"expert_policy": push}
OBSERVE |= {"extensions": (ExpertActionInput(), TrainingProgressInput())}
REWARD_NORM = {"normalize_rewards": True, "reward_normalization_gamma": 0.99}
OFFLINE_KW = {**S, **R, "learning_starts": 0, "batch_size": 8}
#: Trained offline for their steps (updates on a dataset's rows), then
#: online for ONLINE steps from that state (train_offline, then train).
OFFLINE = {"IQL-offline", "TD3-offline"}
ONLINE = 40
CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "TD3": (TD3, {**S, **R}, 200),
    "TD3-keyed": (TD3, {**S, **R, "extensions": (Keyed(),)}, 200),
    "TD3-gru": (TD3, {**S, **R, **GRU}, 160),
    "TD3-gru-stored": (TD3, {**S, **R, **GRU, "stored_state": True}, 160),
    "TD3-expert-bc": (TD3, {**S, **R, **BC}, 200),
    "TD3-residual": (TD3, {**S, **R, **RESIDUAL}, 200),
    "TD3-gain": (TD3, {**S, **R, **GAIN}, 200),
    "TD3-observe": (TD3, {**S, **R, **OBSERVE}, 200),
    "TD3-norm": (TD3, {**S, **R, **NORM}, 200),
    "TD3-norm-rewards": (TD3, {**S, **R, **REWARD_NORM}, 200),
    "TD3-pipeline": (
        TD3,
        {**S, **R, "action_pipeline": make_default_action_pipeline(False, 0.3)},
        200,
    ),
    "TD3-offline": (TD3, OFFLINE_KW, 80),
    "AVG": (AVG, {**S, "learning_starts": 20}, 200),
    "AVG-keyed": (AVG, {**S, "learning_starts": 20, "extensions": (Keyed(),)}, 200),
    "DAgger-keyed": (DAgger, {**DAGGER, "extensions": (ActorKeyed(),)}, 200),
    "DAgger-discrete": (
        DAgger,
        {**DAGGER, "env_id": "CartPole-v1", "actor_architecture": ("16", "relu")}
        | {"expert_policy": choose_one},
        200,
    ),
    "IQL": (IQL, {**S, **R}, 200),
    "IQL-keyed": (IQL, {**S, **R, "extensions": (Keyed(),)}, 200),
    "IQL-residual": (IQL, {**S, **R, **RESIDUAL}, 200),
    "IQL-gain": (IQL, {**S, **R, **GAIN}, 200),
    "IQL-observe-norm": (IQL, {**S, **R, **OBSERVE, **NORM}, 200),
    "IQL-offline": (IQL, OFFLINE_KW, 80),
    "DQN": (DQN, {**CART, "target_update_interval": 10}, 240),
    "DQN-keyed": (
        DQN,
        {**CART, "target_update_interval": 10, "extensions": (ValueKeyed(),)},
        240,
    ),
    "PQN-keyed": (PQN, {**PQN_KW, "extensions": (ValueKeyed(),)}, 512),
    "PQN-norm": (PQN, {**PQN_KW, **NORM}, 512),
    "SAC-keyed": (
        SAC,
        {**S, **R, "policy_update_start": 80, "alpha_update_start": 120}
        | {"target_entropy_initial_per_dim": -3.0, "extensions": (Keyed(),)},
        240,
    ),
    "SAC-gru": (
        SAC,
        {**SAC40, **GRU, "policy_update_start": 80, "alpha_update_start": 80},
        160,
    ),
    "SAC-residual": (
        SAC,
        {**SAC40, "expert_policy": push, "extensions": (ResidualPolicy(), Keyed())},
        160,
    ),
    "SAC-expert-bc": (
        SAC,
        {
            **SAC40,
            "expert_policy": push,
            "extensions": (ImitationLoss(push, coef=1.0),),
        },
        160,
    ),
    "SAC-expert": (
        SAC,
        {**SAC40, "expert_policy": push, "expert_buffer_n_steps": 40},
        160,
    ),
    "SAC-augment": (
        SAC,
        {**SAC40, "expert_policy": push, "augment_obs_with_expert_action": True},
        160,
    ),
    "SAC-gain": (
        SAC,
        {**SAC40, "expert_policy": GainExpert(), "use_pid_policy": True},
        160,
    ),
    "SAC-augment-state": (
        SAC,
        {**SAC40, "expert_policy": GainExpert(), "augment_obs_with_expert_state": True},
        160,
    ),
    "SAC-train-frac": (SAC, {**SAC40, "use_train_frac": True}, 160),
    "SAC-norm": (SAC, {**SAC40, **NORM}, 160),
    "SAC-fixed": (SAC, {**SAC40, "fixed_alpha": True, "policy_update_start": 60}, 160),
    "SAC-pid-norm": (
        SAC,
        {**PID, **R, **NORM, "policy_update_start": 40, "alpha_update_start": 40},
        160,
    ),
    "REDQ": (REDQ, {**S, **R}, 120),
    "ASAC": (ASAC, {**S, **R}, 160),
    "CrossQ": (CrossQ, CROSSQ, 160),
    "CrossQ-keyed": (CrossQ, {**CROSSQ, "extensions": (Keyed(),)}, 160),
    "CrossQ-residual": (CrossQ, {**CROSSQ, **RESIDUAL}, 160),
    "CrossQ-expert-bc": (CrossQ, {**CROSSQ, **BC}, 160),
    "CrossQ-gain": (CrossQ, {**CROSSQ, **GAIN}, 160),
    "CrossQ-observe": (CrossQ, {**CROSSQ, **OBSERVE}, 160),
    "CrossQ-norm": (CrossQ, {**CROSSQ, **NORM, **REWARD_NORM}, 160),
    "PPO-keyed": (PPO, {**ON, "extensions": (Keyed(),)}, 128),
    "PPO-norm": (PPO, {**ON, **NORM}, 128),
    "PPO-norm-brax": (PPO, {**ON, **NORM, "env_id": "fast"}, 128),
    "PPO-pid-norm": (
        PPO,
        {**PID, **NORM, "n_steps": 16, "batch_size": 8, "n_epochs": 1},
        128,
    ),
    "APO-norm": (APO, {**ON, **NORM}, 128),
    # The time geometry with brax's forced resets, and recurrent PPO.
    "PPO-time-reset": (
        PPO,
        {**ON, "n_envs": 4, "n_steps": 16, "num_minibatches": 2, "unroll_length": 16}
        | {"num_resets_per_eval": 1, "num_evals": 3, **NORM},
        256,
    ),
    "PPO-gru": (
        PPO,
        {**ON, "n_steps": 16, "num_minibatches": 2, "bptt_length": 4}
        | {"memory": MemoryConfig(kind="gru", hidden_size=8)},
        128,
    ),
    "TQC": (TQC, {**S, **R, "num_critics": 3, "n_quantiles": 5}, 160),
}


def run(name: str, agent: Any, logging: Any = None) -> tuple[Any, Any]:
    """The case's training: ``train`` for its steps or, OFFLINE,
    ``train_offline`` for as many updates on test_offline's dataset."""
    steps = CASES[name][2]
    if name in OFFLINE:
        data = episodes(3, discrete=False)  # Pendulum's observations
        return agent.train_offline(data, steps, seed=SEEDS, logging_config=logging)
    return agent.train(SEEDS, n_timesteps=steps, logging_config=logging)


def record(state: Any, evals: Any, tag: str = "") -> dict[str, np.ndarray]:
    """Every final-state leaf (keys as their data) and logged evaluation."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        out[f"{tag}state" + jax.tree_util.keystr(path)] = np.asarray(leaf)
    return out | {f"{tag}eval/" + k: np.asarray(v) for k, v in (evals or {}).items()}


def leaves(name: str) -> dict[str, np.ndarray]:
    """The case's :func:`record`, then an OFFLINE case's online one."""
    cls, kwargs, _ = CASES[name]
    logging = LoggingConfig(config={}, log_frequency=40, use_wandb=False)
    agent = cls(**kwargs)
    state, evals = run(name, agent, logging)
    out = record(state, evals)
    if name in OFFLINE:
        state, evals = agent.train(
            SEEDS, n_timesteps=ONLINE, initial_state=state, logging_config=logging
        )
        out |= record(state, evals, tag="online/")
    return out


class _Built(Exception):
    pass


def program(name: str) -> str:
    """The text of the case's training program: the jaxpr of the function
    ``train`` vmaps over the seeds (logging off: its run ids vary)."""
    cls, kwargs, _ = CASES[name]
    agent, built = cls(**kwargs), []
    make_train = agent.get_make_train

    def spy() -> Any:
        def build(**kw: Any) -> None:
            built.append(make_train()(**kw))
            raise _Built

        return build

    agent.get_make_train = spy  # type: ignore[method-assign]
    try:
        run(name, agent)
    except _Built:
        pass
    keys = jax.vmap(jax.random.PRNGKey)(jnp.array(SEEDS))
    return str(jax.make_jaxpr(jax.vmap(built[0]))(keys, jnp.arange(len(SEEDS))))


def differences(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> list[str]:
    """Each leaf that is not bitwise the same, with its largest relative gap."""
    found = [f"{k}: only in one record" for k in sorted(ref.keys() ^ got.keys())]
    for k in sorted(ref.keys() & got.keys()):
        a, b = ref[k], got[k]
        if a.shape != b.shape:
            found.append(f"{k}: shape {a.shape} -> {b.shape}")
        elif not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            gap = np.abs(a.astype(np.float64) - b) / np.maximum(np.abs(a), 1e-6)
            found.append(f"{k}: largest relative gap {float(np.nanmax(gap)):.2e}")
    return found


def main(argv: list[str]) -> None:
    mode, folder, names = argv[0], Path(argv[1]), argv[2:] or list(CASES)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        got, text = leaves(name), program(name)
        if mode == "record":
            np.savez(folder / f"{name}.npz", **got)  # type: ignore[arg-type]
            (folder / f"{name}.jaxpr").write_text(text)
            print(f"{name}: recorded {len(got)} arrays")
            continue
        found = differences(dict(np.load(folder / f"{name}.npz")), got)
        same = (folder / f"{name}.jaxpr").read_text() == text
        print(
            f"{name}: {'BITWISE' if not found else f'{len(found)} differ'}; jaxpr "
            + ("same" if same else "differs")
        )
        for line in found:
            print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1:])
