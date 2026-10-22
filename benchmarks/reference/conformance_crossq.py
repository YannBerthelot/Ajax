"""The authors' CrossQ update, recorded for the conformance check (tests/conformance/test_crossq_original.py).

usage: python conformance_crossq.py ENV_ID OUT_DIR   (in envs/crossq: the original's own versions)

Imports nothing from Ajax. The authors' SAC (vendor/adityab_crossq) is built as their train.py builds
it with -algo crossq, but with 256-wide critics: the update's maths does not depend on the width, and
the paper's 2048 would make each record about half a gigabyte. It trains on a gymnasium task by
Stable-Baselines3's loop, uniform random actions for START steps and then the policy, one update per
step. Before the train() calls in CALLS (counted among those that update), the update is run on the
same state and a fresh batch eagerly, its normal draws taped (the next actions', then the actor's when
the policy delay lets it step), and again in float64 with the same draws. RENORM repeats the last call
with every Batch Renorm layer past its 100,000-pass warm-up. One .npz per record: ``in.*`` (the state),
``batch.*``, ``noise``, ``n_updates``, ``f32.*`` and ``f64.*``.
"""

import contextlib
import pathlib
import sys

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "adityab_crossq"))
from sbx import SAC
from sbx.common.type_aliases import ReplayBufferSamplesNp
from sbx.sac.utils import activation_fn

CALLS, RENORM = (1, 24), 24
START, SEED, WIDTH = 1_000, 0, 256
COMMIT = "a318a266679bec3f9d93c44e029860a6da077c89"


def crossq(env: gym.Env) -> SAC:
    """train.py's model for -algo crossq (train.py:126-136, :171-200), WIDTH-wide."""
    return SAC(
        "MlpPolicy",
        env,
        policy_kwargs={
            "activation_fn": activation_fn["relu"],
            "layer_norm": False,
            "batch_norm": True,
            "batch_norm_momentum": 0.99,
            "batch_norm_mode": "brn_actor",
            "dropout_rate": None,
            "n_critics": 2,
            "net_arch": {"pi": [256, 256], "qf": [WIDTH, WIDTH]},
            "optimizer_class": optax.adam,
            "optimizer_kwargs": {"b1": 0.5, "b2": 0.999},
        },
        gradient_steps=1,
        policy_delay=3,
        crossq_style=True,
        td3_mode=False,
        use_bnstats_from_live_net=False,
        policy_q_reduce_fn=jnp.min,
        learning_starts=START,
        learning_rate=1e-3,
        qf_learning_rate=1e-3,
        tau=1.0,
        gamma=0.99,
        buffer_size=10_000,
        seed=SEED,
    )


@contextlib.contextmanager
def taped(tape: list, replay: bool):
    """jax.random.normal logging its draws, or returning the logged ones in order (cast), eagerly."""
    original, drawn = jax.random.normal, iter(list(tape))

    def normal(key, shape=(), dtype=jnp.float32):
        if replay:
            return jnp.asarray(next(drawn), dtype).reshape(shape)
        value = original(key, shape, dtype)
        tape.append(np.asarray(value))
        return value

    jax.random.normal = normal
    try:
        with jax.disable_jit():
            yield
    finally:
        jax.random.normal = original


def flat(prefix: str, tree) -> dict:
    """Every leaf of ``tree`` under its dotted path."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        names = [
            str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p))))
            for p in path
        ]
        out[".".join([prefix, *names])] = np.asarray(leaf)
    return out


def state(qf, actor, ent) -> dict:
    """The update's state: the critics' and the actor's parameters, running statistics, Adams and
    step counts, and the temperature's (the critics' unused target copies left out)."""
    out = {}
    for name, s in (("qf", qf), ("actor", actor)):
        out |= flat(f"{name}.params", s.params) | flat(
            f"{name}.batch_stats", s.batch_stats
        )
        out |= flat(f"{name}.opt", s.opt_state) | {f"{name}.step": np.asarray(s.step)}
    return out | flat("ent.params", ent.params) | flat("ent.opt", ent.opt_state)


def renormed(train_state):
    """``train_state`` with every Batch Renorm layer past its warm-up."""
    stats = flax.traverse_util.flatten_dict(flax.core.unfreeze(train_state.batch_stats))
    stats = {
        k: (jnp.full_like(v, 100_000) if k[-1] == "steps" else v)
        for k, v in stats.items()
    }
    return train_state.replace(
        batch_stats=flax.core.freeze(flax.traverse_util.unflatten_dict(stats))
    )


def to64(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64)
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
        else x,
        tree,
    )


def record(model: SAC, name: str, out: pathlib.Path, renorm: bool) -> None:
    sample = model.replay_buffer.sample(model.batch_size, env=model._vec_normalize_env)
    batch = ReplayBufferSamplesNp(
        sample.observations.numpy(),
        sample.actions.numpy(),
        sample.next_observations.numpy(),
        sample.dones.numpy().flatten(),
        sample.rewards.numpy().flatten(),
    )
    # train()'s policy_delay_indices for one gradient step (sbx/sac/sac.py:201).
    due = (model._n_updates + 1) % model.policy_delay == 0
    indices = flax.core.FrozenDict({0: True} if due else {})
    qf, actor, ent = (
        model.policy.qf_state,
        model.policy.actor_state,
        model.ent_coef_state,
    )
    if renorm:
        qf, actor = renormed(qf), renormed(actor)

    def update(qf, actor, ent, batch):
        return model._train(
            model.crossq_style,
            model.td3_mode,
            model.use_bnstats_from_live_net,
            model.gamma,
            model.tau,
            model.target_entropy,
            1,
            batch,
            indices,
            qf,
            actor,
            ent,
            model.key,
            model.policy_q_reduce_fn,
        )[:3]

    tape: list = []
    with taped(tape, replay=False):
        new32 = update(qf, actor, ent, batch)
    with jax.experimental.enable_x64(), taped(tape, replay=True):
        new64 = update(to64(qf), to64(actor), to64(ent), to64(batch))
    arrays = {f"in.{k}": v for k, v in state(qf, actor, ent).items()}
    arrays |= {f"batch.{f}": np.asarray(getattr(batch, f)) for f in batch._fields}
    arrays |= {f"f32.{k}": v for k, v in state(*new32).items()}
    arrays |= {f"f64.{k}": v for k, v in state(*new64).items()}
    arrays |= {"noise": np.stack(tape), "n_updates": np.asarray(model._n_updates)}
    np.savez(out / f"{name}.npz", **arrays)


class Recorded(SAC):
    """The authors' SAC recording the updates of the train() calls in CALLS."""

    out: pathlib.Path
    env_id: str
    calls = 0

    def train(self, batch_size, gradient_steps):
        self.calls += 1
        if self.calls in CALLS:
            record(self, f"{self.env_id}_{self.calls}", self.out, renorm=False)
        if self.calls == RENORM:
            record(self, f"{self.env_id}_{self.calls}renorm", self.out, renorm=True)
        super().train(batch_size, gradient_steps)


def main(env_id: str, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = crossq(gym.make(env_id))
    model.__class__, model.out, model.env_id = Recorded, out, env_id
    model.learn(total_timesteps=START + max(CALLS) + 1)
    (out / "versions.txt").write_text(
        f"adityab/CrossQ {COMMIT}\njax {jax.__version__}\nflax {flax.__version__}\n"
        f"optax {optax.__version__}\n"
    )


if __name__ == "__main__":
    main(sys.argv[1], pathlib.Path(sys.argv[2]))
