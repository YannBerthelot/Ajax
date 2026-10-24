"""Cal-QL's update (JaxCQL's ConservativeSAC), recorded for the conformance check of Ajax's Cal-QL.

usage: python conformance_calql.py CLONE ENV_ID OUT_DIR   (in envs/jax_offline: see its pyproject)

CLONE is a checkout of nakamotoo/Cal-QL at COMMIT. The repository has no licence, so it is not
vendored: its records are committed instead (tests/conformance/records). Imports nothing from Ajax.
The dataset is conformance_d4rl's, turned into rows and returns-to-go by the original's
qlearning_dataset_and_calc_mc (replay_buffer.py:218-261) through a stand-in env serving it, with
dense rewards (is_sparse_reward False) and conservative_sac_main's other dataset settings. The agent is
conservative_sac_main's (its flags' defaults, ConservativeSAC's config, target entropy -A), with ARCH
networks and batch BATCH so that the records stay small; it trains by main()'s offline loop,
subsample_batch then train(), whose body is written out here to keep each step's key. Before the steps
in RECORDS the step runs eagerly on the same state, batch and key, every normal and uniform draw taped
in order, and again in float64 with the same draws. One .npz per record: ``in.*`` (the state),
``batch.*``, ``draw.<i>.<normal|uniform>``, ``f32.*`` and ``f64.*``.
"""

import contextlib
import pathlib
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
from conformance_d4rl import dataset

COMMIT = "ac6eafe"
RECORDS = (1, 1000)
ARCH, BATCH, SEED = "32-32", 64, 0
CQL_MIN_Q_WEIGHT, ENABLE_CALQL = 5.0, True  # conservative_sac_main.py:53-55


@contextlib.contextmanager
def taped(tape: list, replay: bool):
    """jax.random.normal and uniform logging their draws in order, or returning the logged ones
    (cast), eagerly. A draw on a traced key is passed through untaped: flax checks a parameter's
    shape by evaluating its initialiser abstractly."""
    normal, uniform, drawn = jax.random.normal, jax.random.uniform, iter(list(tape))

    def logged(kind, fresh, shape, dtype, key):
        if isinstance(key, jax.core.Tracer):
            return fresh()
        if replay:
            got, value = next(drawn)
            assert got == kind, (got, kind)
            return jnp.asarray(value, dtype).reshape(shape)
        value = fresh()
        tape.append((kind, np.asarray(value)))
        return value

    def normal_(key, shape=(), dtype=jnp.float32):
        return logged("normal", lambda: normal(key, shape, dtype), shape, dtype, key)

    def uniform_(key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0):
        fresh = lambda: uniform(key, shape, dtype, minval, maxval)  # noqa: E731
        return logged("uniform", fresh, shape, dtype, key)

    jax.random.normal, jax.random.uniform = normal_, uniform_
    try:
        with jax.disable_jit():
            yield
    finally:
        jax.random.normal, jax.random.uniform = normal, uniform


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


def state(train_states: dict, target: dict) -> dict:
    """The policy, both critics and log_alpha with their Adams and steps, and the target critics."""
    out = flat("target", target)
    for name, s in train_states.items():
        out |= flat(f"{name}.params", s.params) | flat(f"{name}.opt", s.opt_state)
        out[f"{name}.step"] = np.asarray(s.step)
    return out


def to64(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64)
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
        else x,
        tree,
    )


def main(clone: pathlib.Path, env_id: str, out: pathlib.Path) -> None:
    sys.path.insert(0, str(clone))
    sys.modules.setdefault(
        "d4rl", types.ModuleType("d4rl")
    )  # for D4RL's files: not read here
    from JaxCQL.conservative_sac import ConservativeSAC
    from JaxCQL.jax_utils import batch_to_jax, init_rng, next_rng
    from JaxCQL.model import FullyConnectedQFunction, TanhGaussianPolicy
    from JaxCQL.replay_buffer import qlearning_dataset_and_calc_mc, subsample_batch

    np.random.seed(SEED)  # set_random_seed (utils.py), which also imports wandb
    init_rng(SEED)
    data = dataset(env_id)
    stand_in = types.SimpleNamespace(
        get_dataset=lambda: data, spec=types.SimpleNamespace(name=env_id)
    )
    rows = qlearning_dataset_and_calc_mc(
        stand_in, 1.0, 0.0, 0.99999, 0.99, is_sparse_reward=False
    )
    # get_d4rl_dataset_with_mc_calculation's renaming (replay_buffer.py:115-122).
    rows = {
        "observations": rows["observations"],
        "actions": rows["actions"],
        "next_observations": rows["next_observations"],
        "rewards": rows["rewards"],
        "dones": rows["terminals"].astype(np.float32),
        "mc_returns": rows["mc_returns"],
    }
    obs_dim, action_dim = rows["observations"].shape[1], rows["actions"].shape[1]
    policy = TanhGaussianPolicy(obs_dim, action_dim, ARCH, True, 1.0, -1.0)
    qf = FullyConnectedQFunction(obs_dim, action_dim, ARCH, True)
    config = ConservativeSAC.get_default_config()
    config.target_entropy = -float(action_dim)  # conservative_sac_main.py:102-103
    sac = ConservativeSAC(config, policy, qf)
    args = (True, CQL_MIN_Q_WEIGHT, ENABLE_CALQL)
    for step in range(1, max(RECORDS) + 1):
        batch = batch_to_jax(subsample_batch(rows, BATCH))
        sac._total_steps += 1  # ConservativeSAC.train's body, its key kept
        rng = next_rng()
        if step in RECORDS:
            tape: list = []
            with taped(tape, replay=False):
                new32 = sac._train_step(
                    sac._train_states, sac._target_qf_params, rng, batch, *args
                )
            with jax.experimental.enable_x64(), taped(tape, replay=True):
                new64 = sac._train_step(
                    to64(sac._train_states),
                    to64(sac._target_qf_params),
                    rng,
                    to64(batch),
                    *args,
                )
            arrays = {
                f"in.{k}": v
                for k, v in state(sac._train_states, sac._target_qf_params).items()
            }
            arrays |= {f"batch.{k}": np.asarray(v) for k, v in batch.items()}
            arrays |= {f"draw.{i}.{kind}": v for i, (kind, v) in enumerate(tape)}
            arrays |= {f"f32.{k}": v for k, v in state(*new32[:2]).items()}
            arrays |= {f"f64.{k}": v for k, v in state(*new64[:2]).items()}
            np.savez_compressed(out / f"{env_id}_{step}.npz", **arrays)
        sac._train_states, sac._target_qf_params, _ = sac._train_step(
            sac._train_states, sac._target_qf_params, rng, batch, *args
        )


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    main(pathlib.Path(sys.argv[1]), sys.argv[2], out)
