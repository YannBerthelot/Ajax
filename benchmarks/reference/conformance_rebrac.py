"""ReBRAC's update, recorded for the conformance check (tests/conformance/test_rebrac_original.py).

usage: python conformance_rebrac.py ENV_ID OUT_DIR   (in envs/jax_offline: see its pyproject for the versions)

Imports nothing from Ajax. The original (vendor/tinkoff_rebrac) learns from a D4RL dataset; here the
dataset is conformance_d4rl's (uniform random actions on MountainCarContinuous and on MountainCarEnds,
which terminates every 20th step, both with [-1, 1] boxes), turned into rows by the original's
qlearning_dataset. The networks, states and
hyperparameters are main()'s (rebrac.py:226-269, Config's defaults), and the training is main()'s loop
body (td3_loop_update_step: update_td3 at even updates, update_td3_no_targets at odd ones). Before the
updates in RECORDS, the update is run eagerly on the same state and batch, the critic's target-noise
draw taped, and again in float64 with the same draw. One .npz per record: ``in.*`` (the state),
``batch.*``, ``noise``, ``update``, ``f32.*`` and ``f64.*``.
"""

import contextlib
import functools
import pathlib
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "tinkoff_rebrac"))
# What only main() and the pixel datasets import: wandb logs, pyrallis parses the command line, d4rl
# and h5py read D4RL's files, vd4rl_utils (dm_control) the pixel ones. No update reaches them.
for _name in ("wandb", "d4rl", "h5py", "pyrallis", "src.utils.vd4rl_utils"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["pyrallis"].wrap = lambda *_args, **_kwargs: (lambda f: f)
sys.modules["src.utils.vd4rl_utils"].ExtendedTimeStep = None
sys.modules["src.utils.vd4rl_utils"].step_type_lookup = None
from conformance_d4rl import dataset  # noqa: E402
from src.algorithms import rebrac  # noqa: E402
from src.networks import DetActor, EnsembleCritic  # noqa: E402
from src.utils.buffer import ReplayBuffer, qlearning_dataset  # noqa: E402
from src.utils.common import Metrics  # noqa: E402

RECORDS = (0, 1, 1000, 1001)
METRICS = ["critic_loss", "q_min", "actor_loss", "batch_entropy"]
METRICS += [
    "bc_mse_policy",
    "bc_mse_random",
    "action_mse",
]  # main()'s bc_metrics_to_log


def buffer_of(rows: dict) -> ReplayBuffer:
    """create_from_d4rl's buffer (buffer.py:109-121) of ``rows``, unnormalised (Config's defaults)."""
    buffer = ReplayBuffer()
    buffer.data = {
        "states": jnp.asarray(rows["observations"], dtype=jnp.float32),
        "actions": jnp.asarray(rows["actions"], dtype=jnp.float32),
        "rewards": jnp.asarray(rows["rewards"], dtype=jnp.float32),
        "next_states": jnp.asarray(rows["next_observations"], dtype=jnp.float32),
        "next_actions": jnp.asarray(rows["next_actions"], dtype=jnp.float32),
        "dones": jnp.asarray(rows["terminals"], dtype=jnp.float32),
    }
    return buffer


def init(config: rebrac.Config, buffer: ReplayBuffer):
    """main()'s actor and critic train states (rebrac.py:229-252)."""
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    actor_module = DetActor(
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=config.actor_ln,
        n_hiddens=config.actor_n_hiddens,
    )
    actor = rebrac.ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        target_params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=2,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
    )
    critic = rebrac.CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )
    return key, actor, critic


def updates(config: rebrac.Config):
    """main()'s two partial updates (rebrac.py:254-268): with the actor, and the critic alone."""
    common = {
        "gamma": config.gamma,
        "actor_bc_coef": config.actor_bc_coef,
        "critic_bc_coef": config.critic_bc_coef,
        "tau": config.tau,
        "policy_noise": config.policy_noise,
        "noise_clip": config.noise_clip,
    }
    full = functools.partial(
        rebrac.update_td3, **common, normalize_q=config.normalize_q
    )
    return full, functools.partial(rebrac.update_td3_no_targets, **common)


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


def state(actor, critic) -> dict:
    """Both networks, their targets, Adams and step counts."""
    out = {}
    for name, s in (("actor", actor), ("critic", critic)):
        out |= flat(f"{name}.params", s.params) | flat(
            f"{name}.target", s.target_params
        )
        out |= flat(f"{name}.opt", s.opt_state) | {f"{name}.step": np.asarray(s.step)}
    return out


def to64(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64)
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
        else x,
        tree,
    )


def main(env_id: str, out: pathlib.Path) -> None:
    config = rebrac.Config()
    buffer = buffer_of(qlearning_dataset(None, dataset=dataset(env_id)))
    key, actor, critic = init(config, buffer)
    metrics = Metrics.create(METRICS)
    full, critic_only = updates(config)
    jitted = jax.jit(full), jax.jit(critic_only)
    for i in range(max(RECORDS) + 1):
        # td3_loop_update_step (rebrac.py:270-291), the delayed_updates of i % policy_freq == 0.
        step_key, batch_key = jax.random.split(key)
        batch = buffer.sample_batch(batch_key, batch_size=config.batch_size)
        with_actor = i % config.policy_freq == 0
        if i in RECORDS:
            update = full if with_actor else critic_only
            tape: list = []
            with taped(tape, replay=False):
                new32 = update(
                    key=step_key,
                    actor=actor,
                    critic=critic,
                    batch=batch,
                    metrics=metrics,
                )
            with jax.experimental.enable_x64(), taped(tape, replay=True):
                new64 = update(
                    key=step_key,
                    actor=to64(actor),
                    critic=to64(critic),
                    batch=to64(batch),
                    metrics=metrics,
                )
            arrays = {f"in.{k}": v for k, v in state(actor, critic).items()}
            arrays |= {f"batch.{k}": np.asarray(v) for k, v in batch.items()}
            arrays |= {f"f32.{k}": v for k, v in state(*new32[1:3]).items()}
            arrays |= {f"f64.{k}": v for k, v in state(*new64[1:3]).items()}
            arrays |= {"noise": np.stack(tape), "update": np.asarray(i)}
            np.savez(out / f"{env_id}_{i}.npz", **arrays)
        key, actor, critic, metrics = jitted[0 if with_actor else 1](
            key=step_key, actor=actor, critic=critic, batch=batch, metrics=metrics
        )


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], out)
