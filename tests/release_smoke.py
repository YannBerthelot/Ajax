"""One smoke test per public entry point of an installed Ajax, for the
Release workflow's installed-package gate: the core install, and each extra
installed on its own.

    python tests/release_smoke.py core|brax|playground|gymnasium|compare|reproduction|wandb|cuda

Each check is the smallest real use, seconds long. It imports only the
standard library and what the install it checks brings, not even pytest,
and nothing from the checkout.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys
import tempfile


def core() -> None:
    import numpy as np

    from ajax import PPO
    from ajax.checkpoint import load_checkpoint, save_checkpoint
    from ajax.compare import AdaStop
    from ajax.logging import (
        LoggingConfig,
        TensorBoardBackend,
        load_scalars_from_tfevents,
    )

    agent = PPO("CartPole-v1", n_envs=1, n_steps=16, batch_size=8)
    logging = LoggingConfig(log_frequency=16, backends=())
    state, evaluations = agent.train(
        seed=[0, 1], n_timesteps=32, logging_config=logging
    )
    assert evaluations["Eval/episodic mean reward"].shape == (2, 2), evaluations

    # CartPole's two actions; a zero observation is the upright start.
    observations = np.zeros((3, 4), np.float32)
    policy = agent.policy(state, seed_index=1)
    actions, _ = policy.act(observations, policy.reset(3))
    assert set(np.asarray(actions).tolist()).issubset({0, 1}), actions
    action = agent.predict(state, observations[0])

    with tempfile.TemporaryDirectory() as folder:
        save_checkpoint(state, f"{folder}/ppo.pkl")
        loaded = load_checkpoint(agent, f"{folder}/ppo.pkl")
        assert agent.predict(loaded, observations[0]) == action

        logging = LoggingConfig(
            log_frequency=16, backends=(TensorBoardBackend(folder),)
        )
        agent.train(seed=[0], n_timesteps=32, logging_config=logging)
        run = f"{folder}/tensorboard/{agent.run_ids[0]}"
        steps = [
            step
            for step, _ in load_scalars_from_tfevents(run)["Eval/episodic_mean_reward"]
        ]
        assert steps == [16, 32], steps

    # AdaStop needs no extra. Scores ten apart, far beyond their spread: the
    # first agent's mean is larger.
    test = AdaStop(["high", "low"], n=5, K=2, B=1_000)
    while not test.finished:
        test.update({"high": 10.0 + np.arange(5.0), "low": np.arange(5.0)})
    assert test.decisions == {("high", "low"): "larger"}, test.decisions


def step_env(env_id: str) -> None:
    """Builds two copies of env_id, resets and steps them once."""
    import jax
    import jax.numpy as jnp

    from ajax.environments.create import build_env_from_id

    env, _ = build_env_from_id(env_id, n_envs=2, episode_length=10)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jnp.zeros((2, env.action_size)))
    assert state.reward.shape == (2,) and bool(jnp.isfinite(state.reward).all())


def gymnasium_env() -> None:
    """Builds two copies of Gymnasium's HalfCheetah (MuJoCo comes with the
    extra), resets and steps them once from compiled code."""
    import warnings

    import jax
    import jax.numpy as jnp

    from ajax.environments.create import build_env_from_id
    from ajax.environments.gymnasium import PythonEnvironmentWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PythonEnvironmentWarning)
        env, params = build_env_from_id("gymnasium:HalfCheetah-v5")
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    _, state = jax.jit(jax.vmap(env.reset))(keys)
    action = jnp.zeros((2, *env.action_space(params).shape))
    _, _, reward, *_ = jax.jit(jax.vmap(env.step))(keys, state, action)
    assert reward.shape == (2,) and bool(jnp.isfinite(reward).all())


def compare() -> None:
    """estimate's confidence sequence (adastop) on made-up scores, all 0.5:
    an agent whose train returns them instead of training."""
    import numpy as np

    from ajax.compare import estimate, half_width

    class MadeUp:
        def train(self, seed, **options):
            return None, {"Eval/episodic mean reward": np.full((len(seed), 1), 0.5)}

    def last(evaluations):
        return evaluations["Eval/episodic mean reward"][:, -1]

    result = estimate(MadeUp, 1, 1, {"score": (last, (0.0, 1.0), half_width(0.5))}, n=8)
    lower, upper = result.interval["score"]
    assert result.mean["score"] == 0.5 and lower <= 0.5 <= upper, result


def reproduction() -> None:
    """load on a local copy of the published tables: no network."""
    import huggingface_hub  # noqa: F401 (load's download, not called)
    import pandas as pd

    from ajax.reproduction import load

    runs = pd.DataFrame({"agent": ["PPO", "PPO"], "seed": [0, 1], "return": [1.5, 2.5]})
    with tempfile.TemporaryDirectory() as folder:
        runs.to_parquet(f"{folder}/runs.parquet")
        pathlib.Path(folder, "summary.jsonl").write_text(
            '{"agent": "PPO", "verdict": "equal"}\n'
        )
        assert load("runs", folder).equals(runs)
        assert load("summary", folder)["verdict"].tolist() == ["equal"]


def wandb_backend() -> None:
    """The backend and the package, without an account: nothing is sent."""
    import wandb  # noqa: F401

    from ajax.logging import LoggingConfig, WandbBackend

    LoggingConfig(backends=(WandbBackend(project="ajax-smoke"),))


def cuda() -> None:
    """GitHub's runners have no GPU: JAX finds none and falls back to the
    CPU, so the check is that the CUDA plugin is installed and JAX loads."""
    import jax

    names = [
        dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()
    ]
    assert any(name.startswith(("jax-cuda", "jax_cuda")) for name in names), (
        "no JAX CUDA plugin installed"
    )
    print(jax.devices())


CHECKS = {
    "core": core,
    "brax": lambda: step_env("fast"),
    "playground": lambda: step_env("CartpoleBalance"),
    "gymnasium": gymnasium_env,
    "compare": compare,
    "reproduction": reproduction,
    "wandb": wandb_backend,
    "cuda": cuda,
}

if __name__ == "__main__":
    # A script's own directory comes first on sys.path: here tests/, whose
    # tests/logging package would hide the standard library's logging,
    # which JAX imports.
    here = pathlib.Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if pathlib.Path(entry).resolve() != here]
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        sys.exit(f"usage: python tests/release_smoke.py {'|'.join(CHECKS)}")
    CHECKS[sys.argv[1]]()
    print(f"{sys.argv[1]}: ok")
