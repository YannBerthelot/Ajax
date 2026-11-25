"""Spinning Up's VPG (vendor/openai_spinningup), the REINFORCE reference.

usage: python refcmp_spinup.py NAME reference FIRST_SEED N_SEEDS (refcmp_worker)

Imports nothing from Ajax. ``vpg()`` runs unmodified: the ``spinup`` modules
it imports are replaced by single-process stand-ins (its MPI helpers: one
process, whose statistics are the batch's own; its logger, which also
evaluates the policy), ``gym.spaces`` by gymnasium's, and its environment by
the gymnax environment Ajax trains on, through the old gym API ``vpg`` steps
(:class:`OldGymEnv`). One torch thread per process; one process per seed.
"""

import pathlib
import sys
import types

VENDOR = pathlib.Path(__file__).parent / "vendor" / "openai_spinningup"


class OldGymEnv:
    """A gymnax env as the old gym API vpg steps: ``reset() -> obs``,
    ``step(a) -> (obs, reward, done, info)``, done at a termination or the
    time limit, obs the episode's last observation then (vpg resets itself)."""

    def __init__(self, env_id: str, seed: int):
        import gymnasium
        import gymnax
        import jax
        import numpy as np
        from gymnax.environments import spaces

        self.env, self.params = gymnax.make(env_id)
        self.key, self.jax, self.np = jax.random.PRNGKey(seed), jax, np
        obs_space = self.env.observation_space(self.params)
        act_space = self.env.action_space(self.params)
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, obs_space.shape, np.float32
        )
        if isinstance(act_space, spaces.Discrete):
            self.action_space = gymnasium.spaces.Discrete(act_space.n)
        else:
            low, high = (
                np.broadcast_to(b, act_space.shape)
                for b in (act_space.low, act_space.high)
            )
            self.action_space = gymnasium.spaces.Box(
                low.astype(np.float32), high.astype(np.float32), dtype=np.float32
            )

    def _next_key(self):
        self.key, key = self.jax.random.split(self.key)
        return key

    def reset(self):
        obs, self.state = self.env.reset(self._next_key(), self.params)
        return self.np.asarray(obs)

    def step(self, action):
        _, self.state, reward, term, trunc, info = self.env.step(
            self._next_key(), self.state, action, self.params
        )
        done = bool(term) or bool(trunc)
        return self.np.asarray(info["final_observation"]), float(reward), done, {}


def evaluate(ac, env, episodes: int) -> float:
    """The mean return of ``episodes`` episodes of the deterministic policy:
    the most likely action, or the Gaussian's mean (as Ajax evaluates)."""
    import torch
    from gymnasium.spaces import Discrete

    returns = []
    for _ in range(episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                if isinstance(env.action_space, Discrete):
                    action = int(ac.pi.logits_net(x).argmax())
                else:
                    action = ac.pi.mu_net(x).numpy()
            obs, reward, done, _ = env.step(action)
            total += reward
        returns.append(total)
    return sum(returns) / len(returns)


def install_stand_ins(curve: list, eval_env, spec) -> None:
    """The modules vpg.py and core.py import, for one process; the logger
    evaluates after every ``spec.eval_every`` steps (vpg logs once an epoch,
    after its update)."""
    import gymnasium
    import numpy as np

    steps_per_epoch = spec.reference_config["steps_per_epoch"]

    class EpochLogger:
        def __init__(self, **_):
            self.epoch = 0

        def setup_pytorch_saver(self, ac):
            self.ac = ac

        def dump_tabular(self):
            self.epoch += 1
            if self.epoch * steps_per_epoch % spec.eval_every == 0:
                curve.append(evaluate(self.ac, eval_env, spec.n_eval_episodes))

        def __getattr__(self, name):  # save_config, log, store, save_state, ...
            return lambda *_, **__: None

    def statistics(x):  # mpi_statistics_scalar on one process: population std
        x = np.asarray(x, dtype=np.float32)
        return x.mean(), np.sqrt(((x - x.mean()) ** 2).mean())

    def module(name, **attributes):
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__dict__.update(attributes)

    nothing = lambda *_, **__: None  # noqa: E731
    module("gym", spaces=gymnasium.spaces)
    sys.modules["gym.spaces"] = gymnasium.spaces
    packages = ("spinup", "spinup.utils", "spinup.algos", "spinup.algos.pytorch")
    for name in (*packages, "spinup.algos.pytorch.vpg"):
        module(name)
    module("spinup.utils.logx", EpochLogger=EpochLogger)
    module(
        "spinup.utils.mpi_pytorch",
        setup_pytorch_for_mpi=nothing,
        sync_params=nothing,
        mpi_avg_grads=nothing,
    )
    module(
        "spinup.utils.mpi_tools",
        mpi_fork=nothing,
        mpi_avg=lambda x: x,
        proc_id=lambda: 0,
        num_procs=lambda: 1,
        mpi_statistics_scalar=statistics,
    )


def run(spec, seeds: list) -> list:
    import importlib.util

    import torch

    torch.set_num_threads(1)
    curves = []
    for seed in seeds:
        curve: list = []
        install_stand_ins(curve, OldGymEnv(spec.env_id, 10_000 + seed), spec)
        modules = {}
        for name, path in (
            ("spinup.algos.pytorch.vpg.core", "core.py"),
            ("vpg", "vpg.py"),
        ):
            loader = importlib.util.spec_from_file_location(name, VENDOR / path)
            modules[name] = importlib.util.module_from_spec(loader)
            sys.modules[name] = modules[name]
            loader.loader.exec_module(modules[name])
        kwargs = dict(spec.reference_config)
        steps = kwargs["steps_per_epoch"]
        kwargs["ac_kwargs"] = {"hidden_sizes": kwargs.pop("hidden_sizes")}
        modules["vpg"].vpg(
            lambda seed=seed: OldGymEnv(spec.env_id, seed),
            seed=seed,
            epochs=spec.n_timesteps // steps,
            save_freq=spec.n_timesteps,
            **kwargs,
        )
        curves.append(curve)
    return curves


def config(spec) -> dict:
    """vpg()'s resolved keywords: its defaults under the spec's settings."""
    return {
        "steps_per_epoch": 4000,
        "gamma": 0.99,
        "pi_lr": 3e-4,
        "vf_lr": 1e-3,
        "train_v_iters": 80,
        "lam": 0.97,
        "max_ep_len": 1000,
        "activation": "tanh",
        **spec.reference_config,
        "epochs": spec.n_timesteps // spec.reference_config["steps_per_epoch"],
    }


if __name__ == "__main__":
    from refcmp_worker import main

    main(
        lambda spec, _, seeds: run(spec, seeds),
        lambda spec, _: config(spec),
        ("torch", "numpy", "scipy", "gymnasium", "jax", "jaxlib", "gymnax"),
    )
