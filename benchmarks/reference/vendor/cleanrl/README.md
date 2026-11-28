# vwxyzjn/cleanrl (vendored, unmodified subset)

The six scripts in `cleanrl/` and `cleanrl_utils/buffers.py` (with the empty
`cleanrl_utils/__init__.py`), which the off-policy scripts import, are copied
unchanged from [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) at commit
`fe8d8a0` (2026-04-20). MIT License, see `LICENSE`.

`benchmarks/speed` runs them as their docs do, `python cleanrl/<script>.py
--env-id ... --total-timesteps ...`, in the uv project
`benchmarks/speed/envs/cleanrl`, to time CleanRL's SAC, TD3 (its JAX version),
PPO, DQN (its JAX version) and PQN.
