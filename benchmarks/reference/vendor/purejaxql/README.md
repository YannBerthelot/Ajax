# purejaxql's PQN (vendored, unmodified)

`pqn_gymnax.py` is copied unchanged from
[mttga/purejaxql](https://github.com/mttga/purejaxql) at commit
`47af6d7b35c89ddfe633aaf7341bdb8964cb7cce` (2025-11-12), the code of the PQN
paper's authors (Gallici et al., *Simplifying Deep Temporal Difference
Learning*, 2024). Apache License 2.0, see `LICENSE`.

`../../refcmp_jax.py` runs its `make_train` with the paper's CartPole config
(`purejaxql/config/alg/pqn_cartpole.yaml`, copied into `refcmp_specs.py`).
Two things adapt it to this environment without editing it: stubs for the
command-line imports it does not use here (hydra, omegaconf, wandb,
safetensors), and a wrapper that turns gymnax 1.0's
`terminated, truncated` into the single `done` gymnax 0.0.x returned.
SHA-256 of `pqn_gymnax.py`: `c946e7d10f7339d26ded386706a9032c200f51ac1b36c00bd9fac71f958b5b49`.
