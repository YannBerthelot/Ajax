# twni2016/pomdp-baselines (vendored, unmodified)

The files under `policies/`, `torchkit/` and `utils/helpers.py` are copied
unchanged from
[twni2016/pomdp-baselines](https://github.com/twni2016/pomdp-baselines) at
commit `e7c19c32a20033d75414b29fbc466c77c211e968`, the code of Ni,
Eysenbach and Salakhutdinov, *Recurrent Model-Free RL Can Be a Strong
Baseline for Many POMDPs* (ICML 2022). MIT License, see `LICENSE`.

Only what `policies/models/policy_rnn.py`'s `ModelFreeOffPolicy_Separate_RNN`
imports is kept: `policies/models/` (its package imports every agent class),
`policies/rl/`, the `torchkit` modules they import, and `utils/helpers.py`.
`utils/logger.py` is not: the update never logs, and the module no longer
imports on Python 3.10 and later (`from collections import Set`).

`../../conformance_ni_recurrent.py` builds the agent in `envs/torch` and
records one update of its recurrent SAC and TD3;
`tests/conformance/test_ni_recurrent_original.py` compares it with Ajax's.
