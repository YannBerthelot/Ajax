# openai/baselines (vendored, unmodified subset)

The files under `baselines/` are copied unchanged from
[openai/baselines](https://github.com/openai/baselines) at commit `ea25b9e`
(2020-01-31), the reference code of Ajax's PPO: `ppo2/` (`learn`, its `Model`
and `Runner`, its `defaults`) and the modules they import, found by importing
them (`common/` with its policies, networks, distributions, Atari wrappers and
vector envs; `a2c/utils.py`; `bench/`; `logger.py`). MIT License, see `LICENSE`.

`../../conformance_ppo2.py` runs `ppo2.learn` in TensorFlow 1.15
(`../../envs/tf115`); `tests/conformance/test_ppo2_original.py` compares one
update with Ajax's.
