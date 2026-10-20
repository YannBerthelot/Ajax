# sfujim/TD3_BC (vendored, unmodified)

`TD3_BC.py`, `utils.py` and `main.py` are copied unchanged from
[sfujim/TD3_BC](https://github.com/sfujim/TD3_BC) at commit
`8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f` (2021-12-17), the code of the
TD3+BC paper's first author (Fujimoto and Gu, *A Minimalist Approach to
Offline Reinforcement Learning*, NeurIPS 2021). MIT License, see `LICENSE`.

`../../conformance_sfujim_td3_bc.py` imports `TD3_BC.py` and `utils.py` as
they are and drives them with `main.py`'s offline loop, written out there,
on a dataset collected from a gymnax task;
`tests/conformance/test_td3bc_sfujim.py` compares one of their updates with
Ajax's.
