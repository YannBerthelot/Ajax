# sfujim/TD3 (vendored, unmodified)

`TD3.py`, `utils.py` and `main.py` are copied unchanged from
[sfujim/TD3](https://github.com/sfujim/TD3) at commit
`34770ccdcb51df6f6c7d85c1ba7e40d71b940d16` (2023-07-14), the code of the TD3
paper's first author (Fujimoto, van Hoof and Meger, *Addressing Function
Approximation Error in Actor-Critic Methods*, 2018). MIT License, see
`LICENSE`.

`../../conformance_sfujim_td3.py` imports `TD3.py` and `utils.py` as they are
and drives them with `main.py`'s loop, written out there, on a gymnax task;
`tests/conformance/test_td3_sfujim.py` compares one of their updates with
Ajax's.
