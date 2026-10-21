# xtma/apo (vendored, unmodified subset)

`apo/algos/apg/appo.py`, `apo/algos/apg/base.py`, `apo/algos/utils.py` and
the packages' `__init__.py` files are copied unchanged from
[xtma/apo](https://github.com/xtma/apo) at commit `bfcd997`, the code of the
APO paper's authors (Ma et al., *Average-Reward Reinforcement Learning with
Trust Region Methods*, IJCAI 2021). MIT License, see `LICENSE`. They run on
the authors' rlpyt fork (`../xtma_rlpyt`).

`../../conformance_apo.py` runs APPO's update in `envs/torch`;
`tests/conformance/test_apo_original.py` compares one update with Ajax's.
