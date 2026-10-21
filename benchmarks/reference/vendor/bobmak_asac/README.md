# BobMak/asac (vendored, unmodified)

`darer/*.py` and `hparams/InvertedPendulum-v5/asac.yaml` are copied unchanged
from [BobMak/asac](https://github.com/BobMak/asac) at commit `4b68862`
(2025-10-02), the code of the ASAC paper's authors (Adamczyk et al.,
*Average-Reward Soft Actor-Critic*, 2025). MIT License, see `LICENSE`.

`../../conformance_asac.py` runs `darer/ASAC.py`'s agent by its own `learn()`
loop in `envs/asac` (the authors' pinned versions) on gymnasium tasks;
`tests/conformance/test_asac_original.py` compares one of its updates with
Ajax's.
