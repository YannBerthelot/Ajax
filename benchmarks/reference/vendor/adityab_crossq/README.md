# The authors' CrossQ (vendored, unmodified)

The `sbx/` files are copied unchanged from
[adityab/CrossQ](https://github.com/adityab/CrossQ) at commit
`a318a266679bec3f9d93c44e029860a6da077c89` (2024-04-08). That is the code of the CrossQ paper's
authors (Bhatt et al., *CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater
Sample Efficiency and Simplicity*, ICLR 2024), a fork of SBX. They are MIT-licensed; see `LICENSE`
and `NOTICE`.

Only the modules its SAC imports are kept: `sbx/__init__.py`, `version.txt`, `sbx/common/` (minus
the on-policy algorithm) and `sbx/sac/` (minus the evaluation callback). CrossQ is a named
hyperparameter set of this SAC (`train.py -algo crossq`), which
`../../conformance_crossq.py` builds as `train.py` does, in the environment
`../../envs/crossq` pins: the versions of the original's `setup.py`.
