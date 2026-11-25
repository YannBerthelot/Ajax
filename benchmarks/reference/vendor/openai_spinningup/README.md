# openai/spinningup (vendored, unmodified)

`vpg.py` and `core.py` are copied unchanged from
[openai/spinningup](https://github.com/openai/spinningup) at commit
`038665d62d569055401d91856abb287263096178`
(`spinup/algos/pytorch/vpg/`): Spinning Up's Vanilla Policy Gradient,
REINFORCE (Williams, 1992) with a learned baseline and GAE. MIT License, see
`LICENSE`.

`../../refcmp_spinup.py` runs `vpg()` as it is, on a gymnax task through the
old gym API it steps, with single-process stand-ins for the `spinup` modules
it imports (its MPI helpers and its logger); the stand-in logger evaluates
the policy for the reference comparison of Ajax's REINFORCE.
