Scratch only (branch wip/nested-jit-scratch, no PR). Base: 96bf03d, the tip of
PR #288. The patch makes `NESTED_JIT=family` wrap each family's update in its
own `jax.jit`, and `NESTED_JIT=step` wrap DQN's single gradient step (the
inner jit #288 removed); unset or `off` is #288 as it is.
