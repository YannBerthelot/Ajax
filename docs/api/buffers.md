# Buffers

Off-policy agents store transitions in a flashbax replay buffer. With
`memory=`, they store per-environment trajectories and replay sequences
R2D2-style. The world models keep their own replay
([DreamerV3](../algorithms/dreamerv3.md), [TD-MPC2](../algorithms/tdmpc2.md)).

::: ajax.buffers.utils

::: ajax.buffers.dataset

::: ajax.agents.recurrent
