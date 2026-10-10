"""Shared fixtures for agent test suites.

This file is for **wire tests** -- the lightweight code-tier checks that
verify the active training path actually does what the configuration
says. The motivating bug: in May 2026 we discovered that PPO's
``do_update.body_fn`` did not iterate over minibatches (it took the
whole ``(num_minibatches, mb_size, feat)`` shuffled batch as one
full-batch update), so ``num_minibatches=32`` silently became
``num_minibatches=1`` -- a 32x reduction in SGD step count vs brax PPO.
The unit tests on CartPole still passed because even broken-but-
functional PPO solves CartPole. The bug was invisible until we tried
mujoco_playground manip envs and saw all 50 seeds collapse to the
same low-reward floor.

Tests should be split into three tiers per agent (per
``tests/agents/<Agent>/``):

* ``test_<agent>_code.py``     -- programmatic invariants (function
  call counts, config values reach consumers, output types, error
  paths).

* ``test_<agent>_structure.py`` -- network architecture, param tree
  shape, minibatch dimensions, wrapper-stack composition. JIT-safe.
  Use ``chex.assert_shape`` / ``chex.assert_tree_shape_prefix``.

* ``test_<agent>_rl.py``        -- algorithm correctness on synthetic
  data: GAE on a known trajectory, loss signs, bootstrap conventions
  (terminated vs truncated), entropy formulas. JIT-safe. Use hand-
  computed expected values and ``chex.assert_trees_all_close``.

The existing ``test_<agent>.py`` files are integration smoke tests
(train returns without error on toy envs); keep those but treat them
as the weakest layer.
"""
