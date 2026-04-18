"""Shared composable modules usable by any agent.

Each submodule groups related callables that can be composed into an
agent's `make_train` at init time. None of these should depend on a
specific agent's state layout — they operate on the generic pieces
(critic_state, observations, actions, rng, etc.).
"""
