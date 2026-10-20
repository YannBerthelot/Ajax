"""REDQ's critics read actions in the env's units, as watchernyu/REDQ's
(7b5d1bf) read act_limit * tanh: Ajax's agents act in [-1, 1], and a critic
trained on those takes Adam steps of another size on its action inputs."""

from __future__ import annotations

from ajax.agents.REDQ.REDQ import REDQ

NET = ("8", "relu")


def test_redq_sets_its_critics_units():
    def units(env_id: str):
        state, _ = REDQ(
            env_id,
            actor_architecture=NET,
            critic_architecture=NET,
            learning_starts=4,
            num_critic_updates=2,
        ).train(seed=[0], n_timesteps=8)
        return state.critic_state.action_units

    assert units("Pendulum-v1") == ((0.0,), (2.0,))  # torque in [-2, 2]
    assert units("MountainCarContinuous-v0") is None  # already [-1, 1]
