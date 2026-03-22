# ruff: noqa: C901
from typing import Callable

import jax
import jax.numpy as jnp
from gymnax.environments import environment

from ajax.environments.interaction import get_raw_obs
from ajax.wrappers import GymnaxWrapper


class AltitudePotentialWrapper(GymnaxWrapper):
    """
    Potential-based reward shaping for the approach phase.

    Adds F(s, s') = gamma * phi(s') - phi(s) at each step.
    Guaranteed not to change the optimal policy (Ng et al. 1999).

    phi(s) = -clip(|altitude - target| / max_distance, 0, 1)
      → 0 at target (maximum potential)
      → -1 at max_distance away (minimum potential)

    F(s, s') > 0 when moving toward target
    F(s, s') < 0 when moving away from target

    Gives the critic a dense signal during the approach phase,
    solving the long credit assignment horizon introduced by the box.
    The terminal cumulative expert reward is preserved unchanged.
    """

    def __init__(
        self,
        env: environment.Environment,
        gamma: float = 0.99,
        max_distance: float = 5000.0,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.max_distance = max_distance

    def potential(self, state: environment.EnvState) -> jnp.ndarray:
        distance = jnp.abs(state.target_altitude - state.z)
        return -jnp.clip(distance / self.max_distance, 0.0, 1.0)

    def step(
        self,
        key,
        state: environment.EnvState,
        action,
        params: environment.EnvParams = None,
    ):
        obs, new_state, reward, done, info = self._env.step(key, state, action, params)
        # Potential-based shaping: suppressed on terminal transitions
        # so phi(s') of a reset state doesn't corrupt the signal
        shaping = (1.0 - info["terminated"]) * self.gamma * self.potential(
            new_state
        ) - self.potential(state)
        return obs, new_state, reward + shaping, done, info


class EarlyTerminationWrapper(GymnaxWrapper):
    """Observation modifying wrapper"""

    def __init__(
        self,
        env: environment.Environment,
        trunc_condition: Callable[[environment.EnvState, environment.EnvParams], bool],
        expert_policy: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        """Set the observation transformation"""
        super().__init__(env)
        self.trunc_condition = trunc_condition
        self.expert_policy = expert_policy

    def is_terminal(
        self, state: environment.EnvState, params: environment.EnvParams
    ) -> jax.Array:
        """Check whether state is terminal."""
        # Check termination and construct updated state
        return self.trunc_condition(state, params) or self._env.is_terminal(
            state, params
        )

    def step(
        self,
        key,
        state: environment.EnvState,
        action,
        params: environment.EnvParams = None,
    ):
        """Step the env and return the transformed obs"""
        in_box = self.trunc_condition(state, params)
        expert_action = self.expert_policy(
            get_raw_obs(env_state=state, env=self._env, mode="gymnax")
        )  # TODO : handle mode
        obs, state, reward, done, info = self._env.step(
            key, state, (1 - in_box) * action + in_box * expert_action, params
        )
        # truncated = self.trunc_condition(state, params)

        info["terminated"] = jnp.logical_and(done, jnp.logical_not(info["truncated"]))

        # init_carry = (
        #     reward,  # rewards
        #     key,
        #     obs,
        #     done,  # done
        #     state,
        #     jnp.zeros(1),  # entropy_sum
        #     jnp.zeros(1),  # step_count
        #     jnp.zeros(1),  # step_count_2
        #     reward,  # last reward
        # )
        # init_carry = (
        #     jnp.expand_dims(reward, axis=0),  # rewards
        #     key,
        #     jnp.expand_dims(obs, axis=0),
        #     jnp.expand_dims(done, axis=0),  # done
        #     jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), state),
        #     jnp.zeros(1),  # entropy_sum
        #     jnp.zeros(1),  # step_count
        #     jnp.zeros(1),  # step_count_2
        #     jnp.expand_dims(reward, axis=-1),  # last reward
        # )
        # if truncated:
        # rewards_expert, *_ = jax.lax.while_loop(
        #     while_env_not_done,
        #     step_environment_expert("gymnax", self._env, params, self.expert_policy),
        #     init_carry,
        # )

        # reward += rewards_expert * truncated
        return (
            obs,
            state,
            reward,
            done,
            info,
        )  # -1 to make all rewards negative and minize the time to reach the box TODO: improve
