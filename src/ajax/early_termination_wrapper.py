# ruff: noqa: C901
from typing import Callable

import jax
import jax.numpy as jnp
from gymnax.environments import environment

from ajax.environments.interaction import get_raw_obs
from ajax.wrappers import GymnaxWrapper


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
