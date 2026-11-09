"""Gymnasium's v5 MuJoCo tasks, stepped by MJX.

Each task loads Gymnasium's own model file unchanged (assets/, Gymnasium 1.2.0,
MIT licence) and translates the step, observation, reward, termination and
reset of gymnasium/envs/mujoco/<task>_v5.py, with the same defaults and the
same keyword arguments. Like Gymnasium, a step sets the action as the
control, runs frame_skip physics steps and reads the result; the action
enters the costs as given, and MuJoCo clamps the control to the model's
range. Episodes end by termination here and by the time limit of Gymnasium's
registry (``episode_length``), which Ajax's wrappers apply.

The random draws follow Gymnasium's distributions, not its generator's
stream: a reset is a sample of the same initial-state distribution.
"""

from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from ajax.random_keys import derive_key

ASSETS = Path(__file__).parent / "assets"


class GymnasiumTask(mjx_env.MjxEnv):
    """What the tasks share: the model, Gymnasium's reset and step."""

    xml_file: str
    frame_skip: int
    episode_length: int = 1000

    def __init__(self, physics: str = "jax"):
        model = mujoco.MjModel.from_xml_path(str(ASSETS / self.xml_file))
        sim_dt = float(model.opt.timestep)
        config = config_dict.create(
            ctrl_dt=sim_dt * self.frame_skip,
            sim_dt=sim_dt,
            episode_length=self.episode_length,
            impl=physics,
        )
        super().__init__(config)
        self._xml_path = str(ASSETS / self.xml_file)
        self._mj_model = model
        self._mjx_model = mjx.put_model(model, impl=physics)
        # Gymnasium's init_qpos and init_qvel: the state mj_resetData restores.
        self.init_qpos = jnp.asarray(model.qpos0)
        self.init_qvel = jnp.zeros(model.nv)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def uniform(self, key: jax.Array, scale: float, size: int) -> jax.Array:
        return jax.random.uniform(key, (size,), minval=-scale, maxval=scale)

    def set_state(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Gymnasium's set_state: the state, then mj_forward."""
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, impl=self._mjx_model.impl.value
        )
        return mjx.forward(self._mjx_model, data)

    def initial_state(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The task's reset_model draw: (qpos, qvel)."""
        raise NotImplementedError

    def observe(self, data: mjx.Data) -> jax.Array:
        raise NotImplementedError

    def outcome(
        self, before: mjx.Data, after: mjx.Data, action: jax.Array
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """The step's reward, termination and reward terms."""
        raise NotImplementedError

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos, qvel = self.initial_state(rng)
        data = self.set_state(qpos, qvel)
        zeros = jax.eval_shape(self.outcome, data, data, jnp.zeros(self.action_size))[2]
        metrics = {name: jnp.zeros(()) for name in zeros}
        return mjx_env.State(
            data, self.observe(data), jnp.zeros(()), jnp.zeros(()), metrics, {}
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self._mjx_model, state.data, action, self.frame_skip)
        reward, terminated, terms = self.outcome(state.data, data, action)
        return state.replace(
            data=data,
            obs=self.observe(data),
            reward=reward,
            done=terminated.astype(reward.dtype),
            metrics=terms,
        )


class InvertedPendulum(GymnasiumTask):
    """inverted_pendulum_v5.py: 1 per step while the pole stays within 0.2
    rad, ending when it leaves or the state is not finite."""

    xml_file = "inverted_pendulum.xml"
    frame_skip = 2

    def __init__(self, reset_noise_scale: float = 0.01, physics: str = "jax"):
        self.reset_noise_scale = reset_noise_scale
        super().__init__(physics)

    def initial_state(self, key):
        qpos_key = derive_key(key, "reset_qpos")
        qvel_key = derive_key(key, "reset_qvel")
        scale = self.reset_noise_scale
        qpos = self.init_qpos + self.uniform(qpos_key, scale, self.mj_model.nq)
        qvel = self.init_qvel + self.uniform(qvel_key, scale, self.mj_model.nv)
        return qpos, qvel

    def observe(self, data):
        return jnp.concatenate([data.qpos, data.qvel])

    def outcome(self, before, after, action):  # noqa: ARG002 (shared signature)
        observation = self.observe(after)
        terminated = ~jnp.all(jnp.isfinite(observation)) | (
            jnp.abs(observation[1]) > 0.2
        )
        reward = jnp.where(terminated, 0.0, 1.0)
        return reward, terminated, {"reward_survive": reward}


class InvertedDoublePendulum(GymnasiumTask):
    """inverted_double_pendulum_v5.py: an alive bonus less the tip's
    distance from upright and the joints' speed, ending when the tip falls
    to a height of 1."""

    xml_file = "inverted_double_pendulum.xml"
    frame_skip = 5

    def __init__(
        self,
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.1,
        physics: str = "jax",
    ):
        self.healthy_reward = healthy_reward
        self.reset_noise_scale = reset_noise_scale
        super().__init__(physics)

    def initial_state(self, key):
        qpos_key = derive_key(key, "reset_qpos")
        qvel_key = derive_key(key, "reset_qvel")
        scale = self.reset_noise_scale
        qpos = self.init_qpos + self.uniform(qpos_key, scale, self.mj_model.nq)
        noise = jax.random.normal(qvel_key, (self.mj_model.nv,))
        return qpos, self.init_qvel + noise * scale

    def observe(self, data):
        return jnp.concatenate(
            [
                data.qpos[:1],
                jnp.sin(data.qpos[1:]),
                jnp.cos(data.qpos[1:]),
                jnp.clip(data.qvel, -10, 10),
                jnp.clip(data.qfrc_constraint, -10, 10)[:1],
            ]
        )

    def outcome(self, before, after, action):  # noqa: ARG002 (shared signature)
        x, _, y = after.site_xpos[0]
        terminated = y <= 1
        distance_penalty = 0.01 * x**2 + (y - 2) ** 2
        velocity_penalty = 1e-3 * after.qvel[1] ** 2 + 5e-3 * after.qvel[2] ** 2
        alive_bonus = jnp.where(terminated, 0.0, self.healthy_reward)
        reward = alive_bonus - distance_penalty - velocity_penalty
        terms = {
            "reward_survive": alive_bonus,
            "distance_penalty": -distance_penalty,
            "velocity_penalty": -velocity_penalty,
        }
        return reward, terminated, terms


class Reacher(GymnasiumTask):
    """reacher_v5.py: the fingertip's distance to a target, less the
    control's square; 50-step episodes, no termination."""

    xml_file = "reacher.xml"
    frame_skip = 2
    episode_length = 50

    def __init__(
        self,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 1.0,
        physics: str = "jax",
    ):
        self.reward_dist_weight = reward_dist_weight
        self.reward_control_weight = reward_control_weight
        super().__init__(physics)
        self.fingertip = self.mj_model.body("fingertip").id
        self.target = self.mj_model.body("target").id

    def initial_state(self, key):
        qpos_key = derive_key(key, "reset_qpos")
        qpos = self.init_qpos + self.uniform(qpos_key, 0.1, self.mj_model.nq)
        # Gymnasium redraws the goal uniformly in [-0.2, 0.2]^2 until it
        # lies inside the disc of radius 0.2: uniform in that disc, drawn
        # here by its radius and angle.
        radius_draw, angle_draw = jax.random.uniform(
            derive_key(key, "reset_goal"), (2,)
        )
        radius = 0.2 * jnp.sqrt(radius_draw)
        angle = 2 * jnp.pi * angle_draw
        goal = radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        qpos = qpos.at[-2:].set(goal)
        qvel_key = derive_key(key, "reset_qvel")
        qvel = self.init_qvel + self.uniform(qvel_key, 0.005, self.mj_model.nv)
        return qpos, qvel.at[-2:].set(0.0)

    def fingertip_to_target(self, data: mjx.Data) -> jax.Array:
        # Gymnasium's get_body_com reads the body frame's position, xpos.
        return data.xpos[self.fingertip] - data.xpos[self.target]

    def observe(self, data):
        theta = data.qpos[:2]
        return jnp.concatenate(
            [
                jnp.cos(theta),
                jnp.sin(theta),
                data.qpos[2:],
                data.qvel[:2],
                self.fingertip_to_target(data)[:2],
            ]
        )

    def outcome(self, before, after, action):  # noqa: ARG002 (shared signature)
        reward_dist = -jnp.linalg.norm(self.fingertip_to_target(after))
        reward_dist = reward_dist * self.reward_dist_weight
        reward_ctrl = -jnp.sum(jnp.square(action)) * self.reward_control_weight
        terms = {"reward_dist": reward_dist, "reward_ctrl": reward_ctrl}
        return reward_dist + reward_ctrl, jnp.array(False), terms


class Swimmer(GymnasiumTask):
    """swimmer_v5.py: the speed along x, less a small control cost; no
    termination."""

    xml_file = "swimmer.xml"
    frame_skip = 4

    def __init__(
        self,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        physics: str = "jax",
    ):
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.reset_noise_scale = reset_noise_scale
        self.exclude_positions = exclude_current_positions_from_observation
        super().__init__(physics)

    def initial_state(self, key):
        qpos_key = derive_key(key, "reset_qpos")
        qvel_key = derive_key(key, "reset_qvel")
        scale = self.reset_noise_scale
        qpos = self.init_qpos + self.uniform(qpos_key, scale, self.mj_model.nq)
        qvel = self.init_qvel + self.uniform(qvel_key, scale, self.mj_model.nv)
        return qpos, qvel

    def observe(self, data):
        position = data.qpos[2:] if self.exclude_positions else data.qpos
        return jnp.concatenate([position, data.qvel])

    def outcome(self, before, after, action):
        x_velocity = (after.qpos[0] - before.qpos[0]) / self.dt
        forward_reward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        terms = {"reward_forward": forward_reward, "reward_ctrl": -ctrl_cost}
        return forward_reward - ctrl_cost, jnp.array(False), terms


#: Each task by its Gymnasium id.
TASKS: dict[str, type[GymnasiumTask]] = {
    "InvertedPendulum-v5": InvertedPendulum,
    "InvertedDoublePendulum-v5": InvertedDoublePendulum,
    "Reacher-v5": Reacher,
    "Swimmer-v5": Swimmer,
}


def make(task_id: str, physics: str = "jax", **kwargs: Any) -> GymnasiumTask:
    """The task of a Gymnasium id, such as "Reacher-v5"; keywords are the
    Gymnasium task's own (``reset_noise_scale``, ...)."""
    task: Optional[type[GymnasiumTask]] = TASKS.get(task_id)
    if task is None:
        raise ValueError(f"No MuJoCo task {task_id!r}; known: {sorted(TASKS)}.")
    return task(physics=physics, **kwargs)
