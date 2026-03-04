from typing import Optional

import jax
import jax.numpy as jnp
import yaml
from flax import struct
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState

from ajax.logging.wandb_logging import (
    LoggingConfig,
)
from ajax.state import exponential_schedule, linear_schedule, polynomial_schedule


@struct.dataclass
class StableState:
    z: float
    z_dot: float
    theta_dot: float


def distance_to_stable_fn(state: PlaneState):
    z = state[..., 1]
    target = state[..., 6]
    z_dot = state[..., 2]
    theta_dot = state[..., 4]
    return jnp.abs(z - target) + jnp.abs(z_dot - 0.0) + jnp.abs(theta_dot - 0.0)


def get_sweep_values(
    baseline: bool = True,
    auto_imitation: bool = True,
    constant_imitation: bool = True,
    pre_train: bool = True,
):
    imitation_coef_list = []
    imitation_coef_offset_list = [0.0]
    if baseline:
        imitation_coef_list += [None]

    if pre_train:
        pre_train_step_list = [0, int(1e5)]
    else:
        pre_train_step_list = [0]

    if constant_imitation:
        imitation_coef_list += [100.0, 10.0, 1.0, 0.0]  # type: ignore[list-item]
        # imitation_coef_list += [0.0]

    if auto_imitation:
        imitation_coef_list += [
            "auto_10000.0",  # type: ignore[list-item]
            # "auto_1000.0",  # type: ignore[list-item]
            # "auto_100.0",  # type: ignore[list-item]
            # "auto_10.0",  # type: ignore[list-item]
            # "auto_1.0",  # type: ignore[list-item]
            # "auto_0.1",  # type: ignore[list-item]
            # "auto_0.01",  # type: ignore[list-item]
            # "auto_0.001",  # type: ignore[list-item]
            # "auto_0.0001",  # type: ignore[list-item]
        ]

    return {
        "imitation_coef": imitation_coef_list,
        "imitation_coef_offset": imitation_coef_offset_list,
        "pre_train_n_steps": pre_train_step_list,
    }


def get_distance_fn_from_imitation_coef(imitation_coef):
    if "auto" in str(imitation_coef):
        return distance_to_stable_fn


def get_log_config(
    project_name,
    agent_name,
    use_wandb,
    log_frequency,
    sweep: bool = False,
    **kwargs,
):
    return LoggingConfig(
        project_name=project_name,
        run_name=agent_name,
        config={
            "debug": False,
            "log_frequency": log_frequency,
            **kwargs,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=True,
        use_wandb=use_wandb,
        sweep=sweep,
    )


def strip_str_seq_to_seq_of_str(seq: str):
    return [x.strip(" ") for x in seq.lstrip("(").rstrip(")").split(",")]


def process_hyperparams(hpp: dict):
    if "actor_architecture" in hpp.keys():
        hpp["actor_architecture"] = strip_str_seq_to_seq_of_str(
            hpp["actor_architecture"]
        )
    if "critic_architecture" in hpp.keys():
        hpp["critic_architecture"] = strip_str_seq_to_seq_of_str(
            hpp["critic_architecture"]
        )
    if "normalize" in hpp.keys():
        normalize = hpp.pop("normalize")
        hpp["normalize_observations"] = normalize
        hpp["normalize_rewards"] = normalize
    return hpp


def load_hyperparams(agent: str = "PPO", env_id: str = "Plane"):
    file_name = f"hyperparams/ajax_{agent.lower()}.yml"
    with open(file_name) as stream:
        try:
            hyperparams_data = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
        return process_hyperparams(hyperparams_data[env_id])


def get_policy_score(policy, env: Plane, env_params: PlaneParams):
    key = jax.random.PRNGKey(0)

    def run_episode(key):
        obs, state = env.reset(key, env_params)

        def step_fn(carry, _):
            obs, state = carry
            action = policy(obs)
            obs, state, reward, done, info = env.step(key, state, action, env_params)
            return (obs, state), (reward, done)

        _, (rewards, dones) = jax.lax.scan(
            f=step_fn,
            init=(obs, state),
            xs=None,
            length=env_params.max_steps_in_episode,
        )
        rewards_before_done = (rewards * (1 - dones)).sum()
        rewards_with_last_step = rewards_before_done + rewards[jnp.argmax(dones)]
        return rewards_with_last_step

    keys = jax.random.split(key, 1000)
    returns = jax.vmap(run_episode, in_axes=[0])(keys)
    return returns.mean()


def get_mode() -> str:
    return "GPU" if jax.default_backend() == "gpu" else "CPU"


def _resolve_schedule_factor(
    state,
    schedule: Optional[str],
    train_frac: Optional[float],
) -> float:
    if schedule is None:
        return 1.0

    if train_frac is not None:
        if schedule == "linear":
            val = linear_schedule(train_frac)
        if schedule == "exponential":
            val = exponential_schedule(train_frac)
        if schedule == "polynomial":
            val = polynomial_schedule(train_frac)
    else:
        if schedule == "linear":
            val = state.linear_schedule
        if schedule == "exponential":
            val = state.exponential_schedule
        if schedule == "polynomial":
            val = state.polynomial_schedule

    return val

    # raise ValueError(f"Unrecognized schedule {schedule}")
