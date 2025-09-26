import itertools
from functools import partial

import jax
import jax.numpy as jnp
import yaml
from flax import struct
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState
from tqdm import tqdm

from ajax.agents.PPO.PPO_pre_train import PPO
from ajax.logging.wandb_logging import (
    LoggingConfig,
    upload_tensorboard_to_wandb,
)
from ajax.stable_utils import get_expert_policy


@struct.dataclass
class StableState:
    z: float
    z_dot: float
    theta_dot: float


def distance_to_stable_fn(state: PlaneState, stable_state: StableState):
    z = state[..., 2]
    z_dot = state[..., 3]
    return jnp.abs(z - stable_state.z) + jnp.abs(z_dot - stable_state.z_dot)


def get_sweep_values(
    baseline: bool = True,
    auto_imitation: bool = True,
    constant_imitation: bool = True,
    pre_train: bool = True,
):
    imitation_coef_list = []
    imitation_coef_offset_list = []
    pre_train_step_list = []

    if baseline:
        imitation_coef_list = [0.0]
        imitation_coef_offset_list = [0.0]
        pre_train_step_list = [0]

    if auto_imitation:
        imitation_coef_list += [
            "auto_10.0",  # type: ignore[list-item]
            "auto_1.0",  # type: ignore[list-item]
            "auto_0.1",  # type: ignore[list-item]
        ]
    if constant_imitation:
        imitation_coef_list += [
            1.0,
            1e-1,
            1e-2,
            1e-3,
        ]
    if pre_train:
        pre_train_step_list += [int(1e5)]

    return {
        "imitation_coef": imitation_coef_list,
        "imitation_coef_offset": imitation_coef_offset_list,
        "pre_train_n_steps": pre_train_step_list,
    }


def get_distance_fn_from_imitation_coef(imitation_coef):
    if "auto" in str(imitation_coef):
        return partial(
            distance_to_stable_fn,
            stable_state=StableState(z=target_altitude, z_dot=0, theta_dot=0),
        )


def get_log_config(project_name):
    return LoggingConfig(
        project_name=project_name,
        run_name="PPO",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=True,
        use_wandb=use_wandb,
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


if __name__ == "__main__":
    project_name = "Plane_sweep_3"
    n_timesteps = int(1e6)
    n_seeds = 20
    log_frequency = 5000
    use_wandb = True
    target_altitude = 5000  # meters
    logging_config = get_log_config(project_name)
    agent = PPO

    key = jax.random.PRNGKey(42)
    env = Plane()
    env_params = PlaneParams(
        target_altitude_range=(target_altitude, target_altitude),
    )

    expert_policy = get_expert_policy(target_altitude, Plane, env_params)

    sweep_values = get_sweep_values(
        baseline=True, auto_imitation=False, constant_imitation=False, pre_train=False
    )
    env_id = "Plane"

    hyperparams = load_hyperparams("PPO", env_id)

    for pre_train_n_steps, imitation_coef, imitation_coef_offset in tqdm(
        itertools.product(
            sweep_values["pre_train_n_steps"],
            sweep_values["imitation_coef"],
            sweep_values["imitation_coef_offset"],
        )
    ):
        distance_to_stable = get_distance_fn_from_imitation_coef(imitation_coef)
        PPO_agent = PPO(
            env_id=env,
            env_params=env_params,
            expert_policy=expert_policy,
            pre_train_n_steps=pre_train_n_steps,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
            **hyperparams,
        )
        PPO_agent.train(
            seed=list(range(n_seeds)),
            logging_config=logging_config,
            n_timesteps=n_timesteps,
        )
        upload_tensorboard_to_wandb(
            PPO_agent.run_ids, logging_config, use_wandb=use_wandb
        )
