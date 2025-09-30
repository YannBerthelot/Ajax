import itertools

import jax
import jax.numpy as jnp
import yaml
from flax import struct
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState
from tqdm import tqdm

from ajax.agents.PPO.PPO_pre_train import PPO

# from ajax.agents.PPO.PPO import PPO
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


def distance_to_stable_fn(state: PlaneState):
    z = state[..., 2]
    target = state[..., 6]
    z_dot = state[..., 3]
    return jnp.abs(z - target) + jnp.abs(z_dot - 0.0)


def get_sweep_values(
    baseline: bool = True,
    auto_imitation: bool = True,
    constant_imitation: bool = True,
    pre_train: bool = True,
):
    imitation_coef_list = []
    imitation_coef_offset_list = [0.0]
    if pre_train:
        pre_train_step_list = [int(1e5)]
    else:
        pre_train_step_list = [0]

    if baseline:
        imitation_coef_list += [0.0]

    if auto_imitation:
        imitation_coef_list += [
            "auto_100.0",  # type: ignore[list-item]
            "auto_10.0",  # type: ignore[list-item]
            "auto_1.0",  # type: ignore[list-item]
        ]
    if constant_imitation:
        imitation_coef_list += [
            1.0,
            1e-1,
            1e-2,
            1e-3,
        ]

    return {
        "imitation_coef": imitation_coef_list,
        "imitation_coef_offset": imitation_coef_offset_list,
        "pre_train_n_steps": pre_train_step_list,
    }


def get_distance_fn_from_imitation_coef(imitation_coef):
    if "auto" in str(imitation_coef):
        return distance_to_stable_fn


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


if __name__ == "__main__":
    project_name = "tests_plane_norm_2"
    n_timesteps = int(1e6)
    n_seeds = 10
    num_episode_test = 10
    log_frequency = 4096
    use_wandb = True
    logging_config = get_log_config(project_name)
    agent = PPO

    key = jax.random.PRNGKey(42)
    env = Plane()
    max_alt = 8_000
    min_alt = 3_000
    env_params = PlaneParams(
        target_altitude_range=(min_alt, max_alt),
        initial_altitude_range=(min_alt, max_alt),
        max_steps_in_episode=10_000,
    )

    expert_policy = get_expert_policy(env, env_params)
    policy_score = get_policy_score(expert_policy, env, env_params)

    if (
        jnp.isnan(expert_policy(jnp.array([0, 0, 0, 0, 0, 0, max_alt]))).any()
        or jnp.isnan(expert_policy(jnp.array([0, 0, 0, 0, 0, 0, min_alt]))).any()
    ):
        raise ValueError("No stable policy for {max_alt} or {min_alt}")
    print(f"Expert policy mean score: {policy_score}")

    sweep_values = get_sweep_values(
        baseline=True, auto_imitation=False, constant_imitation=False, pre_train=False
    )
    print(f"{sweep_values=}")
    env_id = "Plane"

    hyperparams = load_hyperparams(agent.name, env_id)
    mode = "CPU"
    for pre_train_n_steps, imitation_coef, imitation_coef_offset in tqdm(
        itertools.product(
            sweep_values["pre_train_n_steps"],
            sweep_values["imitation_coef"],
            sweep_values["imitation_coef_offset"],
        )
    ):
        distance_to_stable = get_distance_fn_from_imitation_coef(imitation_coef)
        _agent = agent(
            env_id=env,
            env_params=env_params,
            # expert_policy=expert_policy,
            # pre_train_n_steps=pre_train_n_steps,
            # imitation_coef=imitation_coef,
            # distance_to_stable=distance_to_stable,
            # imitation_coef_offset=imitation_coef_offset,
            **hyperparams,
        )
        if mode == "CPU":
            for seed in tqdm(range(n_seeds)):
                _agent.train(
                    seed=[seed],
                    logging_config=logging_config,
                    n_timesteps=n_timesteps,
                    num_episode_test=num_episode_test,
                )
                upload_tensorboard_to_wandb(
                    _agent.run_ids, logging_config, use_wandb=use_wandb
                )
        else:
            _agent.train(
                seed=list(range(n_seeds)),
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
            )
            upload_tensorboard_to_wandb(
                _agent.run_ids, logging_config, use_wandb=use_wandb
            )
