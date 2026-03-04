import os
from functools import partial
from typing import Optional

import dill as pickle
import jax
import jax.numpy as jnp
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState
from tqdm import tqdm

from ajax import SAC
from ajax.early_termination_wrapper import EarlyTerminationWrapper
from ajax.logging.wandb_logging import (
    upload_tensorboard_to_wandb,
)
from ajax.plane.plane_exps_utils import (
    _resolve_schedule_factor,
    get_log_config,
    get_mode,
    get_policy_score,
    get_sweep_values,
    load_hyperparams,
)
from ajax.stable_utils import get_expert_policy

if __name__ == "__main__":
    mode = get_mode()
    mode = "GPU"
    agent = SAC
    project_name = f"tests_{agent.name}_plane_early_trunc_tests_debug"
    n_timesteps = int(1e6)
    n_seeds = 1
    num_episode_test = 25
    log_frequency = 5_000
    action_scale = 1.0
    use_wandb = True
    schedule = "polynomial"
    sweep_mode = False  # True is no logging until the very end (faster) false is logging during training (slower)

    key = jax.random.PRNGKey(42)
    env = Plane()

    def trunc_condition(
        state: PlaneState,
        params: PlaneParams,
        schedule: Optional[str] = None,
        train_frac: Optional[float] = None,
    ):
        factor = _resolve_schedule_factor(state, schedule, train_frac)

        altitude_ok = abs(state.target_altitude - state.z) < (500 * factor)
        # velocity_ok = abs(state.z_dot) < 50 * factor
        # rotation_ok = abs(state.theta_dot) < 1.0 * factor

        return altitude_ok  # & velocity_ok & rotation_ok

    max_alt = 8_000
    min_alt = 3_000
    env_params = PlaneParams(
        target_altitude_range=(min_alt, max_alt),
        initial_altitude_range=(min_alt, max_alt),
        max_steps_in_episode=10_000,
    )

    if "expert_policy.pkl" in os.listdir():
        with open("expert_policy.pkl", "rb") as f:
            expert_policy = pickle.load(f)  # deserialize using load()
    else:
        expert_policy = get_expert_policy(env, env_params)

        with open("expert_policy.pkl", "wb") as f:  # open a text file
            pickle.dump(expert_policy, f)  # serialize the list

    policy_score = get_policy_score(expert_policy, env, env_params)

    if (
        jnp.isnan(expert_policy(jnp.array([0, 0, 0, 0, 0, 0, max_alt]))).any()
        or jnp.isnan(expert_policy(jnp.array([0, 0, 0, 0, 0, 0, min_alt]))).any()
    ):
        raise ValueError("No stable policy for {max_alt} or {min_alt}")
    print(f"Expert policy mean score: {policy_score}")
    #########

    # Config

    sweep_values = get_sweep_values(
        baseline=False, auto_imitation=True, constant_imitation=False, pre_train=False
    )

    #######

    print(f"{sweep_values=}")
    env_id = "Plane"

    hyperparams = load_hyperparams(agent.name, env_id)
    residual = True
    for schedule in (None, "linear", "exponential", "polynomial"):
        fixed_alpha = True
        logging_config = get_log_config(
            project_name=project_name,
            agent_name=agent.name,
            log_frequency=log_frequency,
            n_seeds=n_seeds,
            use_wandb=use_wandb,
            sweep=sweep_mode,
            schedule=schedule,
            residual=residual,
            fixed_alpha=fixed_alpha,
        )

        trunc_condition = partial(trunc_condition, schedule=schedule)
        _agent = agent(
            env_id=EarlyTerminationWrapper(
                env, trunc_condition=trunc_condition, expert_policy=expert_policy
            )
            if trunc_condition is not None
            else env,
            env_params=env_params,
            expert_policy=expert_policy,
            **hyperparams,
            action_scale=1.0,
            early_termination_condition=trunc_condition,
            residual=residual,
            fixed_alpha=fixed_alpha,
        )
        if mode == "CPU":
            for seed in tqdm(range(n_seeds)):
                _agent.train(
                    seed=[seed],
                    logging_config=logging_config,
                    n_timesteps=n_timesteps,
                    num_episode_test=num_episode_test,
                )
                if sweep_mode:
                    upload_tensorboard_to_wandb(_agent.run_ids, logging_config)
        else:
            _agent.train(
                seed=list(range(n_seeds)),
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=num_episode_test,
            )
            if sweep_mode:
                upload_tensorboard_to_wandb(_agent.run_ids, logging_config)
