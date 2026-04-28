from typing import Optional

import jax
import jax.numpy as jnp
import yaml
from flax import struct
from target_gym import Plane, PlaneParams
from target_gym.plane.env import PlaneState

from ajax.logging.wandb_logging import LoggingConfig
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


def get_log_config(
    project_name: str,
    agent_name: str,
    use_wandb: bool,
    log_frequency: int,
    sweep: bool = False,
    group_name: Optional[str] = None,
    **kwargs,
) -> LoggingConfig:
    """
    Build a LoggingConfig. All extra kwargs are stored in the W&B run config,
    so every hyperparameter passed here is visible in the W&B sweep table.
    """
    return LoggingConfig(
        project_name=project_name,
        run_name=agent_name,
        group_name=group_name,
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


def get_policy_score(policy, env: Plane, env_params: PlaneParams) -> float:
    """Run policy for 1000 episodes and return mean episodic return."""
    key = jax.random.PRNGKey(0)

    has_state = hasattr(policy, "init_state")

    def run_episode(key):
        obs, state = env.reset(key, env_params)
        expert_state = policy.init_state(1) if has_state else None

        def step_fn(carry, _):
            obs, state, expert_state = carry
            if has_state:
                action, expert_state = policy(expert_state, obs[None])
                action = action[0]
            else:
                action = policy(obs)
            obs, state, reward, done, info = env.step(key, state, action, env_params)
            return (obs, state, expert_state), (reward, done)

        _, (rewards, dones) = jax.lax.scan(
            f=step_fn,
            init=(obs, state, expert_state),
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


def strip_str_seq_to_seq_of_str(seq: str):
    return [x.strip(" ") for x in seq.lstrip("(").rstrip(")").split(",")]


def process_hyperparams(hpp: dict) -> dict:
    if "actor_architecture" in hpp:
        hpp["actor_architecture"] = strip_str_seq_to_seq_of_str(
            hpp["actor_architecture"]
        )
    if "critic_architecture" in hpp:
        hpp["critic_architecture"] = strip_str_seq_to_seq_of_str(
            hpp["critic_architecture"]
        )
    if "normalize" in hpp:
        normalize = hpp.pop("normalize")
        hpp["normalize_observations"] = normalize
        hpp["normalize_rewards"] = normalize
    return hpp


def load_hyperparams(agent: str = "PPO", env_id: str = "Plane") -> dict:
    from pathlib import Path
    _ajax_root = Path(__file__).parents[3]
    file_name = _ajax_root / "hyperparams" / f"ajax_{agent.lower()}.yml"
    with open(file_name) as stream:
        try:
            hyperparams_data = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
        return process_hyperparams(hyperparams_data[env_id])


def _resolve_schedule_factor(
    state,
    schedule: Optional[str],
    train_frac: Optional[float],
) -> float:
    if schedule is None:
        return 1.0
    if train_frac is not None:
        if schedule == "linear":
            return linear_schedule(train_frac)
        if schedule == "exponential":
            return exponential_schedule(train_frac)
        if schedule == "polynomial":
            return polynomial_schedule(train_frac)
    else:
        if schedule == "linear":
            return state.linear_schedule
        if schedule == "exponential":
            return state.exponential_schedule
        if schedule == "polynomial":
            return state.polynomial_schedule
    return 1.0
