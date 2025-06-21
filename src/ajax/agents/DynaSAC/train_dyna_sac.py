import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import jax
from jax.tree_util import Partial as partial

from ajax.agents.AVG.train_AVG import init_AVG
from ajax.agents.AVG.train_AVG import training_iteration as secondary_training_iteration
from ajax.agents.DynaSAC.state import AVGState, DynaSACConfig, SACState
from ajax.agents.sac.train_sac import init_sac
from ajax.agents.sac.train_sac import training_iteration as primary_training_iteration
from ajax.environments.utils import check_env_is_gymnax, get_state_action_shapes
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType

PROFILER_PATH = "./tensorboard"


def profile_memory(timestep):
    jax.profiler.save_device_memory_profile(f"memory{timestep}.prof")


def safe_get_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve an environment variable.

    Args:
        var_name (str): The name of the environment variable.
        default (Optional[str]): Default value if the variable is not set.

    Returns:
        Optional[str]: The value of the environment variable or default.
    """
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value


def get_agent_state_from_agent_state(
    target: Union[SACState, AVGState], source: Union[SACState, AVGState]
) -> Union[SACState, AVGState]:
    target_actor_state = target.actor_state.replace(params=source.actor_state.params)  # type: ignore[union-attr]
    target_critic_state = target.critic_state.replace(params=source.critic_state.params)  # type: ignore[union-attr]
    return target.replace(  # type: ignore[union-attr]
        actor_state=target_actor_state, critic_state=target_critic_state
    )


def make_train(
    env_args: EnvironmentConfig,
    primary_actor_optimizer_args: OptimizerConfig,
    primary_critic_optimizer_args: OptimizerConfig,
    secondary_actor_optimizer_args: OptimizerConfig,
    secondary_critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: DynaSACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    sac_length: int = 1,
    avg_length: int = 1,
):
    """
    Create the training function for the SAC agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_config (SACConfig): SAC agent configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        total_timesteps (int): Total timesteps for training.
        num_episode_test (int): Number of episodes for evaluation during training.

    Returns:
        Callable: JIT-compiled training function.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""
        primary_agent_state = init_sac(
            key=key,
            env_args=env_args,
            actor_optimizer_args=primary_actor_optimizer_args,
            critic_optimizer_args=primary_critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        secondary_agent_state = init_AVG(
            key=key,
            env_args=env_args,
            actor_optimizer_args=secondary_actor_optimizer_args,
            critic_optimizer_args=secondary_critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            num_critics=2,
        )

        # agent_state = DynaSACState(
        #     primary=primary_agent_state, secondary=secondary_agent_state
        # )

        num_updates = total_timesteps // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env, env_args.env_params)

        primary_training_iteration_scan_fn = partial(
            primary_training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config.primary,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
        )

        secondary_training_iteration_scan_fn = partial(
            secondary_training_iteration,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config.secondary,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
        )
        # unroll_length = 4  # IMPORTANT: has to match between loops for reproducibility, otherwise a N x 1 loop might not yield the same results as a N loop. has to be >1 as well for some reason to be reproducible
        inner_length = sac_length + avg_length
        print(sac_length, avg_length)

        def dyna_train_loop(
            carry: tuple[SACState, AVGState], _: Any
        ) -> tuple[tuple[SACState, AVGState], None]:
            primary_agent_state, secondary_agent_state = carry
            new_primary_agent_state, _ = jax.lax.scan(
                f=primary_training_iteration_scan_fn,
                init=primary_agent_state,
                xs=None,
                length=sac_length,
                unroll=1,
            )

            transfered_secondary_agent_state = get_agent_state_from_agent_state(
                source=new_primary_agent_state, target=secondary_agent_state
            )
            new_secondary_agent_state, _ = jax.lax.scan(
                f=secondary_training_iteration_scan_fn,
                init=transfered_secondary_agent_state,
                xs=None,
                length=avg_length,
                unroll=1,
            )
            transfered_primary_agent_state = get_agent_state_from_agent_state(
                target=new_primary_agent_state, source=new_secondary_agent_state
            )
            assert isinstance(
                transfered_primary_agent_state, SACState
            ), "transfered_primary_agent_state is not a SACState"  # to make mypy happy
            return (
                transfered_primary_agent_state,
                new_secondary_agent_state,
            ), None

        (primary_agent_state, secondary_agent_state), _ = jax.lax.scan(
            f=dyna_train_loop,
            init=(primary_agent_state, secondary_agent_state),
            xs=None,
            length=int(num_updates // inner_length),
        )
        return primary_agent_state

    return train
