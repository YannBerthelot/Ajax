import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.agents.AVG.train_AVG import init_AVG
from ajax.agents.AVG.train_AVG import training_iteration as secondary_training_iteration
from ajax.agents.DynaSAC.state import AVGState, DynaSACConfig, SACState
from ajax.agents.sac.train_sac import init_sac
from ajax.agents.sac.train_sac import training_iteration as primary_training_iteration
from ajax.distillation import policy_distillation, value_distillation
from ajax.environments.utils import check_env_is_gymnax, get_state_action_shapes
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.state import (
    AlphaConfig,
    # DoubleTrainState,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    get_double_train_state,
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


def repeat_first_entry(tree, num_repeats: int):
    return jax.tree.map(lambda x: jnp.tile(x, reps=num_repeats), tree)


def tile_to_batch(state, new_batch_size=10):
    return jax.tree_util.tree_map(
        lambda x: (
            jnp.tile(x, (new_batch_size,) + (1,) * (x.ndim - 1)).reshape(
                new_batch_size, -1
            )
            if isinstance(x, jnp.ndarray)
            else x
        ),
        state,
    )


def print_shape_diffs(source_tree, target_tree):
    def diff(src, tgt):
        src_shape, tgt_shape = jnp.shape(src), jnp.shape(tgt)
        if src_shape != tgt_shape:
            print(f"Shape mismatch: source {src_shape} vs target {tgt_shape}")
        return None

    jax.tree_util.tree_map(diff, source_tree, target_tree)


def broadcast_to_match(source_tree, target_tree):
    def maybe_broadcast(src, tgt):
        # Skip broadcasting for None or scalar values
        if src is None:
            return None
        if isinstance(src, (int, float)) and jnp.isscalar(tgt):
            return src

        # Check if shape matches, or if it needs to be repeated along axis 0
        src_shape, tgt_shape = jnp.shape(src), jnp.shape(tgt)
        if src_shape == tgt_shape:
            return src
        elif (
            len(src_shape) == len(tgt_shape) and src_shape[0] == 1 and tgt_shape[0] > 1
        ):
            reps = [tgt_shape[0]] + [1] * (len(tgt_shape) - 1)
            return jnp.tile(src, reps)
        elif src_shape == ():  # scalar to broadcast
            return jnp.broadcast_to(src, tgt_shape)
        else:
            raise ValueError(f"Cannot broadcast {src_shape} to {tgt_shape}")

    return jax.tree_util.tree_map(maybe_map, maybe_broadcast, source_tree, target_tree)


def maybe_map(fn, source_tree, target_tree):
    return fn(source_tree, target_tree) if source_tree is not None else None


def distill_source_to_target(
    source, target, teacher_inputs, student_inputs, num_epochs=1, actor: bool = False
):
    teacher_values = source.apply_fn(source.params, teacher_inputs, mutable=False)
    if actor:
        student_state = policy_distillation(
            target,
            teacher_values,
            student_inputs,
            num_epochs=num_epochs,
        )
    else:
        student_state = value_distillation(
            target, teacher_values, student_inputs, num_epochs=num_epochs
        )
    return student_state.params


def get_agent_state_from_agent_state(
    target: Union[SACState, AVGState],
    source: Union[SACState, AVGState],
    target_observations,
    source_observations,
    actions,
    num_epochs: int = 1,
    transfer_collector_state: bool = False,
) -> Union[SACState, AVGState]:
    distilled_actor_params = distill_source_to_target(
        source.actor_state,
        target.actor_state,
        student_inputs=target_observations,
        teacher_inputs=source_observations,
        actor=True,
        num_epochs=num_epochs,
    )
    distilled_critic_params = distill_source_to_target(
        source.critic_state,
        target.critic_state,
        student_inputs=jnp.hstack([target_observations, actions]),
        teacher_inputs=jnp.hstack(
            [source_observations, actions]
        ),  # actions are not normalized, so we can use them directly
        num_epochs=num_epochs,
    )
    target_actor_state = target.actor_state.replace(params=distilled_actor_params)  # type: ignore[union-attr]
    target_critic_state = target.critic_state.replace(params=distilled_critic_params)  # type: ignore[union-attr]
    if transfer_collector_state:
        target_collector_state = target.collector_state.replace(
            env_state=source.collector_state.env_state,
            last_obs=source.collector_state.last_obs,
            last_terminated=source.collector_state.last_terminated,
            last_truncated=source.collector_state.last_truncated,
        )  # type: ignore[union-attr]

        new_collector_state = broadcast_to_match(
            target.collector_state, target_collector_state
        )

        new_target_state = target.replace(  # type: ignore[union-attr]
            actor_state=target_actor_state,
            critic_state=target_critic_state,
            collector_state=new_collector_state,
        )

        return new_target_state

    return target.replace(  # type: ignore[union-attr]
        actor_state=target_actor_state,
        critic_state=target_critic_state,
    )


def fix_target_params(state):
    if state.target_params is not None:
        # If target_params are already set, return the state as is
        return state
    return state.replace(target_params=state.params)


def copy_state_to_state(
    source: Union[SACState, AVGState],
    target: Union[SACState, AVGState],
    tau: float,
    transfer_collector_state: bool = False,
):
    if transfer_collector_state:
        new_collector_state = target.collector_state.replace(
            env_state=source.collector_state.env_state,
            last_obs=source.collector_state.last_obs,
            last_terminated=source.collector_state.last_terminated,
            last_truncated=source.collector_state.last_truncated,
        )
    else:
        new_collector_state = target.collector_state

    new_actor_second_state = target.actor_state.second_state.replace(
        params=source.actor_state.params,
    ).soft_update(tau=tau)

    new_actor_state = target.actor_state.replace(second_state=new_actor_second_state)
    # new_source_critic_state = fix_target_params(
    #     source.critic_state
    # )  # to handle avg not having target params

    new_critic_second_state = target.critic_state.second_state.replace(
        params=source.critic_state.params,
    ).soft_update(tau=tau)
    new_critic_state = target.critic_state.replace(second_state=new_critic_second_state)
    return target.replace(
        actor_state=new_actor_state,
        critic_state=new_critic_state,
        collector_state=new_collector_state,
    )


def make_train(
    primary_env_args: EnvironmentConfig,
    secondary_env_args: EnvironmentConfig,
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
    num_epochs: int = 10,
    n_epochs_sac: int = 10,
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
    mode = "gymnax" if check_env_is_gymnax(primary_env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""
        raw_primary_agent_state = init_sac(
            key=key,
            env_args=primary_env_args,
            actor_optimizer_args=primary_actor_optimizer_args,
            critic_optimizer_args=primary_critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        raw_secondary_agent_state = init_AVG(
            key=key,
            env_args=secondary_env_args,
            actor_optimizer_args=secondary_actor_optimizer_args,
            critic_optimizer_args=secondary_critic_optimizer_args,
            network_args=network_args.replace(penultimate_normalization=True),
            alpha_args=alpha_args,
            num_critics=1,
        )

        raw_secondary_agent_state = raw_secondary_agent_state.replace(
            critic_state=fix_target_params(raw_secondary_agent_state.critic_state)
        )

        DoubleTrainStateAVG = get_double_train_state(
            "avg", dyna_factor=agent_config.dyna_factor
        )  # type: ignore[arg-type]
        DoubleTrainStateSAC = get_double_train_state(
            "sac", dyna_factor=agent_config.dyna_factor
        )

        def build_states_from_states_env_and_mode(
            raw_primary_agent_state, raw_secondary_agent_state, mode
        ):
            env_norm_info = (
                raw_secondary_agent_state.collector_state.env_state.info[
                    "normalization_info"
                ]
                if mode == "brax"
                else raw_secondary_agent_state.collector_state.env_state.normalization_info
            )
            primary_agent_actor_state = DoubleTrainStateSAC.from_LoadedTrainState(
                raw_primary_agent_state.actor_state,
                second_state=raw_secondary_agent_state.actor_state,
                norm_info=env_norm_info,
            )

            primary_agent_critic_state = DoubleTrainStateSAC.from_LoadedTrainState(
                raw_primary_agent_state.critic_state,
                second_state=raw_secondary_agent_state.critic_state,
                norm_info=env_norm_info,
            )

            primary_agent_state = raw_primary_agent_state.replace(
                actor_state=primary_agent_actor_state,
                critic_state=primary_agent_critic_state,
            )

            secondary_agent_actor_state = DoubleTrainStateAVG.from_LoadedTrainState(
                raw_secondary_agent_state.actor_state,
                second_state=raw_primary_agent_state.actor_state,
                norm_info=env_norm_info,
            )
            secondary_agent_critic_state = DoubleTrainStateAVG.from_LoadedTrainState(
                raw_secondary_agent_state.critic_state,
                second_state=raw_primary_agent_state.critic_state,
                norm_info=env_norm_info,
            )

            secondary_agent_state = raw_secondary_agent_state.replace(
                actor_state=secondary_agent_actor_state,
                critic_state=secondary_agent_critic_state,
            )

            return primary_agent_state, secondary_agent_state

        primary_agent_state, secondary_agent_state = (
            build_states_from_states_env_and_mode(
                raw_primary_agent_state, raw_secondary_agent_state, mode=mode
            )
        )

        # primary_agent_state = DoubleTrainState.from_LoadedTrainState(
        #     raw_primary_agent_state
        # )
        # secondary_agent_state = DoubleTrainState.from_LoadedTrainState(
        #     raw_secondary_agent_state
        # )

        # primary_agent_state = raw_primary_agent_state

        # agent_state = DynaSACState(
        #     primary=primary_agent_state, secondary=secondary_agent_state
        # )
        n_unroll = 2

        num_updates = total_timesteps // primary_env_args.n_envs
        _, action_shape = get_state_action_shapes(primary_env_args.env)

        primary_training_iteration_scan_fn = partial(
            primary_training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config.primary,
            mode=mode,
            env_args=primary_env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
            n_epochs=n_epochs_sac,
        )

        secondary_training_iteration_scan_fn = partial(
            secondary_training_iteration,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config.secondary,
            mode=mode,
            env_args=secondary_env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=False,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
        )

        def dyna_train_loop(
            carry: tuple[SACState, AVGState], _: Any
        ) -> tuple[tuple[SACState, AVGState], None]:
            primary_agent_state, secondary_agent_state = carry

            new_primary_agent_state, _ = jax.lax.scan(
                f=primary_training_iteration_scan_fn,
                init=primary_agent_state,
                xs=None,
                length=sac_length,
                unroll=n_unroll,
            )
            timestep = new_primary_agent_state.collector_state.timestep

            def do_training(_):
                transfered_secondary_agent_state = copy_state_to_state(
                    source=new_primary_agent_state,
                    target=secondary_agent_state,
                    tau=agent_config.dyna_tau(timestep),  # type: ignore[arg-type]
                )

                new_secondary_agent_state, _ = jax.lax.scan(
                    f=secondary_training_iteration_scan_fn,
                    init=transfered_secondary_agent_state,
                    xs=None,
                    length=avg_length,
                    unroll=n_unroll,
                )

                transfered_primary_agent_state = copy_state_to_state(
                    source=new_secondary_agent_state,
                    target=new_primary_agent_state,
                    tau=agent_config.dyna_tau(timestep),  # type: ignore[arg-type]
                )

                # transfered_primary_agent_state = get_agent_state_from_agent_state(
                #     target=new_primary_agent_state,
                #     source=new_secondary_agent_state,
                #     source_observations=normalized_obs,
                #     target_observations=observations,
                #     actions=actions,
                #     num_epochs=num_epochs,
                # )

                return transfered_primary_agent_state, new_secondary_agent_state

            def skip_training(_):
                return new_primary_agent_state, secondary_agent_state

            cond_pred = new_primary_agent_state.collector_state.timestep > 10_000

            transfered_primary_agent_state, new_secondary_agent_state = jax.lax.cond(
                cond_pred, do_training, skip_training, operand=None
            )

            assert jax.tree.map(
                lambda x, y: jnp.allclose(x, y),
                transfered_primary_agent_state.actor_state.params,
                new_primary_agent_state.actor_state.params,
            ), (
                "transfered_primary_agent_state and new_primary_agent_state are not"
                " equal"
            )

            assert jax.tree.map(
                lambda x, y: jnp.allclose(x, y),
                transfered_primary_agent_state.critic_state.params,
                new_primary_agent_state.critic_state.params,
            ), (
                "transfered_primary_agent_state and new_primary_agent_state are not"
                " equal"
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
            length=num_updates,
        )
        return primary_agent_state

    return train
