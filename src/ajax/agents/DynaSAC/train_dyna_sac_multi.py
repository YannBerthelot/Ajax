import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.agents.AVG.train_AVG import init_AVG
from ajax.agents.AVG.train_AVG import training_iteration as training_iteration_AVG
from ajax.agents.DynaSAC.state import AVGState, DynaSACConfig, SACState
from ajax.agents.sac.train_sac import init_sac
from ajax.agents.sac.train_sac import training_iteration as training_iteration_SAC
from ajax.buffers.utils import get_batch_from_buffer, get_buffer
from ajax.distillation import distillation
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
    normalize_observation,
)
from ajax.types import BufferType

PROFILER_PATH = "./tensorboard"
DEBUG = False


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


def polyak_average_params(old_params, new_params, alpha: float):
    """Performs Polyak averaging of parameters."""
    return jax.tree_util.tree_map(
        lambda p_old, p_new: (1 - alpha) * p_old + alpha * p_new, old_params, new_params
    )


def polyak_average_trainstates(state_old, state_new, alpha: float):
    """Returns a new TrainState with params averaged between two TrainStates."""
    mixed_params = polyak_average_params(state_old.params, state_new.params, alpha)
    return state_old.replace(params=mixed_params)


def distill_source_to_target(
    teacher,
    student,
    teacher_inputs,
    student_inputs,
    num_epochs=1,
    actor: bool = False,
    vmap: bool = False,
    distillation_lr: float = 1e-4,
    alpha_polyak_primary_to_secondary: float = 1e-3,
    alpha_polyak_secondary_to_primary: float = 1e-3,
):
    mode = "actor" if actor else "critic"
    if vmap:
        teacher_values = jax.vmap(teacher.apply_fn, in_axes=(0, None))(
            teacher.params, teacher_inputs
        )
    else:
        teacher_values = teacher.apply_fn(teacher.params, teacher_inputs)
    if vmap:
        new_student_state, loss = distillation(
            student,
            teacher_values=teacher_values,
            student_inputs=student_inputs,
            num_epochs=num_epochs,
            distillation_lr=distillation_lr,
            mode=mode,
            vmap=vmap,
        )
    else:
        new_student_state, loss = jax.vmap(
            distillation, in_axes=(0, None, None, None, None, None)
        )(student, teacher_values, student_inputs, mode, num_epochs, distillation_lr)
        loss = loss.reshape(-1)
    alpha = (
        alpha_polyak_secondary_to_primary if vmap else alpha_polyak_primary_to_secondary
    )
    student_state = polyak_average_trainstates(
        student, new_student_state, alpha=alpha
    )  # TODO : parameterize
    return student_state, loss


def get_agent_state_from_agent_state(
    student: Union[SACState, AVGState],
    teacher: Union[SACState, AVGState],
    student_observations,
    teacher_observations,
    actions,
    num_epochs: int = 1,
    transfer_collector_state: bool = False,
    vmap: bool = False,
    actor_distillation_lr: float = 1e-4,
    critic_distillation_lr: float = 1e-4,
    alpha_polyak_primary_to_secondary: float = 1e-3,
    alpha_polyak_secondary_to_primary: float = 1e-3,
) -> tuple[Union[SACState, AVGState], tuple[jnp.ndarray, jnp.ndarray]]:
    distilled_actor, actor_distillation_loss = distill_source_to_target(
        teacher=teacher.actor_state,
        teacher_inputs=teacher_observations,
        student=student.actor_state,
        student_inputs=student_observations,
        actor=True,
        num_epochs=num_epochs,
        vmap=vmap,
        distillation_lr=actor_distillation_lr,
        alpha_polyak_primary_to_secondary=alpha_polyak_primary_to_secondary,
        alpha_polyak_secondary_to_primary=alpha_polyak_secondary_to_primary,
    )
    distilled_critic, critic_distillation_loss = distill_source_to_target(
        teacher=teacher.critic_state,
        teacher_inputs=jnp.hstack(
            [teacher_observations, actions]
        ),  # actions are not normalized, so we can use them directly
        student=student.critic_state,
        student_inputs=jnp.hstack([student_observations, actions]),
        num_epochs=num_epochs,
        vmap=vmap,
        distillation_lr=critic_distillation_lr,
    )

    distilled_critic = distilled_critic.soft_update(tau=0.01)  # TODO : parameterize
    # student_actor_state = student.actor_state.replace(params=distilled_actor_params)  # type: ignore[union-attr]
    # student_critic_state = student.critic_state.replace(params=distilled_critic_params)  # type: ignore[union-attr]
    # student_critic_state = student_critic_state.soft_update(tau=0.5)
    if transfer_collector_state:
        target_collector_state = student.collector_state.replace(
            env_state=teacher.collector_state.env_state,
            last_obs=teacher.collector_state.last_obs,
            last_terminated=teacher.collector_state.last_terminated,
            last_truncated=teacher.collector_state.last_truncated,
        )  # type: ignore[union-attr]

        new_collector_state = broadcast_to_match(
            student.collector_state, target_collector_state
        )

        new_target_state = student.replace(  # type: ignore[union-attr]
            actor_state=distilled_actor,
            critic_state=distilled_critic,
            collector_state=new_collector_state,
        )

        return new_target_state

    return student.replace(  # type: ignore[union-attr]
        actor_state=distilled_actor,
        critic_state=distilled_critic,
    ), (actor_distillation_loss, critic_distillation_loss)


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

    return target.replace(
        collector_state=new_collector_state,
    )


def replace_with_nan(metrics_dict):
    new_metrics = {}
    for key, value in metrics_dict.items():
        # Extract the shape from the traced value
        shape = value.aval.shape  # Get shape from abstract value
        dtype = (
            value.aval.dtype
        )  # Preserve dtype (usually float32, but could be int32 etc.)
        if jnp.issubdtype(dtype, jnp.floating):
            new_metrics[key] = jnp.full(shape, jnp.nan, dtype=dtype)
        else:
            # For integer types, use a placeholder like -1 or similar (NaN is not valid for int)
            new_metrics[key] = jnp.full(shape, -1, dtype=dtype)
    return new_metrics


def frozen_dict_equal(x, y):
    return jax.tree.all(jax.tree.map(lambda x, y: (x == y).all(), x, y))


def check_equality(x, y, msg: str):
    assert frozen_dict_equal(x, y), msg


def check_inequality(x, y, msg: str):
    assert not frozen_dict_equal(x, y), f"{msg}"


def make_train(
    primary_env_args: EnvironmentConfig,
    secondary_env_args: EnvironmentConfig,
    primary_actor_optimizer_args: OptimizerConfig,
    primary_critic_optimizer_args: OptimizerConfig,
    secondary_actor_optimizer_args: OptimizerConfig,
    secondary_critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    alt_buffer: BufferType,
    agent_config: DynaSACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    sac_length: int = 1,
    avg_length: int = 1,
    num_epochs: int = 1,
    n_epochs_sac: int = 1,
    n_avg_agents: int = 1,
    actor_distillation_lr: float = 1e-4,
    critic_distillation_lr: float = 1e-4,
    n_distillation_samples: int = 1_000,
    alpha_polyak_primary_to_secondary: float = 1e-3,
    initial_alpha_polyak_secondary_to_primary: float = 1e-3,
    final_alpha_polyak_secondary_to_primary: float = 1e-3,
    sweep: bool = False,
    transition_mix_fraction: float = 0.5,
    transfer_mode: str = "copy",
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
    log = (logging_config is not None) and (not sweep)
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    def alpha_polyak_schedule(t: int):
        return initial_alpha_polyak_secondary_to_primary + (t / total_timesteps) * (
            final_alpha_polyak_secondary_to_primary
            - initial_alpha_polyak_secondary_to_primary
        )

    # Start async logging if logging is enabled
    if logging_config is not None and not sweep:
        start_async_logging()
    secondary_mode = "sac"
    init_secondary = init_sac if secondary_mode == "sac" else init_AVG
    init_secondary_args = (
        {
            "env_args": secondary_env_args,
            "actor_optimizer_args": secondary_actor_optimizer_args,
            "critic_optimizer_args": secondary_critic_optimizer_args,
            "network_args": network_args.replace(penultimate_normalization=True),
            "alpha_args": alpha_args,
            "num_critics": 1,
        }
        if secondary_mode == "avg"
        else {
            "env_args": secondary_env_args,
            "actor_optimizer_args": primary_actor_optimizer_args,
            "critic_optimizer_args": primary_critic_optimizer_args,
            "network_args": network_args,
            "alpha_args": alpha_args,
            "buffer": None,
        }
    )
    primary_training_iteration = training_iteration_SAC
    secondary_training_iteration = (
        training_iteration_SAC if secondary_mode == "sac" else training_iteration_AVG
    )

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""
        primary_agent_state = init_sac(
            key=key,
            env_args=primary_env_args,
            actor_optimizer_args=primary_actor_optimizer_args,
            critic_optimizer_args=primary_critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        partial_init_secondary = partial(
            init_secondary,
            **init_secondary_args,
        )
        avg_keys = jax.random.split(key, n_avg_agents)
        # secondary_agent_state = (
        #     jax.vmap(partial_init_secondary)(avg_keys)
        #     if secondary_mode == "avg"
        #     else partial_init_secondary(key)
        # )
        secondary_agent_state = init_sac(
            key=key,
            env_args=secondary_env_args,
            actor_optimizer_args=primary_actor_optimizer_args,
            critic_optimizer_args=primary_critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=None,
        )
        if primary_agent_state.collector_state.buffer_state is not None:
            buffer_size = primary_agent_state.collector_state.buffer_state.experience[
                "obs"
            ].shape[1]
        distillation_buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=n_distillation_samples,
            n_envs=primary_env_args.n_envs,
        )

        n_unroll = 2

        num_updates = total_timesteps // (primary_env_args.n_envs * sac_length)
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
            log=False,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
            n_epochs=n_epochs_sac,
            transition_mix_fraction=1.0,
        )

        training_secondary_args = (
            {
                "recurrent": network_args.lstm_hidden_size is not None,
                "action_dim": action_shape[0],
                "agent_config": agent_config.secondary,
                "mode": mode,
                "env_args": secondary_env_args,
                "num_episode_test": num_episode_test,
                "log_fn": log_fn,
                "index": index,
                "log": False,
                "total_timesteps": total_timesteps,
                "log_frequency": (
                    logging_config.log_frequency if logging_config is not None else None
                ),
            }
            if secondary_mode == "avg"
            else {
                "buffer": buffer,
                "recurrent": network_args.lstm_hidden_size is not None,
                "action_dim": action_shape[0],
                "agent_config": agent_config.primary.replace(learning_starts=0),
                "mode": mode,
                "env_args": secondary_env_args,
                "num_episode_test": num_episode_test,
                "log_fn": log_fn,
                "index": index,
                "log": False,
                "total_timesteps": total_timesteps,
                "log_frequency": (
                    logging_config.log_frequency if logging_config is not None else None
                ),
                "horizon": (
                    logging_config.horizon if logging_config is not None else None
                ),
                "n_epochs": n_epochs_sac,
                "transition_mix_fraction": transition_mix_fraction,
            }
        )

        secondary_training_iteration_scan_fn = (
            jax.vmap(partial(secondary_training_iteration, **training_secondary_args))
            if secondary_mode == "avg"
            else partial(secondary_training_iteration, **training_secondary_args)
        )

        _, init_metrics_to_log = jax.lax.scan(
            f=secondary_training_iteration_scan_fn,
            init=secondary_agent_state,
            xs=None,
            length=avg_length,
            unroll=n_unroll,
        )
        assert secondary_agent_state.collector_state.buffer_state is None

        def dyna_train_loop(
            carry: tuple[SACState, AVGState], _: Any
        ) -> tuple[tuple[SACState, AVGState], None]:
            primary_agent_state, secondary_agent_state = carry

            timestep = primary_agent_state.collector_state.timestep

            alpha_polyak_secondary_to_primary = alpha_polyak_schedule(timestep)

            new_primary_agent_state, metrics_to_log_primary = jax.lax.scan(
                f=primary_training_iteration_scan_fn,
                init=primary_agent_state,
                xs=None,
                length=sac_length,
                unroll=n_unroll,
            )
            assert secondary_agent_state.collector_state.buffer_state is None
            # new_primary_agent_state = new_primary_agent_state.replace(
            #     critic_state=primary_agent_state.critic_state,
            #     actor_state=primary_agent_state.actor_state,
            #     # collector_state=primary_agent_state.collector_state,
            # )
            # jax.debug.print(
            #     "{x}/{y} n_updates:{z} n_envs:{a}",
            #     x=new_primary_agent_state.collector_state.timestep,
            #     y=total_timesteps,
            #     z=num_updates,
            #     a=primary_env_args.n_envs,
            # )

            def do_training(new_primary_agent_state, secondary_agent_state):
                def get_obs_normed_obs_action(agent_state, secondary_agent_state, key):
                    (
                        observations,
                        _,
                        _,
                        _,
                        _,
                        actions,
                    ) = get_batch_from_buffer(
                        distillation_buffer,
                        agent_state.collector_state.buffer_state,
                        key,  # TODO : change
                    )
                    if secondary_mode == "sac":
                        return observations, actions, observations

                    env_norm_info = (
                        secondary_agent_state.collector_state.env_state.info[
                            "normalization_info"
                        ]
                        if mode == "brax"
                        else secondary_agent_state.collector_state.env_state.normalization_info
                    )
                    flattened_norm_info = jax.tree.map(
                        lambda x: jnp.mean(x, axis=(0, 1)).reshape(1, -1),
                        env_norm_info.obs,
                    )

                    normalized_obs = normalize_observation(
                        observations, flattened_norm_info
                    )
                    return observations, actions, normalized_obs

                observations, actions, normalized_obs = get_obs_normed_obs_action(
                    new_primary_agent_state,
                    secondary_agent_state,
                    key,  # TODO : change key
                )
                # transfered_secondary_agent_state = copy_state_to_state(
                #     source=new_primary_agent_state,
                #     target=secondary_agent_state,
                #     tau=agent_config.dyna_tau(timestep),  # type: ignore[arg-type]
                # )

                def transfer_agent_state(
                    source_state,
                    target_state,
                    transfer_mode: str,
                    teacher_observations,
                    student_observations,
                    buffer_transfer: bool = False,
                ):
                    if transfer_mode == "distillation":
                        (
                            new_state,
                            (
                                actor_distillation_loss,
                                critic_distillation_loss,
                            ),
                        ) = get_agent_state_from_agent_state(
                            student=target_state,
                            student_observations=student_observations,
                            teacher=source_state,
                            teacher_observations=teacher_observations,
                            actions=actions,
                            num_epochs=num_epochs,
                            vmap=True,
                            actor_distillation_lr=actor_distillation_lr,
                            critic_distillation_lr=critic_distillation_lr,
                            alpha_polyak_primary_to_secondary=alpha_polyak_primary_to_secondary,
                            alpha_polyak_secondary_to_primary=alpha_polyak_secondary_to_primary,
                        )
                        losses = (
                            actor_distillation_loss,
                            critic_distillation_loss,
                        )
                    elif transfer_mode == "copy":
                        if buffer_transfer:
                            n_envs = secondary_env_args.n_envs
                            new_env_state = jax.tree.map(
                                lambda x: jnp.tile(
                                    x,
                                    (n_envs,) + (1,) * (x.ndim - 1),
                                ),
                                source_state.collector_state.env_state,
                            )
                            new_collector_state = target_state.collector_state.replace(
                                env_state=new_env_state,
                                last_obs=jnp.tile(
                                    source_state.collector_state.last_obs,
                                    reps=(n_envs,)
                                    + (1,)
                                    * (source_state.collector_state.last_obs.ndim - 1),
                                ),
                                buffer_state=source_state.collector_state.buffer_state,
                            )
                        else:
                            new_collector_state = target_state.collector_state

                        new_state = target_state.replace(
                            actor_state=source_state.actor_state,
                            critic_state=source_state.critic_state,
                            collector_state=new_collector_state,
                        )
                        losses = (
                            jnp.nan,
                            jnp.nan,
                        )
                    else:  # None
                        new_state = target_state
                        losses = (
                            jnp.nan,
                            jnp.nan,
                        )
                    return new_state, losses

                transfered_secondary_agent_state, losses_1 = transfer_agent_state(
                    source_state=new_primary_agent_state,
                    target_state=secondary_agent_state,
                    transfer_mode=transfer_mode,
                    teacher_observations=observations,
                    student_observations=normalized_obs,
                    buffer_transfer=True,
                )
                if DEBUG:
                    jax.debug.callback(
                        check_inequality,
                        secondary_agent_state.actor_state.params,
                        transfered_secondary_agent_state.actor_state.params,
                        msg="Actor params should have changed after update 1",
                    )
                    jax.debug.callback(
                        check_equality,
                        new_primary_agent_state.actor_state.params,
                        transfered_secondary_agent_state.actor_state.params,
                        msg=(
                            "Actor params should NOT have changed between copy and"
                            " update"
                        ),
                    )
                new_secondary_agent_state, metrics_to_log_secondary = jax.lax.scan(
                    f=secondary_training_iteration_scan_fn,
                    init=transfered_secondary_agent_state,
                    xs=None,
                    length=avg_length,
                    unroll=n_unroll,
                )
                if avg_length == 0:
                    jax.debug.callback(
                        check_equality,
                        new_secondary_agent_state.actor_state.params,
                        transfered_secondary_agent_state.actor_state.params,
                        msg=(
                            "Actor params should NOT have changed between copy and"
                            " update"
                        ),
                    )
                if DEBUG:
                    jax.debug.callback(
                        check_inequality,
                        new_secondary_agent_state.actor_state.params,
                        transfered_secondary_agent_state.actor_state.params,
                        msg="Actor params should have changed after update 2",
                    )
                    jax.debug.callback(
                        check_equality,
                        new_primary_agent_state.actor_state.params,
                        transfered_secondary_agent_state.actor_state.params,
                        msg=(
                            "Actor params should NOT have changed between copy and"
                            " update"
                        ),
                    )

                assert (
                    observations.shape[0] == normalized_obs.shape[0] == actions.shape[0]
                ), (
                    f"observations shape {observations.shape}, "
                    f"normalized_obs shape {normalized_obs.shape}, "
                    f"actions shape {actions.shape}"
                )

                transfered_primary_agent_state, losses_2 = transfer_agent_state(
                    source_state=new_secondary_agent_state,
                    target_state=new_primary_agent_state,
                    transfer_mode=transfer_mode,
                    student_observations=observations,
                    teacher_observations=normalized_obs,
                    buffer_transfer=False,
                )
                if DEBUG:
                    jax.debug.callback(
                        check_inequality,
                        transfered_primary_agent_state.actor_state.params,
                        new_primary_agent_state.actor_state.params,
                        msg="Actor params should have changed after update 3",
                    )

                    jax.debug.callback(
                        check_equality,
                        transfered_primary_agent_state.actor_state.params,
                        new_secondary_agent_state.actor_state.params,
                        msg=(
                            "Actor params should NOT have changed between copy and"
                            " update"
                        ),
                    )

                return (
                    transfered_primary_agent_state,
                    secondary_agent_state,
                    (*losses_1, *losses_2),
                    metrics_to_log_secondary,
                )

            def skip_training(new_primary_agent_state, secondary_agent_state):
                nan_metrics = replace_with_nan(init_metrics_to_log)
                return (
                    new_primary_agent_state,
                    secondary_agent_state,
                    (
                        jnp.nan,
                        jnp.nan,
                        jnp.nan,
                        jnp.nan,
                    ),
                    nan_metrics,
                )

            cond_pred = (
                new_primary_agent_state.collector_state.timestep
                >= 1e4  # FIXME : add learning starts here
            )  # TODO : investigate

            (
                transfered_primary_agent_state,
                new_secondary_agent_state,
                distillation_losses,
                metrics_to_log_secondary,
            ) = jax.lax.cond(
                cond_pred,
                do_training,
                skip_training,
                new_primary_agent_state,
                secondary_agent_state,
            )
            metrics_to_log_primary = {
                f"{key} primary": val for key, val in metrics_to_log_primary.items()
            }
            metrics_to_log_secondary = {
                f"{key} secondary": val for key, val in metrics_to_log_secondary.items()
            }
            full_metrics_to_log = dict(
                **metrics_to_log_primary, **metrics_to_log_secondary
            )

            # metrics_to_log.update(metrics_to_log_secondary)
            # metrics_to_log = metrics_to_log_primary

            # (
            #     transfered_primary_agent_state,
            #     new_secondary_agent_state,
            #     distillation_losses,
            # ) = do_training()
            timestep = new_primary_agent_state.collector_state.timestep

            def wandb_log_fn(agent_state, _metrics_to_log):
                _metrics_to_log.update(
                    {
                        "timestep": timestep,
                        "Distillation/sac_to_avg_actor_distillation_loss": (
                            distillation_losses[0]
                        ),
                        "Distillation/sac_to_avg_critic_distillation_loss": (
                            distillation_losses[1]
                        ),
                        "Distillation/avg_to_sac_actor_distillation_loss": (
                            distillation_losses[2]
                        ),
                        "Distillation/avg_to_sac_critic_distillation_loss": (
                            distillation_losses[3]
                        ),
                    }
                )
                # _metrics_to_log = jax.tree.map(lambda x: x.squeeze(), _metrics_to_log)
                # jax.debug.breakpoint()
                jax.debug.callback(log_fn, _metrics_to_log, index)
                return agent_state  # .replace(n_logs=agent_state.n_logs + 1)

            def no_op_none(agent_state, _metrics_to_log):
                return agent_state

            score = full_metrics_to_log["Eval/episodic mean reward primary"]
            if logging_config is not None:
                # log_flag = jnp.logical_and(
                #     (
                #         timestep
                #         - (primary_agent_state.n_logs * logging_config.log_frequency)
                #         >= logging_config.log_frequency
                #     ),
                #     log,
                # )

                # flag = jnp.logical_or(
                #     jnp.logical_and(log_flag, timestep > 1),
                #     timestep >= (total_timesteps - primary_env_args.n_envs),
                # )

                #####
                log_frequency = logging_config.log_frequency
                log_flag = (
                    timestep - ((primary_agent_state.n_logs) * log_frequency)
                    >= log_frequency
                )

                flag = jnp.logical_or(
                    jnp.logical_and(log_flag, timestep > 1),
                    timestep >= (total_timesteps - primary_env_args.n_envs),
                )
                transfered_primary_agent_state = jax.lax.cond(
                    flag,
                    wandb_log_fn,
                    no_op_none,
                    transfered_primary_agent_state,
                    full_metrics_to_log,
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
            ), score

        (primary_agent_state, secondary_agent_state), score = jax.lax.scan(
            f=dyna_train_loop,
            init=(
                primary_agent_state,
                secondary_agent_state,
            ),
            xs=None,
            length=num_updates,
        )

        # stop_async_logging()
        window_size = int(0.1 * total_timesteps)

        average_score = jnp.nanmean(score[-window_size:])
        return primary_agent_state, average_score

    return train
