"""Training loop for Upside-Down RL (UDRL).

UDRL re-frames RL as supervised learning of a command-conditioned policy
pi(a | s, command), with command = (desired_return, desired_horizon).
At rollout, each parallel env carries its own command which decays with the
realised reward and resets at episode boundaries. At training, the realised
return-to-go and steps-until-done are used as targets for the same command
slots, and the actor is fit by NLL/MSE on the executed actions.

Implementation reuses Ajax's actor network (with a 2-dim wider input layer
to accommodate the appended command), the standard collector init, and the
shared logging plumbing. There is no critic and no replay buffer: this is an
on-policy supervised-learning loop close in shape to PPO.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.SAC.utils import SquashedNormal
from ajax.agents.UDRL.state import UDRLConfig, UDRLState
from ajax.agents.UDRL.utils import (
    compute_returns_to_go_horizons,
    update_command,
)
from ajax.environments.interaction import (
    get_pi,
    init_collector_state,
    step,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


@struct.dataclass
class UDRLAuxiliaries:
    actor_loss: jnp.ndarray
    mean_rtg: jnp.ndarray
    mean_horizon: jnp.ndarray


def init_UDRL(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
    window_size: int = 10,
) -> UDRLState:
    rng, init_key, collector_key = jax.random.split(key, 3)

    continuous = check_if_environment_has_continuous_actions(
        env_args.env, env_params=env_args.env_params
    )
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=continuous,
        action_value=False,
        squash=continuous,
        num_critics=1,
        extra_obs_dim=2,  # the (d_r, d_h) command is appended to obs
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        window_size=window_size,
        expert_state_aug_dim=2,  # reserves 2 trailing slots in last_obs for command
    )

    # Seed last_obs's command slots with (return_init, horizon_init).
    n_envs = env_args.n_envs
    init_cmd = jnp.tile(
        jnp.array(
            [agent_config.command_return_init, agent_config.command_horizon_init],
            dtype=collector_state.last_obs.dtype,
        ),
        (n_envs, 1),
    )
    new_last_obs = jnp.concatenate(
        [collector_state.last_obs[:, :-2], init_cmd], axis=-1
    )
    collector_state = collector_state.replace(last_obs=new_last_obs)

    return UDRLState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


def _udrl_collect_step(
    agent_state: UDRLState,
    _: Any,
    *,
    env_args: EnvironmentConfig,
    mode: str,
    agent_config: UDRLConfig,
    recurrent: bool,
):
    """One env step. Reads (d_r, d_h) from the trailing 2 dims of last_obs,
    forwards the actor on the augmented obs, steps the env, then writes the
    next augmented obs (raw_next_obs ++ updated_command) back into
    collector_state.last_obs.
    """
    cs = agent_state.collector_state
    rng, action_key, step_key = jax.random.split(agent_state.rng, 3)
    rng_step = (
        jax.random.split(step_key, env_args.n_envs) if mode == "gymnax" else step_key
    )

    last_obs = cs.last_obs  # (n_envs, raw_obs_dim + 2)
    pi, new_actor_state = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=last_obs,
        recurrent=recurrent,
    )
    action, log_probs = pi.sample_and_log_prob(seed=action_key)

    new_env_obs, new_env_state, reward, terminated, truncated, _ = jax.lax.stop_gradient(
        step(
            rng_step,
            cs.env_state,
            action,
            env_args.env,
            mode,
            env_args.env_params,
        )
    )
    done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool)).astype(
        jnp.float32
    )

    prev_d_r = last_obs[:, -2]
    prev_d_h = last_obs[:, -1]
    new_d_r, new_d_h = update_command(
        prev_d_r,
        prev_d_h,
        reward,
        done,
        return_init=agent_config.command_return_init,
        horizon_init=agent_config.command_horizon_init,
    )
    new_command = jnp.stack([new_d_r, new_d_h], axis=-1)
    new_last_obs = jnp.concatenate([new_env_obs, new_command], axis=-1)

    transition = Transition(
        obs=last_obs,
        action=action,
        reward=reward[:, None],
        terminated=terminated[:, None],
        truncated=truncated[:, None],
        next_obs=new_last_obs,
        log_prob=log_probs,
    )

    new_cs = cs.replace(
        rng=rng,
        _env_state=new_env_state,
        last_obs=new_last_obs,
        timestep=cs.timestep + env_args.n_envs,
        last_terminated=terminated,
        last_truncated=truncated,
    )
    return (
        agent_state.replace(
            collector_state=new_cs, actor_state=new_actor_state, rng=rng
        ),
        transition,
    )


def actor_loss_fn(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    obs_with_realised_command: jax.Array,
    actions: jax.Array,
    bc_loss_type: str,
) -> jax.Array:
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_with_realised_command,
    )
    if isinstance(pi, distrax.Categorical):
        # gymnax discrete actions arrive as (B,); other paths may carry a
        # trailing length-1 dim. Strip only that case so log_prob always
        # sees integer indices of shape (B,).
        if actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        log_p = pi.log_prob(actions)
        return -log_p.mean()

    if bc_loss_type == "mse":
        if hasattr(pi, "unsquashed_mean"):
            mu = pi.unsquashed_mean()
        else:
            mu = pi.mean()
        target = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
        return jnp.mean((mu - target) ** 2)

    # NLL on the squashed-normal / normal
    if isinstance(pi, SquashedNormal):
        eps = 1e-3
        target = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
        return -pi.log_prob(target).sum(-1).mean()
    return -pi.log_prob(actions).sum(-1).mean()


_actor_value_and_grad = jax.value_and_grad(actor_loss_fn)


def _replace_command_with_realised(
    obs: jax.Array, rtg: jax.Array, horizon: jax.Array
) -> jax.Array:
    """Overwrite the trailing 2 dims of obs with the per-step realised
    (RTG, horizon)."""
    base = obs[..., :-2]
    rtg = rtg.reshape(rtg.shape[:-1])  # drop trailing 1 if present
    horizon = horizon.reshape(horizon.shape[:-1])
    cmd = jnp.stack([rtg, horizon], axis=-1)
    return jnp.concatenate([base, cmd], axis=-1)


def _shuffle_and_minibatch(
    rng: jax.Array, arrays: Tuple[jax.Array, ...], batch_size: int
):
    """Return a tuple of shape (n_batches, batch_size, ...) per-array.
    The leading axis is partial-batch-padded by repeating earlier samples;
    UDRL's loss is a mean so duplicates do not bias gradients meaningfully
    on small budgets, and this keeps the function shape-static for jit.
    """
    n = arrays[0].shape[0]
    # Effective batch_size never exceeds n (avoids 0-length minibatches when
    # callers pass n_steps * n_envs < batch_size).
    bs = min(batch_size, n)
    n_batches = n // bs
    perm = jax.random.permutation(rng, n)
    take = n_batches * bs
    perm = perm[:take]
    out = tuple(a[perm].reshape((n_batches, bs) + a.shape[1:]) for a in arrays)
    return out


def training_iteration(
    agent_state: UDRLState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    agent_config: UDRLConfig,
    recurrent: bool,
) -> Tuple[UDRLState, UDRLAuxiliaries]:
    collect_fn = partial(
        _udrl_collect_step,
        env_args=env_args,
        mode=mode,
        agent_config=agent_config,
        recurrent=recurrent,
    )
    agent_state, rollout = jax.lax.scan(
        collect_fn, agent_state, xs=None, length=agent_config.n_steps
    )

    # rollout fields have leading axes (T, n_envs, ...). Compute realised
    # (RTG, horizon) per timestep with done masking, then flatten time x env
    # for shuffling.
    dones = jnp.logical_or(
        rollout.terminated.astype(bool), rollout.truncated.astype(bool)
    ).astype(jnp.float32)
    rtg, horizon = compute_returns_to_go_horizons(
        rollout.reward, dones, gamma=agent_config.gamma
    )

    obs_with_cmd = _replace_command_with_realised(rollout.obs, rtg, horizon)

    # Flatten (T, n_envs) -> (T * n_envs)
    flat_obs = obs_with_cmd.reshape((-1,) + obs_with_cmd.shape[2:])
    flat_action = rollout.action.reshape((-1,) + rollout.action.shape[2:])

    rng, shuffle_key = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    def epoch_step(carry, epoch_key):
        a_state = carry
        (obs_batches, act_batches) = _shuffle_and_minibatch(
            epoch_key, (flat_obs, flat_action), agent_config.batch_size
        )

        def batch_step(a_state, batch):
            obs_b, act_b = batch
            loss, grads = _actor_value_and_grad(
                a_state.actor_state.params,
                a_state.actor_state,
                obs_b,
                act_b,
                agent_config.bc_loss_type,
            )
            new_actor_state = a_state.actor_state.apply_gradients(grads=grads)
            return a_state.replace(actor_state=new_actor_state), loss

        a_state, losses = jax.lax.scan(batch_step, a_state, (obs_batches, act_batches))
        return a_state, losses.mean()

    epoch_keys = jax.random.split(shuffle_key, agent_config.n_epochs)
    agent_state, epoch_losses = jax.lax.scan(epoch_step, agent_state, epoch_keys)

    aux = UDRLAuxiliaries(
        actor_loss=epoch_losses.mean(),
        mean_rtg=rtg.mean(),
        mean_horizon=horizon.mean(),
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[Any] = None,
    **_unused: Any,
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    recurrent = network_args.lstm_hidden_size is not None

    @partial(jax.jit, static_argnames=("resume_from_state",))
    def train(
        key: jax.Array,
        index: Optional[int] = None,
        initial_state: Optional[UDRLState] = None,
        resume_from_state: bool = False,
    ):
        init_key, _ = jax.random.split(key, 2)
        if resume_from_state and initial_state is not None:
            agent_state = initial_state
        else:
            agent_state = init_UDRL(
                key=init_key,
                env_args=env_args,
                actor_optimizer_args=actor_optimizer_args,
                critic_optimizer_args=critic_optimizer_args,
                network_args=network_args,
                agent_config=agent_config,
            )

        per_iter = env_args.n_envs * agent_config.n_steps
        num_updates = max(total_timesteps // per_iter, 1)

        scan_fn = partial(
            training_iteration,
            env_args=env_args,
            mode=mode,
            agent_config=agent_config,
            recurrent=recurrent,
        )
        agent_state, aux = jax.lax.scan(
            scan_fn, agent_state, xs=None, length=num_updates
        )
        return agent_state, aux

    return train
