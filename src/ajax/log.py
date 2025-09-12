from typing import Any, Callable, Dict, Protocol

import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.evaluate import evaluate
from ajax.state import BaseAgentState


class AuxiliaryLogsProtocol(Protocol):
    ...


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return_dict = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, jax.Array):
                    # convert 0-d array to Python scalar
                    return_dict[f"{key}/{subkey}"] = (
                        subval[0] if jnp.ndim(subval) > 0 else subval
                    )
                else:
                    return_dict[f"{key}/{subkey}"] = subval
        else:
            if isinstance(val, jax.Array):
                return_dict[key] = val.item() if val.ndim == 0 else val
            else:
                return_dict[key] = val
    return return_dict


def prepare_metrics(aux):
    log_metrics = flatten_dict(to_state_dict(aux))
    return {key: val for (key, val) in log_metrics.items() if not (jnp.isnan(val))}


def no_op(agent_state, aux, *args):  # TODO : build from auxiliary logs?
    fake_metrics_to_log = {
        "timestep": -1,  # must be int
        "Eval/episodic mean reward": jnp.nan,
        "Eval/episodic entropy": jnp.nan,
        "Eval/mean average reward": jnp.nan,
        "Eval/mean bias": jnp.nan,
        "Train/episodic mean reward": jnp.nan,
    }
    aux_keys = flatten_dict(to_state_dict(aux)).keys()
    fake_metrics_to_log.update(dict.fromkeys(aux_keys, jnp.nan))
    return fake_metrics_to_log


def no_op_none(agent_state, index, timestep):
    pass


@partial(
    jax.jit,
    static_argnames=[
        "mode",
        "env_args",
        "num_episode_test",
        "recurrent",
        "lstm_hidden_size",
        "log",
        "verbose",
        "log_fn",
        "log_frequency",
        "total_timesteps",
        "avg_reward_mode",
    ],
)
def evaluate_and_log(
    agent_state: BaseAgentState,
    aux: AuxiliaryLogsProtocol,
    index: int,
    mode: str,
    env_args: int,
    num_episode_test: int,
    recurrent: bool,
    lstm_hidden_size: int,
    log: bool,
    verbose: bool,
    log_fn: Callable,
    log_frequency: int,
    total_timesteps: int,
    avg_reward_mode: bool = False,
):
    timestep = agent_state.collector_state.timestep

    def run_and_log(
        agent_state: BaseAgentState, aux: AuxiliaryLogsProtocol, index: int
    ):
        eval_key = agent_state.eval_rng
        obs_normalization = (
            "obs_normalization_info" in agent_state.collector_state.env_state.info
            if mode == "brax"
            else "normalization_info" in dir(agent_state.collector_state.env_state)
        )
        eval_rewards, eval_entropy, avg_avg_reward, avg_bias = evaluate(
            env_args.env,
            actor_state=agent_state.actor_state,
            num_episodes=num_episode_test,
            rng=eval_key,
            env_params=env_args.env_params,
            recurrent=recurrent,
            lstm_hidden_size=lstm_hidden_size,
            norm_info=(
                (
                    agent_state.collector_state.env_state.info["normalization_info"]
                    if mode == "brax"
                    else agent_state.collector_state.env_state.normalization_info
                )
                if obs_normalization
                else None
            ),
            avg_reward_mode=avg_reward_mode,
        )

        eval_rewards = eval_rewards.mean()

        metrics_to_log = {
            "timestep": timestep,
            "Eval/episodic mean reward": eval_rewards,
            "Eval/mean average reward": avg_avg_reward,
            "Eval/mean bias": avg_bias,
            "Eval/episodic entropy": eval_entropy,
            "Train/episodic mean reward": (
                agent_state.collector_state.episodic_mean_return
            ),
        }

        metrics_to_log.update(flatten_dict(to_state_dict(aux)))

        if verbose:
            jax.debug.print(
                (
                    "[Eval] Step={timestep_val}, Reward={rewards_val},"
                    " Entropy={entropy_val}"
                ),
                timestep_val=timestep,
                rewards_val=eval_rewards,
                entropy_val=eval_entropy,
            )

        if log:
            # jax.debug.print(
            #     "Calling log function {metrics_to_log}", metrics_to_log=metrics_to_log
            # )
            jax.debug.callback(log_fn, metrics_to_log, index)
        return metrics_to_log

    _, eval_rng = jax.random.split(agent_state.eval_rng)
    agent_state = agent_state.replace(eval_rng=eval_rng)

    log_flag = (
        timestep - (agent_state.n_logs * log_frequency) >= log_frequency
        if log
        else False
    )

    agent_state = agent_state.replace(
        n_logs=jax.lax.select(log_flag, agent_state.n_logs + 1, agent_state.n_logs)
    )
    flag = jnp.logical_or(
        jnp.logical_and(log_flag, timestep > 1),
        timestep >= (total_timesteps - env_args.n_envs),
    )
    metrics_to_log = jax.lax.cond(flag, run_and_log, no_op, agent_state, aux, index)

    del aux

    jax.clear_caches()

    return agent_state, metrics_to_log
