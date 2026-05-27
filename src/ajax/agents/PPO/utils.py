from typing import Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute gae advantages

    Args:
        traj_batch (Transition): The transition buffer
        last_val (jax.Array): The value of the last state encoutered.
        gamma (float): The discount factor to consider.
        gae_lambda (float): The gae lambda parameter to consider.

    Returns:
        tuple[jax.Array, jax.Array]: Gae and value (carry-over values to feed for next\
              iteration)\
            and the gae for actual return
    """

    def _get_advantages(
        gae: tuple[jax.Array, jax.Array, Optional[jax.Array]],
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Iteratively compute advantages in gae style using previous gae, next value\
              and transition buffer

        Args:
            gae_and_next_value (tuple[jax.Array, jax.Array]): Previous gae and next_value
            transition (Transition): The transitions to consider

        Returns:
            tuple[tuple[jax.Array, jax.Array], jax.Array]: The updated gaes + \
                the transition's values
        """
        # current estimation of gae + value at t+1 because we have working in reverse
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        done = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * done * gae

        # tuple is carry-over state for scan, gae after the comma is the actual return at the end of the scan
        return gae, gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=jnp.zeros_like(values[-1]),
        xs=(rewards, values, next_values, terminateds, truncateds),
        reverse=True,
        unroll=1,
    )
    returns = advantages + values
    return advantages, returns


@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
def _compute_gae_vtrace(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Brax PPO's V-trace-style GAE (port of ``brax.training.agents.ppo.losses.compute_gae``).

    Differences from :func:`_compute_gae` (standard GAE):

    1. **Truncation mask zeros the delta entirely at truncated steps.**
       At a truncation boundary, the env auto-resets and ``next_obs``
       comes from a fresh initial state. Bootstrapping the value at
       that fresh state is causally meaningless (the action you took
       at truncation has no effect on what the reset state is), so
       brax treats the truncation step as a dropped frame: zero delta,
       zero advantage. Standard GAE bootstraps anyway, contaminating
       the policy gradient with noise at every rollout that spans a
       truncation.

    2. **Value target propagates forward (V-trace-style).** The
       accumulator ``acc`` builds ``vs - V``; then ``vs = (vs - V) + V``
       is the value target, and ``vs_t_plus_1`` (the bootstrapped
       value at the next step) replaces the raw ``next_value`` in the
       advantage computation. Lower-variance targets when the value
       function is partially learned.

    3. **Single bootstrap value at the rollout tail.** ``bootstrap_value``
       is computed ONCE on the obs *after* the last timestep (caller
       responsibility), rather than recomputed per-step via
       ``predict_value(next_obs)``. Avoids drift from running-stats
       updates and double-evaluation of overlapping states (M7).

    Args:
        rewards: shape ``(T, n_envs[, 1])``
        values: shape ``(T, n_envs, 1)`` — V(obs[t]) for t = 0..T-1
        bootstrap_value: shape ``(n_envs, 1)`` — V(obs[T]) for the
            *single* obs after the rollout's last action
        terminateds: shape ``(T, n_envs[, 1])`` — 1 at real episode end
        truncateds: shape ``(T, n_envs[, 1])`` — 1 at timeout
        gamma, gae_lambda: standard

    Returns:
        (advantages, value_targets), both shape ``(T, n_envs, 1)``.
    """
    # vs_t_plus_1 for last step = bootstrap_value; the scan walks
    # backward from t=T-1 to t=0 carrying acc = vs_minus_v at t+1.
    # values_t_plus_1[t] for t < T-1 = values[t+1]; for t = T-1 =
    # bootstrap_value. We avoid building this array explicitly; the
    # scan threads it through the carry.

    def _get_vs_step(carry, transition):
        acc, vs_t_plus_1 = carry
        reward, value, terminated, truncated = transition
        truncation_mask = 1.0 - truncated
        non_terminal = 1.0 - terminated
        # delta on the V-trace recursion: zeroed at truncation.
        delta = (reward + gamma * non_terminal * vs_t_plus_1 - value) * truncation_mask
        # acc = vs_minus_v at step t = delta + γ (1 - term) λ (1 - trunc) * acc
        acc = delta + gamma * gae_lambda * non_terminal * truncation_mask * acc
        vs = acc + value  # value target at step t
        # carry: (acc for next step, vs at this step becomes vs_t_plus_1
        # for the previous step in reverse-scan)
        return (acc, vs), (vs, value, reward, terminated, truncated)

    init_carry = (jnp.zeros_like(values[-1]), bootstrap_value)
    _, scan_out = jax.lax.scan(
        f=_get_vs_step,
        init=init_carry,
        xs=(rewards, values, terminateds, truncateds),
        reverse=True,
        unroll=1,
    )
    vs, values_seq, rewards_seq, term_seq, trunc_seq = scan_out
    # Value targets:
    value_targets = vs
    # Advantages: (r + γ (1-term) vs_t_plus_1 - V) * truncation_mask.
    # Build vs_t_plus_1 by shifting vs forward one step and bootstrapping
    # at the tail.
    vs_t_plus_1 = jnp.concatenate([vs[1:], bootstrap_value[None]], axis=0)
    advantages = (rewards_seq + gamma * (1.0 - term_seq) * vs_t_plus_1 - values_seq) * (
        1.0 - trunc_seq
    )
    return advantages, value_targets


@partial(jax.jit, static_argnames=["num_minibatches"])
def get_minibatches_from_batch(
    batch: tuple[jax.Array, ...], rng: jax.Array, num_minibatches: int
):
    """Split a ``(T, n_envs, ...)`` rollout batch into ``num_minibatches``
    minibatches of shape ``(num_minibatches, T*n_envs/num_minibatches, ...)``.

    The flatten step joins the leading ``(T, n_envs)`` dims into a single
    sample axis regardless of how many trailing feature dims a leaf has,
    so leaves with a feature dim (e.g. observations ``(T, n_envs, obs_dim)``)
    and leaves without one (e.g. ``terminated``, ``truncated`` of shape
    ``(T, n_envs)``) are reshuffled with the same permutation and stay
    aligned across leaves.

    Pre-fix history: the previous implementation used
    ``x.reshape((-1, x.shape[-1]))``. For obs ``(T, n_envs, obs_dim)``
    this correctly flattens ``T*n_envs`` samples; but for scalar leaves
    ``(T, n_envs)`` it produces ``(T, n_envs)`` (a no-op since
    ``x.shape[-1] == n_envs``), which then gets shuffled and reshaped
    differently from the obs path and ends up misaligned with the
    samples in the obs minibatch. PPO got away with it because
    ``terminated``/``truncated`` were only used for recurrent ``done``
    resets, which are no-ops for non-recurrent agents. Still, the bug
    was real; this rewrite fixes it.
    """
    T = batch[0].shape[0]
    n_envs = batch[0].shape[1]
    total = T * n_envs
    assert total % num_minibatches == 0, (
        "T * n_envs should be a multiple of num_minibatches, got "
        f"T={T}, n_envs={n_envs}, total={total}, {num_minibatches=}"
    )

    # Flatten the leading (T, n_envs) dims into a single sample axis for
    # every leaf, regardless of trailing feature dims.
    new_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((total,) + x.shape[2:]), batch
    )
    # Same rng + same leading dim => identical permutation across leaves
    # => sample alignment preserved.
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jax.random.permutation(rng, x, axis=0), new_batch
    )
    # Split the flat sample axis into num_minibatches chunks.
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]),
        shuffled_batch,
    )
    return minibatches
