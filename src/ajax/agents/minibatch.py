"""How an on-policy update cuts its rollout into minibatches and runs its
epochs over them: shared by PPO, APO and PQN.

A rollout's RolloutSplit says how it is cut into minibatches, and
so where its agent computes the advantages (GAE):

* flat (feedforward default): GAE once on the whole rollout, then its
  n_steps * n_envs single steps shuffled into the minibatches (PPO's
  Algorithm 1, OpenAI Baselines' ppo2);
* time (with unroll_length): fragments of unroll_length steps,
  each in time order, shuffled into the minibatches, GAE recomputed in
  every minibatch from the current critic (brax's compute_ppo_loss);
* recurrent: GAE once on the whole rollout, then whole sequences of
  length steps in time order shuffled into the minibatches, each
  replayed from the memory it started with.
"""

import dataclasses
import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.deprecation import moved_names
from ajax.state import Transition


@dataclasses.dataclass(frozen=True)
class RolloutSplit:
    """How one rollout is cut into minibatches, chosen once per update from
    the hyperparameters: how many minibatches each epoch has
    (num_minibatches), how many steps each piece holds (length) and
    whether the pieces keep time order (kind, the module docstring)."""

    kind: str  # "flat", "time" or "recurrent"
    num_minibatches: int
    length: int  # steps per fragment ("time") or sequence ("recurrent")

    @property
    def advantages_per_minibatch(self) -> bool:
        """Whether GAE is computed again in each minibatch, from the current
        critic (brax's fragments), rather than once on the rollout."""
        return self.kind == "time"

    def minibatches(self, batch: dict, key: jax.Array, carries: Any = None) -> dict:
        """One epoch's partition of the rollout batch on key; a
        recurrent split's sequences take their start carries along."""
        num_minibatches, length = self.num_minibatches, self.length
        if self.kind == "time":
            return get_minibatches_preserving_time(batch, key, num_minibatches, length)
        if self.kind == "recurrent":
            return sequence_minibatches(batch, carries, key, num_minibatches, length)
        return get_minibatches_from_batch(batch, key, num_minibatches)


def default_bptt_length(n_steps: int, n_envs: int, num_minibatches: int) -> int:
    """The longest sequences that cut every env's rollout evenly and split
    exactly into num_minibatches; an explicit bptt_length splits
    only if it divides this one.

    Libraries disagree on a recurrent PPO's BPTT length (RLlib 20 steps,
    Sample Factory the rollout, Pleines et al., 2022, tuned per task), so the
    minibatches set it. When n_envs is a multiple of num_minibatches
    it is the whole rollout, each minibatch a subset of the envs, as in
    Baselines' ppo2 (ea25b9e, ppo2.py:167-180), CleanRL's ppo_atari_lstm
    (fe8d8a0, :296-313) and PureJaxRL's ppo_rnn (31756b1, :258-291).
    Otherwise each rollout is cut into the fewest equal pieces that split:
    num_minibatches // gcd(n_envs, num_minibatches) of them, which must
    divide n_steps (ValueError if not; no length would split then).
    docs/guide/memory.md compares the libraries and says how to set it.
    """
    pieces = num_minibatches // math.gcd(n_envs, num_minibatches)
    if n_steps % pieces:
        raise ValueError(
            f"Recurrent PPO cannot split {n_envs} envs' rollouts of"
            f" n_steps={n_steps} into {num_minibatches} minibatches of whole"
            f" sequences: each rollout would need {pieces} equal pieces. Choose"
            " n_steps, n_envs and num_minibatches (batch_size when it is 0) so"
            " that they split (docs/guide/memory.md)."
        )
    return n_steps // pieces


def choose_rollout_split(
    agent_config: Any, n_steps: int, n_envs: int, recurrent: bool
) -> RolloutSplit:
    """The rollout's RolloutSplit.

    time only when unroll_length asks for it (n_steps gives
    whole rollouts per env): with as many minibatches as envs, each
    minibatch is then one env's trajectory, its advantages normalised
    within it. It needs fragments that split evenly into more than one
    minibatch: recomputing GAE on a single minibatch, the whole rollout,
    with a critic that just stepped only makes the value targets oscillate
    (the flake of the one-env probing fixture). A recurrent rollout is a
    pool of sequences of bptt_length steps (by default
    default_bptt_length) that must split exactly into the
    minibatches, as Baselines' and CleanRL's recurrent PPO assert: a
    bptt_length that does not divide the default raises ValueError.
    """
    num_minibatches = resolve_num_minibatches(agent_config)
    if recurrent:
        longest = default_bptt_length(n_steps, n_envs, num_minibatches)
        length = agent_config.bptt_length or longest
        if longest % length:
            raise ValueError(
                f"Recurrent PPO minibatches whole sequences: {n_envs} envs'"
                f" rollouts of n_steps={n_steps} cut into sequences of"
                f" bptt_length={length} steps do not split into"
                f" {num_minibatches} minibatches. bptt_length must divide"
                f" {longest}, the longest length that does; None picks it"
                " (docs/guide/memory.md)."
            )
        return RolloutSplit("recurrent", num_minibatches, length)
    unroll = agent_config.unroll_length
    if (
        unroll is not None
        and num_minibatches > 1
        and n_steps % unroll == 0
        and n_steps // unroll * n_envs % num_minibatches == 0
    ):
        return RolloutSplit("time", num_minibatches, unroll)
    return RolloutSplit("flat", num_minibatches, n_steps)


def check_rollout_split(config: Any, n_envs: int, recurrent: bool) -> None:
    """Refuse a bptt_length without memory or not dividing n_steps,
    an unroll_length the update could not use (it would fall back to
    the flat shuffle), and a recurrent rollout whose sequences do not split
    into the minibatches (choose_rollout_split raises)."""
    bptt, n_steps = config.bptt_length, config.n_steps
    if bptt is not None:
        if not recurrent:
            raise ValueError(
                "bptt_length requires a memory config (recurrent PPO);"
                " for feedforward fragment minibatching use unroll_length."
            )
        if n_steps % bptt != 0:
            raise ValueError(f"bptt_length ({bptt}) must divide n_steps ({n_steps}).")
    unroll = config.unroll_length
    if unroll is not None and recurrent:
        raise ValueError(
            "unroll_length is the feedforward fragment length; a recurrent"
            " PPO splits its rollout with bptt_length."
        )
    split = choose_rollout_split(config, n_steps, n_envs, recurrent)
    if unroll is not None and (split.kind, split.length) != ("time", unroll):
        raise ValueError(
            f"unroll_length ({unroll}) must divide n_steps ({n_steps})"
            " into fragments (n_steps // unroll_length * n_envs of them) that"
            " split evenly into num_minibatches"
            f" ({resolve_num_minibatches(config)}) > 1 minibatches."
        )


def resolve_num_minibatches(agent_config: Any) -> int:
    """num_minibatches when positive (brax); otherwise Ajax's legacy
    max(batch_size, n_steps) // min(batch_size, n_steps)."""
    if agent_config.num_minibatches > 0:
        return agent_config.num_minibatches
    sizes = (agent_config.batch_size, agent_config.n_steps)
    return max(sizes) // min(sizes)


def split_fragments(
    batch: Any, perm: jax.Array, num_minibatches: int, length: int
) -> Any:
    """Split a (n_steps, n_envs, ...) rollout into num_minibatches
    minibatches of whole fragments of length steps. Fragment c *
    n_envs + e (chunk c of env e) goes in the order of perm;
    per leaf (num_minibatches, length, fragments_per_minibatch, ...),
    so a scan over the minibatches sees time-major fragments."""
    n_steps, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    fragments_per_env = n_steps // length

    def split(leaf: jax.Array) -> jax.Array:
        # (n_steps, n_envs, ...) -> (fragments_per_env * n_envs, length, ...)
        leaf = leaf.reshape(fragments_per_env, length, n_envs, *leaf.shape[2:])
        leaf = jnp.swapaxes(leaf, 1, 2).reshape(
            fragments_per_env * n_envs, length, *leaf.shape[3:]
        )
        leaf = jnp.take(leaf, perm, axis=0)
        leaf = leaf.reshape(num_minibatches, -1, length, *leaf.shape[2:])
        return jnp.swapaxes(leaf, 1, 2)

    return jax.tree.map(split, batch)


@partial(jax.jit, static_argnames=["num_minibatches", "unroll_length"])
def get_minibatches_preserving_time(
    batch: Any,
    rng: jax.Array,
    num_minibatches: int,
    unroll_length: Optional[int] = None,
) -> Any:
    """Split a (n_steps, n_envs, ...) rollout into minibatches that keep the
    time axis (brax PPO): fragments of unroll_length steps (the whole
    rollout by default), shuffled and split evenly
    (split_fragments). GAE in the loss then scans each fragment and
    bootstraps at its end, brax's per-minibatch (T=unroll_length,
    B=batch_size)."""
    n_steps, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    length = unroll_length or n_steps
    assert n_steps % length == 0, f"n_steps={n_steps} must be a multiple of {length=}."
    n_fragments = n_steps // length * n_envs
    assert n_fragments % num_minibatches == 0, (
        f"{n_fragments} fragments (n_steps={n_steps} / {length=} * n_envs={n_envs})"
        f" do not split into {num_minibatches=}."
    )
    permutation = jax.random.permutation(rng, n_fragments)
    return split_fragments(batch, permutation, num_minibatches, length)


def get_minibatches_from_batch(batch: Any, rng: jax.Array, num_minibatches: int):
    """Split a (n_steps, n_envs, ...) rollout batch into num_minibatches
    minibatches of shape (num_minibatches, n_steps * n_envs / num_minibatches, ...).

    The flatten step joins the leading (n_steps, n_envs) dims into a single
    sample axis regardless of how many trailing feature dims a leaf has,
    so leaves with a feature dim (e.g. observations (n_steps, n_envs, obs_dim))
    and leaves without one (e.g. terminated, truncated of shape
    (n_steps, n_envs)) are reshuffled with the same permutation and stay
    aligned across leaves.
    """
    n_steps, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    total = n_steps * n_envs
    assert total % num_minibatches == 0, (
        "n_steps * n_envs should be a multiple of num_minibatches, got "
        f"n_steps={n_steps}, n_envs={n_envs}, total={total}, {num_minibatches=}"
    )

    # One permutation for every leaf: jax.random.permutation(rng, x, axis=0)
    # per leaf draws the same one, but shuffles a 1-D leaf by sorting it.
    permutation = jax.random.permutation(rng, total)

    def shuffle(leaf: jax.Array) -> jax.Array:
        leaf = jnp.take(leaf.reshape((total,) + leaf.shape[2:]), permutation, axis=0)
        return leaf.reshape((num_minibatches, -1) + leaf.shape[1:])

    return jax.tree.map(shuffle, batch)


def _split_carries(
    carries: jax.Array, permutation: jax.Array, num_minibatches: int
) -> jax.Array:
    """(n_sequences, n_envs, ...) start carries to (num_minibatches,
    n_sequences * n_envs // num_minibatches, ...), in the order of
    permutation (as split_fragments)."""
    carries = carries.reshape(-1, *carries.shape[2:])
    carries = jnp.take(carries, permutation, axis=0)
    return carries.reshape(num_minibatches, -1, *carries.shape[1:])


def sequence_minibatches(
    batch: dict,
    carries: tuple,
    shuffle_key: jax.Array,
    num_minibatches: int,
    length: int,
) -> dict:
    """A recurrent rollout's minibatches of whole sequences, each with the
    actor's and the critic's carry at its start (the critic's ensemble axis
    second)."""
    n_steps, n_envs = batch["obs"].shape[:2]
    permutation = jax.random.permutation(shuffle_key, n_steps // length * n_envs)

    def split(leaf: jax.Array) -> jax.Array:
        return _split_carries(leaf, permutation, num_minibatches)

    actor_starts, critic_starts = carries
    minibatches = split_fragments(batch, permutation, num_minibatches, length)
    minibatches.update(
        {
            "actor_hidden": jax.tree.map(split, actor_starts),
            "critic_hidden": jax.tree.map(
                jax.vmap(split, in_axes=1, out_axes=1), critic_starts
            ),
        }
    )
    return minibatches


def run_epochs(
    agent_state: Any,
    key: jax.Array,
    minibatches: Callable[[jax.Array], Any],
    step: Callable[[Any, Any], tuple[Any, Any]],
    n_epochs: int,
) -> tuple[Any, Any]:
    """n_epochs passes of step(agent_state, minibatch) -> (agent_state,
    aux), each over its own partition minibatches(epoch_key) of the
    rollout (minibatch axis leading); the metrics averaged over every epoch
    and minibatch.

    Each epoch reshuffles, as the PPO references do: OpenAI baselines' ppo2
    and CleanRL's ppo.py shuffle the sample indices inside their epoch
    loop, and APO's official code (xtma/apo,
    apo/algos/apg/appo.py, APPO.optimize_agent) draws
    iterate_mb_idxs(..., shuffle=True) once per epoch.
    """

    def epoch(agent_state: Any, epoch_key: jax.Array) -> tuple[Any, Any]:
        return jax.lax.scan(step, agent_state, minibatches(epoch_key))

    epoch_keys = jax.random.split(key, n_epochs)
    agent_state, aux = jax.lax.scan(epoch, agent_state, epoch_keys)
    return agent_state, jax.tree.map(jnp.mean, aux)


def get_actions_and_log_probs(rollout: Transition) -> tuple[jax.Array, jax.Array]:
    """The rollout's actions and their log-probabilities, each with a
    trailing axis: a discrete action gains one, a continuous action's
    per-dimension log-probabilities are summed."""
    assert rollout.log_prob is not None  # an on-policy rollout carries it
    action, log_prob = rollout.action, rollout.log_prob
    actions = jnp.expand_dims(action, -1) if jnp.ndim(action) < 3 else action
    log_probs = (
        jnp.expand_dims(log_prob, -1)
        if jnp.ndim(log_prob) < 3
        else log_prob.sum(-1, keepdims=True)
    )
    return actions, log_probs


__getattr__ = moved_names(
    __name__,
    {
        "Geometry": "ajax.agents.minibatch:RolloutSplit",
        "minibatch_geometry": "ajax.agents.minibatch:choose_rollout_split",
        "rollout_actions": "ajax.agents.minibatch:get_actions_and_log_probs",
    },
)
