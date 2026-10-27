"""Random keys by name, not by turn.

An agent keeps two streams of keys: its updates' (the agent state's rng) and
its collection's (the collector state's rng). Each advances once per update
or collection step, its carry first (split_carry_first), so a step asking
for one more key moves no other draw. Every other key of a step is derived
from the step's key by name (derive_key), or by index in an inner loop
(fold_index): a draw that exists only under a flag, an option, a condition,
a count or an extension never advances a stream, and two runs that differ by
it share every other draw (common random numbers: Law and Kelton; streams
and substreams, L'Ecuyer et al. 2002; cited from memory, unverified).

Keys are built with fold_in alone, which does not depend on JAX's
jax_threefry_partitionable flag. Under its default (on), fold_in(key, i) is
split(key, n)[i], so split_carry_first(key, n) is split(key, n + 1) and the
keys of an earlier split are kept bit for bit.
"""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp

#: Set in every name's tag: a split or an inner loop hands out the indices
#: from 0, far below it, so no derived key is a key one of them hands out.
NAMED_TAG_BIT = 0x80000000


def make_tag(name: str) -> int:
    """The tag a key named name is folded with: the name's CRC-32 with the
    top bit set, the same in every process (Python's hash is salted per
    process)."""
    return zlib.crc32(name.encode()) | NAMED_TAG_BIT


#: Every name a key is derived by (derive_key), so that the registry's test
#: sees each tag once.
NAMES: tuple[str, ...] = ()

TAGS: dict[str, int] = {name: make_tag(name) for name in NAMES}

#: A step's extensions' key (ajax.agents.update_steps.derive_extensions_key),
#: at the value it had before the registry: "EXTS".
EXTENSIONS_KEY_TAG = 0x45585453

#: ExpertPrefill's stream off the pretrain key, at the value it had before
#: the registry: "pref". CloningPretrain and MCPretrain roll the expert out
#: on split(key, n), so a small tag would reuse one of their keys.
EXPERT_PREFILL_KEY_TAG = 0x70726566

#: Small tags still folded in where they are, each the key a split of the
#: same key hands out at that index. They stay until the change that moves
#: the runs they key anyway names them.
LEGACY_SMALL_TAGS: dict[str, int] = {
    "APO's minibatch entropy keys (agents/APO/APO.py)": 1,
    "ExpertWarmup's decision (extensions/expert.py)": 1,
    "UDRL's action-stage gate key, split(key, 4)[3] (agents/UDRL/UDRL.py)": 3,
    "TD-MPC2's task embeddings (agents/TDMPC2/core.py)": 1,
    "the observation noise, gymnax and brax (wrappers.py)": 0,
}


def derive_key(key: jax.Array, name: str) -> jax.Array:
    """The key named name of a step on key: no split of key, and no other
    name, hands it out. name must be registered (NAMES)."""
    return jax.random.fold_in(key, TAGS[name])


def split_carry_first(key: jax.Array, n: int) -> tuple[jax.Array, ...]:
    """The carry, then n keys: fold_in(key, 0), then fold_in(key, i) for i
    from 1 to n. A step that asks for one more key keeps the carry and the
    keys before it."""
    return tuple(jax.random.fold_in(key, index) for index in range(n + 1))


def fold_index(key: jax.Array, index: int | jax.Array) -> jax.Array:
    """The key of an inner loop's step index (a scan's, a minibatch's), so
    the loop's length moves no other key."""
    return jax.random.fold_in(key, index)


def split_by_index(key: jax.Array, n: int) -> jax.Array:
    """The keys of steps 0 to n - 1 of an inner loop (fold_index), stacked
    on a leading axis."""
    indices = jnp.arange(n, dtype=jnp.uint32)
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, indices)
