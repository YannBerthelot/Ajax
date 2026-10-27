"""The key helpers (ajax.random_keys): keys built by fold_in alone, equal to
the splits they stand for under JAX's default partitionable threefry, the
same under either setting of that flag, and named by tags that no split or
inner loop hands out."""

import jax
import numpy as np
import pytest

from ajax import random_keys
from ajax.random_keys import (
    EXPERT_PREFILL_KEY_TAG,
    EXTENSIONS_KEY_TAG,
    LEGACY_SMALL_TAGS,
    NAMED_TAG_BIT,
    TAGS,
    derive_key,
    fold_index,
    make_tag,
    split_by_index,
    split_carry_first,
)

KEY = jax.random.PRNGKey(3)


def test_split_carry_first_is_a_split_with_the_carry_first() -> None:
    keys = np.stack(split_carry_first(KEY, 3))
    np.testing.assert_array_equal(keys, jax.random.split(KEY, 4))


def test_one_more_key_moves_neither_the_carry_nor_the_keys_before_it() -> None:
    fewer, more = split_carry_first(KEY, 2), split_carry_first(KEY, 3)
    np.testing.assert_array_equal(np.stack(fewer), np.stack(more[:3]))


def test_the_keys_of_an_inner_loop_are_a_split_and_fold_their_index() -> None:
    keys = split_by_index(KEY, 5)
    np.testing.assert_array_equal(keys, jax.random.split(KEY, 5))
    np.testing.assert_array_equal(keys[3], fold_index(KEY, 3))
    np.testing.assert_array_equal(split_by_index(KEY, 8)[:5], keys)


def test_the_keys_do_not_depend_on_the_partitionable_flag(monkeypatch) -> None:
    monkeypatch.setitem(TAGS, "probe", make_tag("probe"))

    def every_key() -> np.ndarray:
        return np.concatenate(
            [
                np.stack(split_carry_first(KEY, 2)),
                split_by_index(KEY, 3),
                np.asarray(derive_key(KEY, "probe"))[None],
            ]
        )

    with jax.threefry_partitionable(True):
        on = every_key()
    with jax.threefry_partitionable(False):
        off = every_key()
    np.testing.assert_array_equal(on, off)


def test_a_named_key_is_no_key_a_split_hands_out(monkeypatch) -> None:
    monkeypatch.setitem(TAGS, "probe", make_tag("probe"))
    named = np.asarray(derive_key(KEY, "probe"))
    assert not (np.asarray(jax.random.split(KEY, 256)) == named).all(axis=1).any()


def test_a_key_is_derived_only_by_a_registered_name() -> None:
    with pytest.raises(KeyError):
        derive_key(KEY, "an unregistered name")


def test_a_tag_is_the_names_crc32_with_the_top_bit_set() -> None:
    """The same in every process: a value pinned here, not Python's hash."""
    assert make_tag("collector") == 0xCEDBF54C
    assert make_tag("") == NAMED_TAG_BIT


def test_the_registry_holds_each_tag_once_past_every_index() -> None:
    """Named tags are distinct, lie past any split's index, and meet neither
    tag kept from before the registry; the legacy small tags are indices a
    split also hands out, listed until their sites take names."""
    assert TAGS == {name: make_tag(name) for name in random_keys.NAMES}
    named = list(TAGS.values())
    assert len(set(named)) == len(named)
    assert all(tag >= NAMED_TAG_BIT for tag in named)
    kept = {EXTENSIONS_KEY_TAG, EXPERT_PREFILL_KEY_TAG}
    assert len(kept) == 2 and not kept.intersection(named)
    assert all(tag >= 2**20 for tag in kept)  # past any realistic split
    assert all(tag < 16 for tag in LEGACY_SMALL_TAGS.values())


def test_the_tags_kept_from_before_hold_their_values() -> None:
    """Moved here unchanged: the keys they derive, and every run, stay."""
    from ajax.agents import update_steps
    from ajax.extensions import pretrain

    assert update_steps.EXTENSIONS_KEY_TAG == EXTENSIONS_KEY_TAG == 0x45585453
    assert pretrain.EXPERT_PREFILL_KEY_TAG == EXPERT_PREFILL_KEY_TAG == 0x70726566
