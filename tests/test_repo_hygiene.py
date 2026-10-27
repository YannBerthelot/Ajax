"""Repository-level invariants that CI must enforce: filesystem case,
which a Linux-only CI is otherwise blind to, the test layout, and no new
random key handed out by its turn."""

import ast
import subprocess
from collections import defaultdict
from pathlib import Path


def _tracked() -> list[str]:
    """The paths git tracks, relative to the repository root."""
    return subprocess.run(
        ["git", "ls-files"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


def test_no_case_insensitive_path_collisions():
    """No two tracked paths may differ only by case.

    On case-insensitive filesystems (macOS APFS default, Windows NTFS)
    such paths collapse onto a single file at checkout, silently
    corrupting the working tree. This happened once with
    ``SAC/SAC.py`` vs ``SAC/sac.py`` (PR #32) — Linux CI cannot see it.
    """
    by_folded = defaultdict(list)
    for path in _tracked():
        by_folded[path.lower()].append(path)

    collisions = {k: v for k, v in by_folded.items() if len(v) > 1}
    assert not collisions, (
        f"Tracked paths collide on case-insensitive filesystems: {collisions}"
    )


def test_every_test_directory_is_a_package():
    """Every directory of ``tests/`` holding Python files is a package, so
    pytest imports each module as ``tests.<dir>.<module>``, the name shared
    helpers and the probing scripts import it by. pytest puts a directory
    without ``__init__.py`` on ``sys.path`` itself, and its modules become
    top-level names that can collide (two ``common.py``)."""
    tracked = _tracked()
    missing = sorted(
        {
            str(Path(path).parent)
            for path in tracked
            if path.startswith("tests/") and path.endswith(".py")
        }.difference(
            {
                str(Path(path).parent)
                for path in tracked
                if path.endswith("/__init__.py")
            }
        )
    )
    assert not missing, f"test directories without __init__.py: {missing}"


#: Keys handed out by position in src/ajax, random_keys.py aside: every
#: jax.random.split, split_carry_first and get_keys_and_new_state call but a
#: stream's single advance (one key after the carry). A step following the
#: rule adds none; removing one lowers this count.
KEYS_BY_POSITION = 95


def _key_calls() -> list[tuple[str, str, ast.Call]]:
    """(file:line, the called name, the call) of every call in src/ajax."""
    root = Path(__file__).resolve().parents[1] / "src" / "ajax"
    paths = [path for path in root.rglob("*.py") if path.name != "random_keys.py"]
    return [
        (f"{path.relative_to(root)}:{node.lineno}", ast.unparse(node.func), node)
        for path in sorted(paths)
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
    ]


def test_no_key_is_folded_in_at_a_literal_index():
    """fold_in(key, 3) is split(key, n)[3] under partitionable threefry."""
    folds = [
        where
        for where, called, call in _key_calls()
        if called == "jax.random.fold_in" and isinstance(call.args[1], ast.Constant)
    ]
    assert not folds, f"Derive these keys by name (derive_key) instead: {folds}"


def test_no_new_key_is_handed_out_by_position():
    """A ratchet: the count may only go down (CONTRIBUTING.md, "Random keys")."""
    found = []
    for where, called, call in _key_calls():
        name = called.rsplit(".", 1)[-1]
        counts = [keyword.value for keyword in call.keywords if keyword.arg == "n"]
        counts.extend(call.args[1:] if name == "split_carry_first" else call.args)
        # get_keys_and_new_state's default count is one key.
        one = ast.Constant(1) if name == "get_keys_and_new_state" else None
        count = (counts or [one])[0]
        single = isinstance(count, ast.Constant) and count.value == 1
        advance = name in ("split_carry_first", "get_keys_and_new_state")
        if called == "jax.random.split" or (advance and not single):
            found.append(where)
    assert len(found) <= KEYS_BY_POSITION, (
        "Advance a stream with split_carry_first(rng, 1) and derive its other"
        f" keys by name (derive_key) or by index (fold_index): {found}"
    )
    assert len(found) == KEYS_BY_POSITION, f"Lower KEYS_BY_POSITION to {len(found)}."
