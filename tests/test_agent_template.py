"""The agent template's calls match the loop's and the steps' signatures.

CONTRIBUTING.md and the architecture page show how an agent calls the shared
loop. When a signature changes and they do not, a newcomer's first agent
fails to build. Every call to a loop helper in their Python blocks and inline
code is bound to the helper's signature (a trailing ``...`` elides the rest),
and a result unpacked into a tuple must have the helper's arity.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import textwrap
import typing

import pytest

from ajax.agents import loop, steps

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ("CONTRIBUTING.md", "docs/developer/architecture.md")
HELPERS = {
    "off_policy": loop.TrainLoop.off_policy,
    "on_policy": loop.TrainLoop.on_policy,
    "critic_step": steps.critic_step,
    "actor_step": steps.actor_step,
    "gradient_step": steps.gradient_step,
    "compute_gradients": steps.compute_gradients,
    "apply_gradients_clipped_together": steps.apply_gradients_clipped_together,
}
METHODS = {"off_policy", "on_policy"}
CODE = re.compile(r"^```python\n(.*?)^```|`([^`\n]+)`", re.MULTILINE | re.DOTALL)


def _calls(doc: str) -> list:
    """``(helper, call, tuple_targets)`` for every helper call in ``doc``'s
    code that parses (``tuple_targets``: ``None`` unless unpacked)."""
    found = []
    for block, inline in CODE.findall((ROOT / doc).read_text()):
        try:
            tree = ast.parse(textwrap.dedent(block or inline))
        except SyntaxError:
            continue
        unpacked = {
            id(node.value): len(node.targets[0].elts)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name in HELPERS:
                found.append((name, node, unpacked.get(id(node))))
    return found


def _arity(fn: typing.Callable) -> int | None:
    returns = typing.get_type_hints(fn).get("return")
    return (
        len(typing.get_args(returns)) if typing.get_origin(returns) is tuple else None
    )


@pytest.mark.parametrize("doc", DOCS)
def test_template_calls_bind(doc):
    calls = _calls(doc)
    for name, call, targets in calls:
        args = list(call.args)
        elided = bool(args) and getattr(args[-1], "value", None) is Ellipsis
        if elided:
            args = args[:-1]
        assert not any(isinstance(a, ast.Starred) for a in args), ast.unparse(call)
        positional = [None] * (len(args) + (name in METHODS))  # + self
        keywords = {k.arg: None for k in call.keywords if k.arg is not None}
        signature = inspect.signature(HELPERS[name])
        bind = signature.bind_partial if elided else signature.bind
        try:
            bind(*positional, **keywords)
        except TypeError as error:
            pytest.fail(f"{doc}: {ast.unparse(call)} does not bind: {error}")
        if targets is not None:
            assert targets == _arity(HELPERS[name]), (
                f"{doc}: {ast.unparse(call)} unpacks {targets} values,"
                f" {name} returns {_arity(HELPERS[name])}"
            )


def test_template_shows_the_helpers():
    """The template still calls what it documents (else the check above
    passes on nothing)."""
    shown = {name for name, _, _ in _calls("CONTRIBUTING.md")}
    assert {"off_policy", "on_policy", "critic_step", "actor_step"} <= shown
