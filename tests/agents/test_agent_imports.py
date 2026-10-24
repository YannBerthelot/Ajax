"""Agents import each other only along a lineage (owner, 2026-10-24): an
agent imports from another only when it is a variant of that agent's
algorithm, its PARENT, and then only the parent's EQUATIONS; plumbing and
components agents share live in shared modules, and a shared module (any
module outside an agent's package) imports no agent. The lineages and the
equations are the cross-agent import study's verdicts (B1 of the agent-file
restructure).

BORROWED lists the imports that break the rule today, each with the batch
that ends it. The test fails on a new one, and on a listed one that is gone
(strike it), so the list only shrinks."""

from __future__ import annotations

import ast
from pathlib import Path

import ajax

SRC = Path(ajax.__file__).parent


def agents(src: Path = SRC) -> list[str]:
    return sorted(
        p.name for p in (src / "agents").iterdir() if (p / "__init__.py").is_file()
    )


AGENTS = agents()
#: Each descendant's parent: the algorithm it is a variant of.
PARENT = {
    "APO": "PPO",  # average-reward PPO (Ma et al., 2021)
    "ASAC": "SAC",  # average-reward SAC
    "AVG": "SAC",  # incremental SAC, its temperature fixed (Vasan et al., 2024)
    "CrossQ": "SAC",  # SAC without targets, batch-renormalised (Bhatt et al., 2024)
    "REDQ": "SAC",  # an ensemble of SAC's critics (Chen et al., 2021)
    "TQC": "SAC",  # SAC with truncated quantile critics (Kuznetsov et al., 2020)
}
#: What a parent's paper defines, which its descendants may import.
EQUATIONS = {
    "PPO": {"clipped_surrogate", "_compute_gae"},
    "SAC": {
        "sample_next_actions",
        "compute_td_target",
        "soft_bellman_target",
        "soft_policy_loss",
        "soft_policy_objective",
        "temperature_loss_fn",
        "alpha_temperature_loss_fn",
    },
}
# B2 declared the replay family on IQL, CrossQ and TD3 only: SAC's descendants
# join it, and leave its state and init, when B4 declares them.
_SAC_FAMILY_INIT = "B4 (the replay family's init, each agent's networks)"
_SAC_STEPS = "B4 (steps.actor_step or gradient_step over SAC's equations)"
_SAC_METRICS = "B4 (the agent's own metrics)"
#: (importer, agent, name) -> the batch that ends the import. An importer is
#: an agent, or a shared module's path under src/ajax.
BORROWED = {
    # B3 declared PPO and APO; PQN joins the on-policy family next.
    ("PQN", "DQN", "init_DQN"): "B3 (the on-policy family's init)",
    ("PQN", "DQN", "AuxiliaryLogs"): "B3 (PQN's own metrics)",
    ("PQN", "DQN", "PolicyAuxiliaries"): "B3 (PQN's own metrics)",
    **{
        (agent, "SAC", "SoftACState"): "B4 (the replay family's state)"
        for agent in ("ASAC", "AVG", "REDQ", "TQC")
    },
    **{
        (agent, "SAC", "SACConfig"): "B4 (declared hyperparameters)"
        for agent in ("REDQ", "TQC")
    },
    **{
        (agent, "SAC", name): _SAC_FAMILY_INIT
        for agent in ("ASAC", "AVG", "REDQ", "TQC")
        for name in ("init_soft_actor_critic", "create_alpha_train_state")
    },
    **{
        (agent, "SAC", name): _SAC_STEPS
        for agent in ("ASAC", "REDQ", "TQC")
        for name in ("soft_actor_step", "update_temperature")
    },
    # CrossQ, declared in B2, keeps SAC's temperature until B4 moves it to a
    # shared module: steps.gradient_step would add optax.global_norm to its
    # program, so B2 has no behaviour-neutral switch.
    **{
        ("CrossQ", "SAC", name): "B4 (the temperature and its state, shared)"
        for name in ("SoftACState", "create_alpha_train_state", "update_temperature")
    },
    **{
        (agent, "SAC", "TemperatureAuxiliaries"): _SAC_METRICS
        for agent in ("ASAC", "AVG", "CrossQ", "REDQ", "TQC")
    },
    ("checkpoint.py", "UDRL", "SegmentBuffer"): "2026-11-17 (the shim's removal)",
    ("extensions/instrumentation.py", "SAC", "sample_next_actions"): (
        "B4 (the agent supplies its TD target)"
    ),
}


def cross_agent_imports(src: Path = SRC) -> set[tuple[str, str, str]]:
    """Every (importer, agent, name) an agent's or a shared module imports
    from another agent, a module import counted once per attribute it reads.
    The package's ``ajax/__init__.py``, the public API, is left out."""
    names = agents(src)

    def agent_of(module: str) -> str | None:
        parts = module.split(".")
        return (
            parts[2]
            if parts[:2] == ["ajax", "agents"] and parts[2:3] and parts[2] in names
            else None
        )

    found = set()
    for path in src.rglob("*.py"):
        rel = path.relative_to(src)
        if rel == Path("__init__.py"):
            continue
        own = rel.parts[1] if rel.parts[0] == "agents" and len(rel.parts) > 2 else None
        importer = own if own in names else rel.as_posix()
        tree = ast.parse(path.read_text())
        modules: dict[str, str] = {}  # a local name bound to an agent's module
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                agent = agent_of(node.module) or agent_of(full)
                if agent is None or agent == importer:
                    continue
                if (src.parent / f"{full.replace('.', '/')}.py").is_file():
                    modules[alias.asname or alias.name] = agent
                else:
                    found.add((importer, agent, alias.name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in modules:
                    found.add((importer, modules[node.value.id], node.attr))
    return found


def _allowed(importer: str, agent: str, name: str) -> bool:
    return PARENT.get(importer) == agent and name in EQUATIONS[agent]


def test_agents_import_each_other_only_along_a_lineage() -> None:
    breaking = {i for i in cross_agent_imports() if not _allowed(*i)}
    assert not breaking - BORROWED.keys(), (
        "imports from another agent that are not its parent's equations: move"
        " what is shared to a shared module",
        sorted(breaking - BORROWED.keys()),
    )
    assert not BORROWED.keys() - breaking, (
        "gone from the code: strike them from BORROWED",
        sorted(BORROWED.keys() - breaking),
    )


def test_every_parent_and_equation_named_exists() -> None:
    """A renamed equation or agent would leave a stale entry that allows
    nothing."""
    for child, parent in PARENT.items():
        assert {child, parent} <= set(AGENTS), (child, parent)
    for parent, names in EQUATIONS.items():
        text = "".join(p.read_text() for p in (SRC / "agents" / parent).rglob("*.py"))
        defined = {
            node.name
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }
        assert names <= defined, (parent, sorted(names - defined))


def test_the_scan_finds_name_module_and_shared_imports(tmp_path: Path) -> None:
    src = tmp_path / "ajax"
    files = {
        "__init__.py": "from ajax.agents.B.B import B\n",
        "agents/__init__.py": "",
        "agents/A/__init__.py": "",
        "agents/A/a.py": "from ajax.agents.B import core\nfrom ajax.agents.A.x import y\n"
        "def f():\n    from ajax.agents.B.core import g\n    return core.h, core.k\n",
        "agents/B/__init__.py": "",
        "agents/B/core.py": "",
        "agents/B/B.py": "",
        "shared.py": "from ajax.agents.A.a import f\n",
    }
    for name, text in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(text)
    assert cross_agent_imports(src) == {
        ("A", "B", "g"),
        ("A", "B", "h"),
        ("A", "B", "k"),
        ("shared.py", "A", "f"),
    }
