"""Planted faults for measuring what the probes catch (the kill matrix).

Each mutant is one textual change to ``src/ajax``: a fault class from the
audit's mutation catalogue (ids B03, S01 ...) or a replay of a historical
bug (H-prefixed, by the commit that fixed it). ``run`` copies the
repository to a scratch directory, applies one mutant, and runs the chosen
pytest selection against the copy, so the checkout is never modified.

    JAX_PLATFORMS=cpu poetry run python -m tests.probing.mutants OUT_DIR \\
        MUTANT_ID "pytest args"

The exit code of the selection says whether it caught the mutant (non-zero)
or not (zero). S06 is an equivalent mutant (the target cannot carry a
gradient where it is computed): every selection must let it survive.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COPIED = ("src", "tests", "benchmarks", "docs", "pyproject.toml")


@dataclasses.dataclass(frozen=True)
class Mutant:
    id: str
    agents: tuple[str, ...]
    path: str
    old: str
    new: str
    fault: str


MUTANTS = {
    m.id: m
    for m in (
        Mutant(
            "B03",
            ("PPO", "PQN"),
            "src/ajax/environments/interaction.py",
            "        terminated=terminated[:, None],\n"
            "        truncated=truncated[:, None],\n"
            "        raw_obs=raw_obs,\n",
            "        terminated=truncated[:, None],\n"
            "        truncated=terminated[:, None],\n"
            "        raw_obs=raw_obs,\n",
            "on-policy transitions swap termination and truncation",
        ),
        Mutant(
            "S01",
            ("SAC",),
            "src/ajax/agents/SAC/core.py",
            "target = rewards + gamma * (1.0 - dones) * (min_q_target",
            "target = rewards + gamma * (min_q_target",
            "SAC target bootstraps through terminations",
        ),
        Mutant(
            "S02",
            ("SAC",),
            "src/ajax/agents/SAC/train_SAC.py",
            "dones = jnp.logical_or(transition.terminated, transition.truncated)",
            "dones = transition.truncated",
            "SAC ignores terminations",
        ),
        Mutant(
            "S06",
            ("SAC",),
            "src/ajax/agents/SAC/core.py",
            "    return jax.lax.stop_gradient(target)\n",
            "    return target\n",
            "no stop_gradient on the SAC target (equivalent: must survive)",
        ),
        Mutant(
            "P04",
            ("PPO",),
            "src/ajax/agents/PPO/train_PPO.py",
            "ratio = jnp.exp(new_log_probs - log_probs)",
            "ratio = jnp.exp(log_probs - new_log_probs)",
            "PPO probability ratio inverted",
        ),
        Mutant(
            "P09",
            ("PPO",),
            "src/ajax/agents/PPO/train_PPO.py",
            "                critic_params=agent_state.critic_state.params,\n"
            "                x=transition.next_obs,\n",
            "                critic_params=agent_state.critic_state.params,\n"
            "                x=transition.obs,\n",
            "PPO bootstraps GAE from the current observation",
        ),
        Mutant(
            "D05",
            ("DQN",),
            "src/ajax/agents/DQN/train_DQN.py",
            "dones = jnp.logical_or(terminated, truncated).astype(jnp.float32)",
            "dones = truncated.astype(jnp.float32)",
            "DQN ignores terminations",
        ),
        Mutant(
            "Q03",
            ("PQN",),
            "src/ajax/agents/PQN/utils.py",
            "v_next = non_terminal * q_next",
            "v_next = q_next",
            "PQN bootstraps through terminations",
        ),
        Mutant(
            "H876cc75",
            ("PPO",),
            "src/ajax/agents/PPO/train_PPO.py",
            "    elif num_minibatches <= 1:\n        use_brax_faithful_mb = False\n",
            "    elif False:\n        use_brax_faithful_mb = False\n",
            "replay of 876cc75: per-minibatch GAE recompute with one minibatch",
        ),
        Mutant(
            "H0382f32",
            ("SAC", "DQN"),
            "src/ajax/environments/interaction.py",
            '            "terminated": terminated[:, None],\n'
            '            "truncated": truncated[:, None],\n',
            '            "terminated": agent_state.collector_state.last_terminated[:, None],\n'
            '            "truncated": agent_state.collector_state.last_truncated[:, None],\n',
            "replay of 0382f32: the buffer stores the previous step's done flags",
        ),
    )
}


def apply(mutant: Mutant, root: str) -> None:
    path = os.path.join(root, mutant.path)
    with open(path) as fh:
        text = fh.read()
    if text.count(mutant.old) != 1:
        raise ValueError(
            f"{mutant.id}: expected one match in {mutant.path}, "
            f"found {text.count(mutant.old)}"
        )
    with open(path, "w") as fh:
        fh.write(text.replace(mutant.old, mutant.new))


def run(mutant_id: str, pytest_args: list[str], scratch: str) -> int:
    """Run ``pytest_args`` against a copy of the repository with one mutant
    applied (``mutant_id`` "none" for the clean copy)."""
    root = os.path.join(scratch, mutant_id)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for name in COPIED:
        src = os.path.join(REPO, name)
        if os.path.isdir(src):
            shutil.copytree(
                src,
                os.path.join(root, name),
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            shutil.copy2(src, root)
    if mutant_id != "none":
        apply(MUTANTS[mutant_id], root)
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(root, "src"),
        AJAX_NO_COMPILE_CACHE="1",
        JAX_PLATFORMS="cpu",
    )
    # The copy's ajax must be the one imported, not the editable install.
    check = subprocess.run(
        [sys.executable, "-c", "import ajax, sys; print(ajax.__file__)"],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    if not check.stdout.strip().startswith(root):
        raise RuntimeError(f"imported {check.stdout.strip()}, not the copy")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args],
        env=env,
        cwd=root,
    ).returncode


if __name__ == "__main__":
    sys.exit(run(sys.argv[2], sys.argv[3].split(), sys.argv[1]))
