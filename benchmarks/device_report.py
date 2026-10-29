"""Steps per second by agent, device and number of seeds, from agent_bench
rows: the report of benchmarks/gpu_check.sh.

    python benchmarks/device_report.py benchmarks/gpu/*-seeds.jsonl

A cell is the median exec_steps_per_s of its trials, every seed's steps
counted. Below the table, each agent's answer to gpu_check's question: the
seed counts at which the GPU ran more steps per second than the CPU, among
the counts measured on both.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load_medians(paths: list[str]) -> dict[tuple[str, str, str, int], float]:
    """{(agent, backend, device_kind, n_seeds): median exec steps/s}. A
    failed agent's row (an error, no timing) is left out."""
    trials: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "exec_steps_per_s" not in row:
                    continue
                key = (
                    row["agent"],
                    row["jax_backend"],
                    row.get("device_kind", row["jax_backend"]),
                    row["n_seeds"],
                )
                trials[key].append(row["exec_steps_per_s"])
    # The median as agent_bench reports it: the middle trial, the upper one
    # of an even count.
    return {key: sorted(values)[len(values) // 2] for key, values in trials.items()}


def report(medians: dict[tuple[str, str, str, int], float]) -> str:
    agents = sorted({key[0] for key in medians})
    devices = sorted({(key[1], key[2]) for key in medians})
    seed_counts = sorted({key[3] for key in medians})
    header = ["agent", "seeds"]
    for backend, kind in devices:
        header.append(f"{backend} ({kind})")
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for agent in agents:
        for seeds in seed_counts:
            cells = [agent, str(seeds)]
            for backend, kind in devices:
                value = medians.get((agent, backend, kind, seeds))
                cells.append("-" if value is None else f"{value:,.0f}")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for agent in agents:
        faster, slower = [], []
        for seeds in seed_counts:
            gpu = [
                value
                for key, value in medians.items()
                if key[:2] == (agent, "gpu") and key[3] == seeds
            ]
            cpu = [
                value
                for key, value in medians.items()
                if key[:2] == (agent, "cpu") and key[3] == seeds
            ]
            if len(gpu) != 1 or len(cpu) != 1:
                continue  # not measured on both devices
            if gpu[0] > cpu[0]:
                faster.append(str(seeds))
            else:
                slower.append(str(seeds))
        lines.append(
            f"- {agent}: GPU faster at {', '.join(faster) or 'no'} seeds;"
            f" CPU at least as fast at {', '.join(slower) or 'no'} seeds."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(load_medians(sys.argv[1:])))
