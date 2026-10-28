"""benchmarks/device_report.py, the report of gpu_check.sh.

Contract (its docstring): a cell is the median exec steps/s of an agent's
trials on one device at one seed count; each agent's answer lists the seed
counts at which the GPU ran more steps per second than the CPU, among those
measured on both; a failed agent's row has no timing and is left out.
"""

import json

import device_report


def row(agent, backend, seeds, exec_steps_per_s, kind=None):
    return {
        "agent": agent,
        "jax_backend": backend,
        "device_kind": kind or backend,
        "n_seeds": seeds,
        "exec_steps_per_s": exec_steps_per_s,
    }


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_the_gpu_is_reported_faster_only_where_both_devices_ran(tmp_path):
    gpu = "NVIDIA L4"
    cpu_rows = [
        row("DQN", "cpu", 1, 100.0),
        row("DQN", "cpu", 1, 300.0),
        row("DQN", "cpu", 1, 200.0),  # median of the three: 200
        row("DQN", "cpu", 16, 1000.0),
        {"agent": "SAC", "env": "Pendulum-v1", "error": "RuntimeError: out of memory"},
    ]
    gpu_rows = [
        row("DQN", "gpu", 1, 50.0, gpu),  # slower than the CPU's 200
        row("DQN", "gpu", 16, 4000.0, gpu),  # faster than the CPU's 1000
        row("DQN", "gpu", 128, 20000.0, gpu),  # no CPU run at 128 seeds
    ]
    paths = [
        write(tmp_path / "cpu.jsonl", cpu_rows),
        write(tmp_path / "gpu.jsonl", gpu_rows),
    ]

    text = device_report.report(device_report.load_medians(paths))

    assert "| agent | seeds | cpu (cpu) | gpu (NVIDIA L4) |" in text
    assert "| DQN | 1 | 200 | 50 |" in text
    assert "| DQN | 128 | - | 20,000 |" in text
    assert "SAC" not in text
    assert "- DQN: GPU faster at 16 seeds; CPU at least as fast at 1 seeds." in text
