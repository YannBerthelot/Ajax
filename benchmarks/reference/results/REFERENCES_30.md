# References at 30 seeds (cloud)

Recorded on 2026-10-17 from main at c011ffb with `record.sh reference
<AGENT> 30`, one agent at a time. All reference seeds 0-29 come from this
machine; the Ajax arms are still the Mac's 16 seeds.
`REFERENCES_30_report.md` holds `refcmp_report.py`'s table.

## Machine

Cloud container: Intel Xeon (family 6, model 173; KVM hides
the name), 4 vCPUs, 15 GiB RAM, Ubuntu 24.04.5 LTS. CPU only.

## Versions

- SB3 venv (Python 3.12.3, outside the repository): stable-baselines3
  2.9.0, torch 2.13.0+cu130, gymnasium 1.1.1.
- Ajax env (`uv sync --locked`): jax 0.7.2, brax 0.14.2; purejaxql
  vendored at 47af6d7.

## Cost

| Reference | Wall | CPU | Mac, 16 seeds |
|---|---|---|---|
| PPO | 354 s | 23 min | 39 s, 5 min |
| DQN | 1,536 s | 99 min | 114 s, 15 min |
| SAC | 1,917 s | 125 min | 171 s, 21 min |
| TD3 | 1,875 s | 122 min | 182 s, 21 min |
| PQN | 407 s | 14 min | 139 s, 6 min |
| APG | 297 s | 7.5 min | 69 s, 2 min |

Total: 106 min wall, 6.5 CPU-hours (user + sys). Per SB3 seed, 2.5-3.5x
the M4 Pro's CPU time; the 8 SB3 processes shared 4 cores.

## Solved at the final evaluation

| Reference | Mac, 16 seeds | Cloud, seeds 0-15 | Cloud, 30 seeds |
|---|---|---|---|
| PPO | 16/16 | 16/16 | 30/30 |
| DQN | 4/16 | 6/16 | 10/30 |
| SAC | 16/16 | 16/16 | 30/30 |
| TD3 | 16/16 | 16/16 | 29/30 |
| PQN | 16/16 | 16/16 | 30/30 |
| APG | 1/16 | 0/16 | 0/30 |

Seeds 0-15 do not reproduce the Mac's curves (other CPU and maths
libraries); only counts compare. Two verdicts change: DQN
matched becomes a gap (p(held) 0.016), APG defaults Ajax better
(p(solved) 0.011).

## What failed or differs

- The network policy blocks download.pytorch.org (proxy 403), so torch
  2.13.0 came from PyPI: the CUDA build, run on the CPU.
- `record.sh` needs zsh, absent from the image; installed with apt.
- The first PQN run was stopped after a minute, before writing a record,
  to run each agent as its own job; it then ran in full.
- Every recording exited cleanly.
