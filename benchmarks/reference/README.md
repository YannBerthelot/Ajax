# Ajax against reference implementations

Does an Ajax agent learn like a well-known implementation of the same
algorithm? Each comparison (`refcmp_specs.py`) trains one Ajax agent and one
reference on the same task and budget, then compares their per-seed
evaluation curves. It runs on demand, not in CI.

| Ajax | Reference | Task | Budget |
|---|---|---|---|
| PPO | Stable-Baselines3 | CartPole-v1 | 200,704 steps |
| DQN | Stable-Baselines3 | CartPole-v1 | 100,000 steps |
| SAC | Stable-Baselines3 | Pendulum-v1 | 20,000 steps |
| TD3 | Stable-Baselines3 | Pendulum-v1 | 20,000 steps |
| PQN | purejaxql, the PQN authors' code (`vendor/`) | CartPole-v1 | 491,520 steps |
| APG | Brax's APG trainer, on gymnax Pendulum | Pendulum-v1 | 2,048,000 steps |

## Two arms

- **matched**: Ajax set to the reference's conventions. A gap here points at
  Ajax's maths, environment or evaluation, or at one of the differences no
  Ajax option can match, which each spec lists under `unmatched`.
- **defaults**: Ajax as shipped. A gap here is a difference of convention,
  to review and document, not a bug by itself.

`solved` (475 on CartPole, -200 on Pendulum) was fixed before any run.
`refcmp_report.py` counts the seeds solved at the final evaluation and how
steadily each seed stays solved over the last third of the run, and tests
both against the reference (its docstring has the statistics).

## Running it

The references do not depend on Ajax, so they are recorded once and kept in
`results/`; only the Ajax arms are re-run, after a change to an agent.
From the repository root:

```bash
benchmarks/reference/record.sh ajax SAC          # both Ajax arms, 16 seeds
uv run python benchmarks/reference/refcmp_report.py
```

A re-run replaces the seeds it records. To record a reference again (a new
SB3 version, a changed spec), pass a Python environment that has
stable-baselines3 for the SB3 ones:

```bash
benchmarks/reference/record.sh reference SAC 16 /path/to/python
benchmarks/reference/record.sh reference PQN     # purejaxql and Brax run in Ajax's env
```

## Cost (Apple M4 Pro, CPU)

Measured on 2026-10-17, 16 seeds. The SB3 references ran as 8
single-thread processes; JAX runs use every core.

| Comparison | Reference, recorded once | Ajax, both arms |
|---|---|---|
| PPO | 39 s (5 CPU-min) | 89 s (3.5 CPU-min) |
| DQN | 114 s (15 CPU-min) | 460 s (10 CPU-min) |
| SAC | 171 s (21 CPU-min) | 814 s (81 CPU-min) |
| TD3 | 182 s (21 CPU-min) | 762 s (85 CPU-min) |
| PQN | 139 s (6 CPU-min) | 162 s (7 CPU-min) |
| APG | 69 s (2 CPU-min) | 170 s (4 CPU-min) |

Re-checking one agent's matched arm takes from 30 s (APG) to 9-10 min
(SAC, TD3). On a CPU, SAC and TD3 with one environment cost Ajax about four
times SB3's CPU time here: their training time grows almost linearly with
the number of seeds, where the on-policy agents' barely grows. Eight seeds halve the Ajax side; the report asks for
16 only when 8 leave the verdict unclear.

## Results

Recorded on 2026-10-17: Ajax at e96ba15, SB3 2.9.0 with torch 2.13.0,
purejaxql at 47af6d7, Brax 0.14.2.

| Agent | Arm | Seeds | Solved | Held | Final | p(solved) | p(held) | Verdict |
|---|---|---|---|---|---|---|---|---|
| PPO | sb3 | 16 | 16/16 | 0.95 | 500.0 | | | |
| PPO | Ajax matched | 16 | 14/16 | 0.91 | 468.2 | 0.48 | 0.47 | match |
| PPO | Ajax defaults | 16 | 10/16 | 0.66 | 438.1 | 0.018 | 0.0012 | gap |
| DQN | sb3 | 16 | 4/16 | 0.30 | 244.2 | | | |
| DQN | Ajax matched | 16 | 3/16 | 0.15 | 182.6 | 1 | 0.074 | match |
| DQN | Ajax defaults | 16 | 14/16 | 0.76 | 473.5 | 0.00097 | 0.0002 | Ajax better |
| SAC | sb3 | 16 | 16/16 | 0.97 | -141.0 | | | |
| SAC | Ajax matched | 16 | 16/16 | 0.98 | -145.7 | 1 | 0.99 | match |
| SAC | Ajax defaults | 16 | 15/16 | 0.95 | -153.8 | 1 | 0.74 | match |
| TD3 | sb3 | 16 | 16/16 | 0.95 | -144.0 | | | |
| TD3 | Ajax matched | 16 | 16/16 | 0.99 | -143.4 | 1 | 0.17 | match |
| TD3 | Ajax defaults | 16 | 15/16 | 0.89 | -172.5 | 1 | 0.21 | match |
| PQN | purejaxql | 16 | 16/16 | 0.95 | 500.0 | | | |
| PQN | Ajax matched | 16 | 16/16 | 0.94 | 500.0 | 1 | 1 | match |
| PQN | Ajax defaults | 16 | 14/16 | 0.89 | 484.1 | 0.48 | 0.49 | match |
| APG | brax | 16 | 1/16 | 0.06 | -543.8 | | | |
| APG | Ajax matched | 16 | 2/16 | 0.12 | -466.1 | 1 | - | match |
| APG | Ajax defaults | 16 | 4/16 | 0.25 | -367.8 | 0.33 | - | match |

- Every matched arm matches its reference. For PPO, that takes the flat
  minibatch geometry of #146 and a value weight of 1.0 (Ajax's value loss
  carries a 0.5 that SB3's does not).
- PPO's defaults differ from SB3's in ways that matter on CartPole: 10/16
  solved against 16/16. The matched arm shows they are conventions, not a
  bug; #147 (the logits head's initial gain) is one of them.
- DQN and APG have little power: their references are unstable themselves
  under these settings (SB3's DQN holds 475 on 4/16 seeds at the end,
  after reaching it on 15/16; Brax's APG solves 1/16). Ajax's own DQN
  defaults hold CartPole better than SB3 with the matched settings.
