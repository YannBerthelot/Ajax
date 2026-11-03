# Agent benchmark, phase 0: timings on the cloud machine

Each agent was timed with `agent_suite.py time LABEL`, one agent at a time with
nothing else running, on 2 and 3 November 2026, from main at `628adbb` (#330).
Each agent's raw lines are in `timing/<LABEL>.jsonl`.

## The machine

4 cores (Intel Xeon @ 2.10 GHz), 15.7 GB of memory, no GPU (JAX 0.11.2 on
the CPU), 30 GB of free disk.

## Pause test: background work kept running

While the session sat idle, a background loop wrote the time and the machine's
uptime every 30 s, 80 times (`pause-test.txt`). The repository ignores
`*.txt`, so that file was added with `git add -f`.

- The whole span is 2,371 s of wall time and 2,370.55 s of uptime.
- The largest gap between two stamps is 31 s, with an uptime gap of 30.00 s
  over it. The other 78 gaps are 30 s.

Verdict: the machine never paused, and background work went on while the
session was idle.

## Timings

| Agent | Benchmark | Compile s | Steps/s, 1 seed | Steps/s per seed, 10 seeds | Peak memory GB, 10 seeds | Hours, 10 seeds at once | Hours, 10 seeds one after another |
|---|---|---|---|---|---|---|---|
| PQN | MinAtar | 4.1 | 18,368 | 2,674 | 2.08 | 0.5 | 0.8 |
| DQN | MinAtar | 2.7 | 1,289 | 408 | 0.89 | 3.6 | 11 |
| PPO-MinAtar | MinAtar | 4.5 | 6,183 | 1,137 | 2.08 | 1.3 | 2.4 |
| UDRL | MinAtar | 3.8 | 1,511 | 415 | 0.91 | 3.5 | 9.6 |
| PPO | Cheetah | 24.5 | 291 | 247 | 2.02 | 1.2 | 10 |
| APO | Cheetah | 26.8 | 1,422 | 524 | 2.19 | 0.5 | 2.0 |
| AVG | Cheetah | 21.5 | 283 | 179 | 1.98 | 1.6 | 10 |
| TQC | Cheetah | 22.2 | 29.3 | 3.68 | 2.98 | 79 | 99 |
| SAC | Cheetah | 23.0 | 138 | 32.4 | 2.20 | 9.0 | 21 |
| TD3 | Cheetah | 20.9 | 151 | 45.8 | 2.20 | 6.4 | 19 |
| ASAC | Cheetah | 21.9 | 137 | 37.1 | 2.13 | 7.9 | 21 |
| RVISAC | Cheetah | 20.9 | 144 | 36.2 | 2.12 | 8.0 | 20 |
| CrossQ | Cheetah | 23.0 | 7.14 | 1.15 | 4.65 | 254 | 408 |
| REDQ | Cheetah | 19.3 | 6.95 | 0.80 | 2.55 | 362 | 419 |
| DreamerV3 | Cheetah | 42.2 | 0.78 | out of memory | above 13.3 | — | 3,731 |
| TDMPC2 | Cheetah | 34.7 | 2.98 | timed out | — | — | 977 |

How to read it:

- **Steps.** A step is one unit of the benchmark's budget. For the agents
  that run several envs (PQN 128, PPO-MinAtar 8, APO 10, DreamerV3 16), it
  counts every env's step.
- **Speeds.** Speeds are measured over 4,096 steps (16,384 for PQN and
  PPO-MinAtar, 20,480 for APO), after the agent's updates have started. Compile
  time is taken out and evaluations are off. At 10 seeds, each seed advances
  by the rate shown.
- **Compile.** Compile s is at 1 seed; at 10 seeds it is up to 6.5 s more.
- **Memory.** Peak memory is the process's peak across both runs.
- **Hours.** Hours are the budget divided by the speed: 1,048,576 steps on
  CheetahRun (1,024,000 for APO) and 5,242,880 on Breakout-MinAtar. They use
  the unrounded speed (`measured_steps / exec_s`), so they can differ a little
  from what the rounded `run_steps_per_s` in the files gives (CrossQ: 254 h,
  not 265 h).
- **Left out of the hours.** Compiling, the warm-up and evaluations are not
  counted. On CheetahRun, evaluations add 10 episodes of 1,000 steps every
  32,768 steps, about 30% more env steps. That matters most for the agents
  whose time goes into the env (PPO, APO, AVG).

How long each timing command took: PQN 48 s, DQN 28 s, PPO-MinAtar 65 s,
UDRL 38 s, PPO 199 s, APO 268 s, AVG 191 s, TQC 1,427 s, SAC 360 s, TD3
332 s, ASAC 322 s, RVISAC 330 s, CrossQ 4,646 s, REDQ 5,865 s, DreamerV3
6,800 s and TDMPC2 7,200 s. That is 7.8 hours in all.

## What timed out or failed

- **DreamerV3 ran out of memory at 10 seeds.** Its 1-seed line was written:
  0.78 steps/s, a warm-up of 1,394 s and 4.03 GB of memory. About 2 minutes
  into the 10-seed run, the kernel killed the process for lack of memory
  (exit 137) with 13.3 GB in use of the machine's 15.7 GB. The kernel's last
  line was `Memory cgroup out of memory: Killed process 5163 (python3)
  total-vm:20732576kB, anon-rss:13951568kB`. Before that, the last thing the
  command printed was the 1-seed line. Without the memory kill, the 2-hour
  limit would have come about 400 s later anyway.
- **TDMPC2 timed out at 10 seeds.** Its 1-seed line was written: 2.98 steps/s,
  a warm-up of 964 s and 2.26 GB of memory. The 10-seed run started at 23:19
  UTC and was still running when the command reached its 2-hour limit about
  80 minutes later. It printed nothing after the 1-seed line.

Neither was retried. Every other agent finished both seed counts.

## Which agents fit the plan

Training the 10 seeds at once was faster than running them one after another
for every agent that finished both. On these 4 cores, batching the 10 seeds
gives 1.1 times (REDQ) to 8.5 times (PPO) the throughput of one seed.

| Group | Agents | Hours for 10 seeds at once |
|---|---|---|
| Within one 2-hour command | PQN, APO, PPO, PPO-MinAtar, AVG | 0.5, 0.5, 1.2, 1.3, 1.6 |
| Within two commands | UDRL, DQN | 3.5, 3.6 |
| Four or five 2-hour commands in one session | TD3, ASAC, RVISAC, SAC | 6.4, 7.9, 8.0, 9.0 |
| Too slow on this machine | TQC, CrossQ, REDQ | 79, 254, 362 |

The hours above are before evaluations. For the third group, the driver
restarts a run from step 0 if the machine that kept its checkpoint is
reclaimed.

The five slow agents would be too slow even with one seed per session. A
single seed alone takes:

- TQC: 9.9 hours;
- CrossQ: 41 hours;
- REDQ: 42 hours;
- TDMPC2: 98 hours, with no 10-seed number;
- DreamerV3: 373 hours, and its 10 seeds at once do not fit in memory.
