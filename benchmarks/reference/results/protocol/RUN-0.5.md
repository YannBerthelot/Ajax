# Reproduction 0.5

Four reference comparisons, each run until AdaStop decided, on 2026-11-02,
matched arm only: SAC and TD3 against Stable-Baselines3 on Pendulum (new),
and PPO against Stable-Baselines3 and PQN against purejaxql on CartPole,
both sides trained again. Protocol version 2; SAC, TD3 and PPO at spec
version 2, PQN at version 3. The runs started from 340ed65, main after
#327, which registered SAC's and TD3's margins. Each comparison's records
carry the commit it ran on: 340ed65 for SAC, then the commits adding the
previous comparison's records (da495af, 1adddad, c2b3318). Those commits
change only the records, so the code and specs are the same throughout.
Every batch recorded a clean tree.

## Machine

From the manifests: a cloud container with an Intel Xeon Processor @
2.80GHz, 4 cores and 15.7 GB, running Linux 6.18.44 on x86_64 (glibc 2.39).
**This is not the pilot's machine type**: the pilot ran on an Intel Xeon
Processor @ 2.10GHz, also with 4 cores and 15.7 GB. The speed rows below
are therefore not comparable with the pilot's. For example, SB3's PPO
produced the same 96 curves as in the pilot, value for value, at 7,663
steps per second here against 9,807 there.

Everything ran on the CPU (`JAX_PLATFORMS=cpu`) with Python 3.12.3. Ajax
used jax 0.11.2, flax 0.12.12 and optax 0.2.8. The references used SB3
2.9.0 with torch 2.13.0 (the CUDA build, run on the CPU) and gymnasium
1.3.0, and purejaxql with jax 0.7.2, flax 0.10.7 and chex 0.1.91. Every
side used gymnax 1.0.1.

## Time

Wall time and CPU time (user plus sys, the run and its children) of each
`run` command, measured with bash's `time`. The last two columns are the
sums over each side's 24 batches.

| Comparison | Wall | CPU | Reference (wall / CPU) | Ajax (wall / CPU) |
|---|---|---|---|---|
| SAC | 335.4 min | 1,178.6 min | 194.8 / 757.5 min | 140.5 / 421.0 min |
| TD3 | 363.3 min | 1,278.0 min | 185.2 / 724.5 min | 178.0 / 553.3 min |
| PPO | 51.9 min | 180.6 min | 42.1 / 162.7 min | 9.7 / 17.8 min |
| PQN | 75.5 min | 170.5 min | 44.2 / 94.6 min | 31.2 / 75.8 min |
| Total | 13.8 h | 46.8 h | | |

The task estimated 7 to 10 hours of wall time and 23 to 29 CPU-hours if
every comparison ran to the cap. All four did, and the runs took 13.8 hours
and 46.8 CPU-hours. SAC and TD3 took 5.6 and 6.1 hours each.

`report SAC TD3 PPO PQN` took 7.8 min and `refcmp_publish.py docs` 10.1
min, almost all of it in the resolution simulations.

## Results

AdaStop ran every comparison to its cap: 24 interims, 96 seeds per side.
It detected no difference in any of them. All four verdicts are similar:
each 95% interval lies inside its margin.

| Comparison | Verdict | Margin | Seeds | Reference | Ajax | Difference [95%] | Resolution |
|---|---|---|---|---|---|---|---|
| SAC, SB3, Pendulum | similar | ±14 | 96 | -146.74 | -147.99 | -1.25 [-7.92, +5.35] | 9.36 |
| TD3, SB3, Pendulum | similar | ±14 | 96 | -148.86 | -144.22 | +4.63 [-2.80, +11.97] | 11.85 |
| PPO, SB3, CartPole | similar | ±25 | 96 | 493.85 | 496.71 | +2.86 [-3.72, +10.31] | 2.65 |
| PQN, purejaxql, CartPole | similar | ±25 | 96 | 497.98 | 499.02 | +1.04 [-1.95, +4.29] | 0.011 (see below) |

The table below gives medians over each side's 24 batches, as
`docs/reproduction/results.md` prints them. Steps per second are the
batch's 4 seeds times the steps per seed, divided by the batch's wall time.
SB3 runs one process per seed, 4 at once; Ajax and purejaxql run one
program for all 4 seeds.

| Comparison | Reference steps/s | CPU-s per 1M steps | Compile s | Ajax steps/s | CPU-s per 1M steps | Compile s |
|---|---|---|---|---|---|---|
| SAC | 164 | 23,620 | 4.8 | 227 | 13,242 | 1.4 |
| TD3 | 175 | 22,334 | 4.9 | 179 | 17,257 | 1.5 |
| PPO | 7,663 | 502 | 4.0 | 33,506 | 54 | 1.7 |
| PQN | 17,757 | 121 | 8.3 | 25,134 | 96 | 1.3 |

- **SAC.** 95 of Ajax's 96 seeds end at -200 or above, the level the spec
  counts as solved, against all 96 of SB3's. Ajax's lowest seed scores
  -202.8, SB3's -194.1. Both medians are -145.4. Ajax uses 56% of SB3's
  CPU per step.
- **TD3.** 94 of Ajax's 96 seeds end at -200 or above, against all 96 of
  SB3's. Ajax's lowest seed scores -207.9, SB3's -196.1. The interval's
  upper end, +11.97, is 2.03 inside the margin. Ajax uses 77% of SB3's CPU
  per step.
- **PPO.** 92 of Ajax's 96 seeds end at 500, against 91 of SB3's. Ajax
  uses about a ninth of SB3's CPU per step. In the pilot (spec version 1),
  Ajax scored 485.49, -8.36 [-21.96, +3.73], with 88 seeds at 500.
- **PQN.** 95 of Ajax's 96 seeds end at 500, against 91 of purejaxql's.
  In the pilot (spec version 1), Ajax scored 499.90, +1.92 [-0.05, +4.71].
- **The references reproduced the pilot exactly.** SB3's 96 PPO curves
  and purejaxql's 96 PQN curves are identical, value for value, to the
  pilot's, although gymnax moved from 1.0.0 to 1.0.1, SB3's project now
  resolves gymnasium 1.3.0 instead of 1.4.0, and the CPU differs. Their
  fingerprints changed with their lockfiles, so they were trained again.
- **Resolution at the ceiling.** On PPO, 183 of 192 seeds score 500; on
  PQN, 186 of 192. As the protocol says, the resolution's simulation then
  comes out small (2.65 and 0.011) and says nothing about a gap made of a
  few seeds that fail; the difference's interval is the guide there. SAC's
  and TD3's resolutions, 9.36 and 11.85, are below their margin of 14.

## What failed or was changed

- No comparison failed: all four `run` commands exited 0, and none was
  stopped or redone. Each records file was committed and pushed after its
  comparison. `tests/benchmarks` passes on the records (82 passed; the one
  skip and two expected failures are marked in the code and do not concern
  the records).
- `/usr/bin/time` is not installed on this machine, so the runs were timed
  with bash's `time`.
- Before any run, building the SB3 reference project took 8.6 min, most of
  it downloading torch and its CUDA wheels, and purejaxql's 13 s. No batch
  counts that time.
- **The first `refcmp_publish.py docs` failed** after 87 s with
  `KeyError: 'COMPARISONS'` in `refcmp_protocol.spec_of`. The session's
  clone was shallow (335 commits) and lacked the pilot's commits. The pilot's
  PPO-inverted_pendulum and PQN-Asterix records ran spec version 1, so
  `spec_of` reads their spec from their commit with `git show`, which
  printed nothing; `spec_of` then executed an empty file. `git fetch
  --unshallow origin main` fetched the pilot's commits, which are on main,
  and the second `docs` passed. No file changed. `report SAC TD3 PPO PQN`
  did not need an old commit: those four comparisons' latest records ran
  their current specs.
- **`results.md` names the pilot's machine.** Its line "Machine: Intel(R)
  Xeon(R) Processor @ 2.10GHz, 4 cores" comes from the first record of the
  first records file, PPO's, which is the pilot's. The rows for SAC, TD3,
  PPO and PQN ran on the 2.80GHz machine; PPO-inverted_pendulum's and
  PQN-Asterix's on the pilot's. Its speed table therefore mixes two machine
  types. The line comes from `refcmp_publish.py` and is left as it is.
- `refcmp_publish.py docs` redraws every plot, so the plots of
  PPO-inverted_pendulum and PQN-Asterix changed although their records did
  not: matplotlib 3.11.2 instead of 3.10.1, and a new date stamp.
- Main moved during the runs (#328, #329, #330). None of those commits
  touches `src/ajax`, `uv.lock`, `benchmarks/reference` or the specs, so
  the records match main's code.
- The pull request merges with a merge commit, not a squash, at the
  owner's choice: the records of TD3, PPO and PQN name commits of this
  branch (da495af, 1adddad, c2b3318), and `spec_of` reads a spec from the
  commit a record names. A merge commit keeps those commits on main, as it
  did the pilot's.
- Not run: DQN and APG, which wait for longer specs. PPO-inverted_pendulum
  and PQN-Asterix keep the pilot's records and rows. Nothing was uploaded.
