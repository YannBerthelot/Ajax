# Protocol pilot

The four reference comparisons, each run until AdaStop decided, on
2026-10-17 and 18, matched arm only. Protocol version 1, every spec at
version 1. The runs started from c4aeac6, the head of
`batch/reproduction-protocol`. Each comparison's records carry the commit
it ran on: c4aeac6 for PPO, then the commits adding the previous
comparison's records (7938ae7, f1c5012, df60228). Those commits change only
the records, so the code and specs are the same throughout. Every batch
recorded a clean tree.

## Machine

From the manifests: a cloud container with an Intel Xeon Processor @
2.10GHz, 4 cores and 15.7 GB, running Linux 6.18.44 on x86_64 (glibc 2.39).
Everything ran on the CPU (`JAX_PLATFORMS=cpu`) with Python 3.12.3. Ajax
used jax 0.7.2, flax 0.10.7 and optax 0.2.6. The references used SB3 2.9.0
with torch 2.13.0 (the CUDA build, run on the CPU), purejaxql with flax
0.10.7 and chex 0.1.91, and Brax 0.14.2 with MuJoCo 3.7.0. Every side used
gymnax 1.0.0.

## Time

Wall time and CPU time (user plus sys, the run and its children) of each
`run` command. In brackets: the sums over each side's 24 batches.

| Comparison | Wall | CPU | Reference (wall / CPU) | Ajax (wall / CPU) |
|---|---|---|---|---|
| PPO | 48.0 min | 149.2 min | 32.8 / 126.6 min | 12.9 / 20.2 min |
| PQN | 61.1 min | 128.5 min | 30.9 / 64.5 min | 29.3 / 63.0 min |
| PQN-Asterix | 121.9 min | 244.6 min | 61.8 / 123.4 min | 59.1 / 120.2 min |
| PPO-inverted_pendulum | 214.0 min | 620.5 min | 82.6 / 323.1 min | 130.1 / 296.0 min |
| Total | 7.4 h | 19.0 h | | |

`report` and `refcmp_publish.py docs` took 8 min each, almost all of it
in the resolution simulations.

## Results

AdaStop ran every comparison to its cap: 24 interims, 96 seeds per side.
It detected no difference in any of them.

| Comparison | Verdict | Margin | Seeds | Reference | Ajax | Difference [95%] | Resolution |
|---|---|---|---|---|---|---|---|
| PPO, SB3, CartPole | similar | ±25 | 96 | 493.85 | 485.49 | -8.36 [-21.96, +3.73] | 12.0 |
| PQN, purejaxql, CartPole | similar | ±25 | 96 | 497.98 | 499.90 | +1.92 [-0.05, +4.71] | 0.008 (see below) |
| PQN, purejaxql, Asterix-MinAtar | inconclusive | ±0.115 | 96 | 2.292 | 2.455 | +0.162 [+0.029, +0.298] | 0.199 |
| PPO, Brax, inverted_pendulum | similar | ±50 | 96 | 1000.0 | 1000.0 | 0.0 [0.0, 0.0] | 0.001 (see below) |

The table below gives medians over each side's 24 batches. Steps per second
are the batch's 4 seeds times the steps per seed, divided by the batch's
wall time. SB3 and Brax run one process per seed, 4 at once; Ajax and
purejaxql run one program for all 4 seeds.

| Comparison | Reference steps/s | CPU-s per 1M steps | Compile s | Ajax steps/s | CPU-s per 1M steps | Compile s |
|---|---|---|---|---|---|---|
| PPO | 9,807 | 395 | 3.5 | 25,087 | 63 | 7.8 |
| PQN | 25,572 | 82 | 6.7 | 26,944 | 80 | 5.6 |
| PQN-Asterix | 31,759 | 63 | 13.6 | 33,435 | 61 | 11.9 |
| PPO-inverted_pendulum | 15,876 | 246 | 56.1 | 10,094 | 225 | 30.5 |

- **PPO.** 88 of Ajax's 96 seeds end at 500, against 91 of SB3's. Ajax
  uses a sixth of SB3's CPU per step.
- **PQN-Asterix.** The margin, 5% of the reference's mean, is ±0.115. That
  is narrower than both the gap's interval and the resolution (0.199), so
  96 seeds cannot settle the comparison. The 95% interval lies above zero
  (Ajax 7% higher), but AdaStop, which controls the error over all 24
  interims, did not call it. Both curves are still rising at 1.2M steps.
- **PPO-inverted_pendulum.** Ajax uses less CPU per step than Brax but
  is slower in wall time. Its single program kept about 2.3 of the 4 cores
  busy, where Brax's four processes kept about 3.9 busy.
- **Resolution at the ceiling.** All 192 inverted_pendulum seeds score
  1000. On PQN CartPole, 185 of 192 score 500 (Ajax 94/96, purejaxql
  91/96). The resolution's simulation shifts every resampled score by the
  gap, so when the scores have almost no spread, any shift is detected.
  The 0.001 is the bisection's floor (the spread falls back to 1, and the
  floor is a thousandth of it), and PQN's 0.008 means no more than that.
  Neither says how many seeds would catch a gap made of collapsed seeds.
- **Rounding in `results.md`.** `docs/reproduction/results.md` prints one
  decimal, so the PQN-Asterix row reads ±0.1 and +0.2 [+0.0, +0.3]. The
  exact values are in the table above. On the PPO CartPole plot, the
  x-axis labels overlap. Both come from `refcmp_publish.py` and are left
  as they are.

## What failed or was changed

- No run failed: all four exited 0.
  `tests/benchmarks/test_reference_protocol.py` passes on the records.
- The first PPO launch was stopped before it wrote any record. Its first
  SB3 batch was also downloading and installing the SB3 project (about
  2.5 GB of torch and CUDA wheels), and that time would have counted in
  the batch's speed. I then built the three reference projects with
  `uv sync --locked --project benchmarks/reference/envs/<library>`,
  imported each one once, and restarted PPO from no records.
- `mkdocs build --strict` aborted because `results.md` was not in the
  nav. Commit ec5ce78 adds it, one line in `mkdocs.yml`. No code, spec or
  lockfile changed.
- Nothing was uploaded and no pull request was opened.

## After the pilot

- **Reporting.** By the time the pilot was reviewed, main had moved on
  (#172, #183, #185), so the records no longer matched the current code's
  fingerprint and `report` refused them. Reports now replay each
  comparison's latest spec version with records, under the spec those
  records ran, read from the commit in their manifests when it has
  changed since. The results page names that version and commit.
- **PPO's specs are now version 2.** #183 removed `fused_grad_clip` from
  the CartPole spec after its results existed, which the spec-hash test
  caught. That option did not do what the spec said: it clipped both
  networks at 1.0 on top of each network's own clip at 0.5, a joint clip
  that cannot bind. The pilot's Ajax PPO on CartPole therefore clipped
  each network on its own, where SB3 clips once over both; the spec
  listed nothing unmatched. Version 2 runs PPO's new single clip over both
  networks at 0.5, as SB3 does. PPO-inverted_pendulum's version 2 writes
  out Adam's eps, 1e-8 as in Brax, which version 1 had by Ajax's old
  default.
- **What would change if PPO ran again on today's main:**
  - **#172** (squashing as a wrapper): nothing. It is a restructure that
    re-recorded no pin, and CartPole's policy is not squashed.
  - **#183** (PPO's optimiser, ppo2's):
    - CartPole: the clip, from one per network to one over both (version
      2). Adam's eps 1e-5 and the value weight were written out, so they
      stay.
    - inverted_pendulum: only Adam's eps, from 1e-8 to 1e-5, which version
      2 pins back to 1e-8. Its arm does not clip.
  - **#185** (gymnax 1.0.1 from PyPI): no environment of the pilot. The
    fork and 1.0.1 differ only in SpaceInvaders-MinAtar; the reference
    projects now pin 1.0.1 as Ajax does.
  - Any change to `src/ajax` or `uv.lock` changes Ajax's fingerprint, and
    the gymnax pin changes the references', so a re-run trains every seed
    again.
- **Fixed since:** the results page prints three significant figures, the
  plots' x axes fewer labels, and the resolution is 0, not the bisection's
  floor, when every score is equal.

