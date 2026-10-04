# Round 3 (pre-registered 2026-10-04, before any round-3 result exists)

Rounds 1-2 (full/, full2/, cmpE/, cmpR/): once both sides use the same two-hot
expectation (D22), every training statistic matches (entropy seed ranges
overlap in 10/10 windows in both pairs; S2 and world-model losses match). The
pre-registered eval statistic S1 still fails in both pairs, driven by the
20000-row checkpoint, where all 6 reference runs score >= 399 (stock 405, 466,
400; exact 470, 495, 469) against 1/6 at 18000 and 0/6 at 16000; no Ajax run
shows this. Over all ten checkpoints the eval sums are ref 1871 / 1824 vs Ajax
2117 / 1892.

Hypotheses for the 20000-row concordance:
- H1 end-of-run artifact on the reference side: the reference jumps at its
  final checkpoint, whatever the run length.
- H2 correlated reference runs (D2): stock make_replay gives every reference
  run selectors.Uniform(seed=0), the same replay index stream, so reference
  runs are not independent and can share good/bad stretches.
- H3 real late-learning difference: the reference improves faster than Ajax
  late in training.

Design: reference = 29eb964 + exact two-hot expectation (D22) + replay sampler
seeded per run ([seed, 0x5EED]; D2), i.e. Ajax's documented choices except D1
(asynchrony, not removable); run_reference_round3.py. Ajax = 5cb0738 (frozen
worktree), unchanged. Both: seeds 0 1 2 3 4, 24000 rows, evaluation every 400
rows (10 episodes, PROTOCOL.md section 4), all else as PROTOCOL.md.

Pre-registered statistics (per seed, then over seeds):
- LATE = mean eval return over the checkpoints in [12000, 24000] (31 points).
  Primary. Rule: LEARNS = mean Ajax LATE >= min reference LATE; WITHIN_RANGE =
  min <= mean Ajax LATE <= max; plus Welch's t-test on the 5 vs 5 per-seed
  LATE values (two-sided p).
- S1 as in round 1 (16000, 18000, 20000), for continuity, same rule.
- Concordance: number of runs per side with eval >= 400 at 20000 and at 24000,
  and the per-side rate of eval >= 400 over all checkpoints in [16000, 24000].

Readings (decided now):
- H1 if >= 4/5 reference runs score >= 400 at 24000 while the reference rate
  at 20000 is no higher than its [16000, 24000] rate.
- H3 if Welch p < 0.05 with reference LATE > Ajax LATE.
- H2 (or noise) if neither: no reference concordance at 20000 beyond its
  late-phase rate and no significant LATE difference.
- If Ajax LATE > reference LATE with p < 0.05, that is a difference too and is
  investigated.

## Amendment (2026-10-04, before any round-3 evaluation existed: the runs had
## not reached their first 400-row checkpoint; found by a synthetic self-test)

The H1 reading above would also fire when the policy simply reaches the cap
late in training (high at 24000 because learning progressed, rate at 20000 at
or below the late rate). H1 is an anomaly of the final checkpoint relative to
the reference's own late phase, so the reading becomes: H1 if >= 4/5 reference
runs score >= 400 at 24000 AND the reference's rate of eval >= 400 over the
checkpoints in [16000, 24000] is at most 0.4. Everything else is unchanged.
