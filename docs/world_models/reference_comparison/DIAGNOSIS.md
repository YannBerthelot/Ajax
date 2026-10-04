# Diagnosis: Ajax DreamerV3 vs reference 29eb964, CartPole 1m comparison

> Committed copy of the root-cause investigation written on 2026-10-04 between
> rounds 1 and 2 of the comparison (README.md in this directory). Lightly
> edited: the paths below point to the committed files where they exist; the
> investigation's probe scripts and outputs (`INV`, about 900 MB) were not
> committed. The re-run recommended at the end became round 2
> (`run_reference_exact.py`, `run_ajax_refdecode.py`). Documentation fixes 1
> and 2 below were applied in commit 765a881 (`deviations.md` D22 and the
> `TwoHot.decode` docstring); fix 3 was not, because `PROTOCOL.md` is kept as
> pre-registered.

Scope: S-A (activation/embed lower in Ajax), S-B (actor entropy lower in Ajax
from the 4000-row window), and the window-1 ret/std gap (0.135 ref vs 0.0099
Ajax). Sources: three investigation lines (init-encoder, static-audit,
matched-experiment) and ten adversarial verifications (verify_0..verify_9),
all under `INV` (below). Nothing in the Ajax worktree, the reference checkout
or the Ajax repo was edited.

Path abbreviations: `REF` = `dreamerv3/` of the reference checkout
(danijar/dreamerv3 at 29eb964; README.md, "Setup"), `AJX` = `src/ajax/` of the
Ajax repository at commit 5cb0738 (the version the comparison ran), `INV` = the
investigation directory (**not committed**: every `INV/...` path below names a
probe script or output that is not in the repository, except
`INV/verify_3/refdecode.py`, committed as `refdecode.py` in this directory),
`FULL` = `results/round1/` in this directory.

## Bottom line

| Symptom | Verified cause | Class | Ajax bug? | Re-run needed? |
|---|---|---|---|---|
| S-A, embed 7-9% lower | Initial-parameter draws of the three seeds per side | Noise (seed draw) | No | No; change how it is reported |
| S-B, entropy ~11% lower | Reference two-hot mean noise (D22) at the 4000-row window; windows >= 8000 rows untested | Documented deviation (impact mis-documented) | No | Yes, for every actor/critic statistic |
| ret/std window 1 | Same D22 noise (per-element, not the init offset) | Documented deviation (impact mis-documented) | No | Yes (same re-run) |

No Ajax algorithm or hyperparameter bug was found. There are documentation
errors: D22's stated impact is wrong (see "Documentation fixes").

## The shared mechanism (D22), measured

Code, both sides:
- Reference `REF/jaxutils.py:226-243` (`TwoHotDist.mean`, odd-n branch):
  `(p2*b2).sum + ((p1*b1)[::-1] + p3*b3).sum`. Heads have 255+1 outputs with
  the padding logit dropped (`REF/nets.py:437-442`), so n = 255 (odd branch).
  Bins are built in float32 inside jit: `symexp(linspace(-20, 0, 128))`
  mirrored, |b| up to 4.85e8 (`REF/nets.py:466-471`).
- Ajax `AJX/distributional.py:234-238` (`TwoHot.decode`):
  `sum((p[m+1:] - p[:m][::-1]) * b[m+1:])`, bins in float64 rounded once to
  float32 (`AJX/distributional.py:170-174`).
- Decode sites. Reference: reward `agent.py:285`, critic/slowcritic
  `agent.py:297-298`, lambda return `:302-310`, advantage `:313-316`.
  Ajax: `learner.py:308` (reward), `:358`, `:370` (slow critic),
  `actor_critic.py:226` (critic value).

Measured facts:
1. Under jit, XLA contracts each mirror pair into a fused multiply-add (one
   product unrounded). Seen in the optimised HLO and ARM64 object code
   (`INV/verify_4/probe_ir.py`, `xla_dump/module_0011.jit_stock.o`). Eager
   reference code and the exact formula both give error 0.
2. Once the central logits move (from update 2) while the outer bins stay
   bitwise mirror-equal, the reference mean has per-element error std
   ~0.08-0.09 (max ~0.3) against a true spread of ~1e-4. Measured on real
   in-graph logits captured from the reference train step
   (`INV/verify_4/analyze_logits.out`) and on synthetic logits by four
   independent probes (`INV/matched-experiment/twohot_noise_probe.py`,
   `INV/verify_1/probe_mean.py`, `INV/verify_6/decode_ref.py`,
   `INV/verify_7/fma_probe.py`; the last correlates -0.98 with an FMA-residual
   model).
3. With exactly zero logits the reference gives a constant whose value
   depends on the compiled program: +0.0153 inside the reference train step,
   -0.169 standalone with precomputed bins, +0.162 when the same formula is
   compiled inside Ajax's step. A constant produces no spread (ret/std at
   updates 0-1 is 0.0033).
4. The noise lasts well past window 1: reference val/std 0.085 (2000 rows),
   0.052 (4000), ~0.01 (6000) vs Ajax vmap3 ~0.003 (field `train["val/std"]`
   of `FULL/ref_s*/records.jsonl.gz` and `FULL/ajax/s*/records.jsonl.gz`;
   `INV/verify_7`).

## S-A: activation/embed lower in Ajax

**Verified cause: seed draw (noise). Not an Ajax bug.**

- Same function: reference params mapped into Ajax's encoder
  (`AJX/agents/DreamerV3/networks.py:387-395` vs `REF/nets.py:246-277`) give
  max |diff| 1.4e-6 to 2.0e-6 (`INV/init-encoder/ajax_probe.py`,
  `INV/verify_5`, `INV/verify_8`). Both metrics average |encoder output| over
  the same [16, 64, 64] trained rows (`REF/agent.py:234,389` vs
  `AJX/agents/DreamerV3/world_model.py:214-218`, `learner.py:470`).
- Same init population: four independent samples (60 to 400 seeds per side)
  agree within 0.2-0.6% (e.g. 0.4019 vs 0.4025, n = 400, `INV/verify_0`;
  0.4049 vs 0.4034, Welch p = 0.53, `INV/verify_8`).
- The run seeds' actual inits: reference checkpoints (initial saves,
  `FULL/ref_s*/checkpoint.ckpt`, **not committed**: 7.6 MB checkpoints)
  0.418-0.447; Ajax key chain
  (`base.py:271-273` -> `train_DreamerV3.py:368` -> `learner.py:189`)
  0.366-0.405 (exact values vary by observation set). Window-1 values follow
  each seed's init within 0.01. 3-vs-3 gap at init is ~2.2 SE (p ~ 0.02-0.04).
- Causal: Ajax seed 165 (highest init of 300) gives window-1 embed 0.460,
  above every reference seed; seed 270 gives 0.329 (`INV/verify_5/ajax_hilo`).
  Matched init and data give identical embed on both sides for 1500 updates
  (`INV/matched-experiment/tables_fullrun_windows.md`).
- Correction to the symptom: the gap is not a constant 7-9%. It is -9.8% at
  2000 rows, falls to about -5% from 12000 rows, and seed ranges overlap from
  ~10000-16000 rows. Relative to each seed's own init, Ajax is 0 to +5.7%,
  because the high-drawn reference seeds drift down (ref s1 0.439 -> 0.376).

Action: report activation/embed relative to each seed's update-0 value, or
use matched inits or >= 10 seeds. Drop S-A from the systematic differences.

## S-B: actor entropy lower in Ajax

**Verified cause at the 4000-row window: D22 (documented deviation, impact
mis-documented). Not an Ajax bug. Later windows (>= 8000 rows): unverified.**

The decisive test (`INV/verify_3`, `cmp_refdecode_vmap3.txt`): the same
3-seed vmapped Ajax program as the main run, with only `TwoHot.decode`
replaced by the reference's literal pair sum (`INV/verify_3/refdecode.py`,
committed as `refdecode.py`),
reproduces the reference's whole window-1 (4000-row) signature:

| 4000-row window | Reference s0/s1/s2 | Ajax vmap3 (main run) | Ajax vmap3 + ref decode |
|---|---|---|---|
| ent/action/mean | 0.517 / 0.506 / 0.518 | 0.460 / 0.449 / 0.461 | 0.505 / 0.503 / 0.518 |
| adv/std | 0.328-0.339 | 0.267-0.286 | 0.331-0.341 |
| val/std | 0.052-0.053 | 0.003 | 0.050-0.052 |
| rew/min | -0.111 to -0.113 | 0.000 | -0.109 to -0.113 |
| actor_loss | 0.161-0.175 | 0.146-0.152 | 0.162-0.166 |
| critic_loss | 5.05-5.30 | 5.48-5.63 | 5.04-5.14 |

That closes ~92% of the entropy gap (mean 0.457 -> 0.509 vs ref 0.514).
Supporting evidence:
- Fixed-data matched experiment (identical init, batches, synchronous, no
  environment): swapping only the decode formula swaps the entropy curves in
  both codebases; ratio 0.92 (updates 481-980) and 0.88 (981-1480), within
  the full run's 0.86-0.89 (`INV/matched-experiment/tables_100.md`,
  `tables_fullrun_windows.md`). Loop-level differences (D1, D2, U3, U6, U7)
  are therefore not needed to produce S-B.
- Single-seed closed-loop A/B (Ajax seed 0, `INV/verify_6`): reference
  decode raises entropy to 0.533 (4000) and 0.330 (6000), slightly above the
  reference range.

Mechanism (partly hypothesis): the decode noise enters values and imagined
rewards, hence returns and advantages, while the return normaliser sits at
its floor of 1 (`limit 1`, `configs.yaml:141`). Measured: wider adv/std,
delayed entropy decline by ~100-200 updates (smoothed matched curves).
Hypothesis, not isolated: the extra zero-mean advantage noise lowers the
effective REINFORCE step after LaProp's RMS normalisation, so the actor
sharpens later.

Important qualifier, measured: Ajax's exact decode is noise-free only while
the outer logits stay bitwise mirror-symmetric, and that depends on the
compiled program. Single-seed (vmap batch 1) Ajax with the exact decode
shows its own noise (window-0 val/std 0.019-0.026 vs 0.0026 under vmap3;
window-1 rew/min -0.72 to -1.04), and its entropy lands near the reference
(0.491-0.509 at 4000) for that reason (`INV/static-audit/compare_all_single_seed.txt`,
`INV/verify_3/cmp_s5.txt`). No seed leakage under vmap: seed 0 is bitwise
identical under [0,1,2] and [0,5,7] (`INV/verify_3/cmp_057.txt`). What breaks
mirror symmetry in the batch-1 program is not identified (a micro-probe of
the output-layer gradient kept symmetry: `INV/static-audit/probe_mirror_grad.out`).
This is a numerical fragility of the bin design (outer bins at +-4.85e8 hold
~1/255 of the mass early on; sensitivity ~p_j b_j), shared by both
codebases, not an Ajax bug. Consequence: Ajax's early actor dynamics differ
between vmap batch sizes at a fixed seed.

Unverified: whether D22 explains the 8000-18000-row windows (ratio 0.86 at
8000, 0.77 at 12000, non-overlapping). A fixed 150-200-update lag predicts
only ~3% at 8000; persistence would need closed-loop amplification (different
collected data) or another cause. Not run: vmap3 + ref decode, or reference
+ exact decode, beyond 4000-6000 rows.

## ret/std in window 1 (0.135 ref vs 0.0099 Ajax)

**Verified cause: D22 per-element decode noise. Documented deviation; not an
Ajax bug. Explains ~100% of the window-1 ret/std, adv/std, val/std and
rew/min gaps.**

- Reference side: patching only `TwoHotDist.mean` to the exact form takes
  ret/std 0.1328 -> 0.0096 and adv/std 0.158 -> 0.0097 with everything else
  identical (`INV/init-encoder/ref_d22.py`, `INV/verify_1/ref_ab.py`,
  `INV/verify_4/ref_arm.py`).
- Ajax side, the actual comparison program (vmap3): reference decode gives
  ret/std 0.138/0.135/0.137, adv/std 0.160-0.163, val/std 0.0847, rew/min
  -0.272 vs reference 0.133-0.138, 0.159-0.162, 0.0846, -0.278
  (`INV/verify_3`, `INV/verify_4/ajax_refmean.py`).
- The constant zero-init offset is not the cause: at updates 0-1 ret/std is
  0.0033; the gap appears at update 2 when per-element noise starts.

## Ajax bugs: none found

No exact fix is required. The static audit (`INV/static-audit`) found no
algorithmic or hyperparameter difference (resolved config key by key;
optimizer chain, AGC, slow critic, retnorm, imagination start, actor loss,
replay context and write-back read as equivalent; code reading, not every
path separately probed). The matched experiment confirms the full-scale
update rule matches once the decode formula is the same, except item 6
below.

Why the tiny parity fixtures could not show the D22 dynamics: they
deliberately avoid the ill-conditioned regime.
`docs/world_models/parity/dreamerv3_train_fixtures.py:32-50` (at 5cb0738),
`:146-150`, `:345-365` replace the three two-hot output layers with a bias
concentrated on the middle bins (`max(-0.5|j-127|, -27)`), making
`sum p|b| ~ 0.45`, precisely so that the FMA noise disappears. Plus warmup 2
and a few updates, so the 1000+-update dynamic effect is out of reach. That
was the right choice for per-update parity; it means a run-level comparison
like this one is the only place the effect shows.

## Documentation fixes (recommended, not applied)

1. `docs/world_models/deviations.md:85` (D22, at 5cb0738): the
   reference column says a zero-init head predicts "0.07 to 0.16"; the
   impact column says "decode within 3·ε·E_p|b|". Replace with: the
   zero-init constant is program-dependent in sign and size (+0.0153 in the
   reference step, -0.169 standalone, +0.162 in Ajax's program); from about
   update 2 the reference adds per-element noise of std ~0.085 to rewards,
   values and slow-critic targets for ~1500-2500 updates at 1m/CartPole
   (val/std 0.085 -> 0.052 -> ~0.01); this sets the reference's early
   ret/adv/val spread and, at the 4000-row window, raises its actor entropy
   ~11% relative to Ajax (vmap3). Also record that Ajax's exact decode is
   exact only for bitwise mirror-symmetric logits, and that its residual
   noise depends on the compiled program (vmap batch 1 vs 3: 10x in window-0
   val/std).
2. `AJX/distributional.py:221-228` docstring: same "0.07 to 0.16" claim.
3. `PROTOCOL.md:756` (U4, this directory): "negligible after the first
   updates" is wrong.

Do not add a reference-formula mode to Ajax: its output depends on XLA fusion
(single-seed patched Ajax gives val/std 0.090 at 4000 vs the reference's
0.052, `INV/verify_6`; vmap3 patched gives 0.050-0.052).

## What this means for the comparison already run

- Stand as is: update counts, the S2 training score (within 1%), world-model
  losses (D22 does not touch them; `INV/verify_6` shows S2 unchanged by the
  patch).
- Do not interpret: every actor/critic statistic (entropy, adv/std, ret/std,
  val/std, rew/min, actor_loss, critic_loss, retnorm scale) and the failed
  eval statistic, whose cause was not investigated and may be affected by
  D22's dynamics. activation/embed needs init-relative reporting.
- Re-run needed: 3 seeds to 20000 rows with the reference's
  `TwoHotDist.mean` monkeypatched to the exact mirror-difference form without
  editing the checkout (`INV/matched-experiment/ref_driver.py --exact-twohot`,
  `INV/init-encoder/ref_d22.py --exact`). Optionally a second Ajax arm, vmap3
  + `INV/verify_3/refdecode.py` (committed as `refdecode.py`). Keep the Ajax
  arm's vmap layout fixed and
  recorded (batch 1 and batch 3 differ numerically). This re-run is what
  tests the >= 8000-row entropy windows. D1/D2 asynchrony changes are not
  needed for S-B.

## Refuted / unverified

Refuted:
- "S-B is a harness mismatch (vmap3 vs single-seed), not D22"
  (static-audit; verify_2's "D22 is not the cause"). Single-seed runs match
  the reference only because a different, unidentified noise source appears
  in that program (different signature: window-1 rew/min -0.7 to -1.0 vs
  -0.11). The same-program test (vmap3 + ref decode) reproduces the
  reference, and verify_2's own single-seed A/B used a noisy control.
- "S-A is a constant 7-9% gap present in every window, non-overlapping": it
  shrinks to ~5% and ranges overlap from ~10000-16000 rows.
- "Each seed keeps its init level": approximately only (ref s1 falls 14%).
- D22 as a constant 0.07-0.17 offset explaining window-1 ret/std: the
  constant gives ret/std 0.003.
- verify_0's "256 outputs, even branch, exactly 0 under jit": the head pads
  255 to 256 and drops the pad (`REF/nets.py:437-442`), so the odd branch
  runs, and it gives a nonzero jit constant (-0.169 / +0.0153).
- Matched-experiment's seed spread "0.001" at updates 1400-1499: it is
  0.005-0.007 with 4 seeds (`INV/verify_9`).
- Ruled out as causes of S-A or S-B: init distributions (all 90 tensors),
  encoder forward, observation normalisation, LaProp denormal handling
  across jax versions, ring size, return normaliser update rule, optimizer
  split, seed leakage under vmap, D1/D2/U3/U6/U7 as necessary causes.

Unverified / open:
- S-B beyond the 6000-row window (closed-loop persistence of the gap).
- The mechanism by which advantage noise slows entropy decline (LaProp RMS
  normalisation is a hypothesis).
- What breaks outer-logit mirror symmetry in Ajax's single-seed program.
- Residual (`INV/verify_9`): on matched data with the same formula, the
  reference's entropy ends 1.6-3.8% lower than Ajax's from ~update 1000
  (both reference runs below all 4 Ajax seeds; p = 1/15 stock, 1/5 exact).
  Opposite sign to S-B, so it cannot contribute to S-B. Candidate (bins
  differ by 1.2e-6 relative) judged unlikely; not bisected.
- Two single-seed seed-0 runs with the reference decode disagree at 4000
  rows (0.509 `INV/static-audit/out_refdecode_s0` vs 0.533
  `INV/verify_6/loop_reftwohot_s0`), consistent with program-dependent
  noise; not resolved.
- The failed pre-registered eval statistic (Ajax 204 vs ref 241-386).
