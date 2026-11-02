# Expert guidance on Pendulum

Take a controller that swings Pendulum up well but not perfectly. Does each
of four published ways of using it make SAC learn faster than SAC alone?

The question, the controller, the arms and the test were registered before
any arm ran, in
[`benchmarks/expert_guidance/PREREG.md`](../../benchmarks/expert_guidance/PREREG.md).
Each seed's score is the mean of its 20 evaluation returns over 20,000
steps (`ajax.compare.auc`). AdaStop compared each arm with SAC alone, with a
family-wise error of 0.05, 4 seeds per arm and interim, at most 12 interims.

## Verdicts

| Arm | Against SAC alone | Seeds | Difference in score [95%] |
|---|---|---|---|
| A1 `ExpertPrefill` (DDPGfD) | no difference found | 48 | +2.6 [-4.3, +9.4] |
| A2 `JSRLCurriculum` (JSRL) | lower | 8 | -308.5 [-369.7, -236.4] |
| A3 `IBRL` | higher | 16 | +21.8 [+12.1, +31.1] |
| A4 `ResidualPolicy` (residual RL) | higher | 8 | +88.3 [+78.2, +97.3] |

- **A1** ran to the last interim without AdaStop telling it apart from SAC
  alone. That does not show the two equal: a gap of 10.7 or more would have
  been found 80% of the time (the resolution), a smaller one could be missed.
- **A2** scored lower, decided after two interims.
- **A3** scored higher, decided after four.
- **A4** scored higher, decided after two.

## Results

The output of `guidance.py report`, from the records:

| Arm | Seeds | Mean evaluation return | Final evaluation | Steps to -200 (median; seeds never) |
|---|---|---|---|---|
| A0 SAC | 48 | -267.9 | -146.6 | 4,000; 0 |
| A1 ExpertPrefill (DDPGfD's demonstrations in the replay) | 48 | -265.3 | -145.4 | 3,000; 0 |
| A2 JSRLCurriculum (JSRL) | 8 | -576.5 | -144.0 | 10,000; 0 |
| A3 IBRL | 16 | -246.2 | -140.8 | 3,000; 0 |
| A4 ResidualPolicy (residual RL) | 8 | -179.7 | -145.3 | 2,000; 0 |

| Comparison | AdaStop | Difference [95%] | Resolution |
|---|---|---|---|
| A1 vs A0 | equal | +2.6 [-4.3, +9.4] | 10.7 |
| A2 vs A0 | smaller | -308.5 [-369.7, -236.4] | 42.8 |
| A3 vs A0 | larger | +21.8 [+12.1, +31.1] | 10.8 |
| A4 vs A0 | larger | +88.3 [+78.2, +97.3] | 9.1 |

The expert's return on the same evaluation episodes was -194.5. The final
evaluations and the steps to -200 are descriptive, not tested; every seed of
every arm reached -200.

The mean evaluation return at some of the 20 evaluations, over each arm's
seeds:

| Arm | 1,000 steps | 2,000 | 3,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|---|
| A0 | -1632 | -845 | -301 | -189 | -142 | -147 |
| A1 | -1599 | -742 | -253 | -246 | -141 | -145 |
| A2 | -1597 | -1194 | -886 | -1377 | -184 | -144 |
| A3 | -1439 | -613 | -203 | -234 | -141 | -141 |
| A4 | -723 | -183 | -142 | -148 | -158 | -145 |

## What each score measures

Each evaluation plays the arm's deterministic policy as the method deploys
it:

- A0, A1 and A2 act with the policy alone.
- A3 acts as IBRL does, with the better of the policy's action and the
  expert's, by the target critics.
- A4 executes the expert's action plus the policy's, clipped to the
  action range.

A2's expert acts during collection only. Its curriculum hands whole episodes
to the policy at step 10,000, half the run, which is also A2's median step
to -200. A1's 2,000 expert steps fill the replay before training and come on
top of the 20,000.

## How it ran

- **Machine:** an Apple M4 Pro (14 cores), on the CPU (`JAX_PLATFORMS=cpu`),
  on 2026-11-02.
- **Time:** 75 minutes of wall time and 4.9 CPU-hours for 32 batches of 4
  seeds. The registered estimate, 8-12 CPU-hours, was for all 60 batches;
  three comparisons were decided early.
- **Code:** the first batch ran on main at `ba5a070`, which merged the
  pre-registration. Each later batch ran on the commit that recorded the
  batch before it. Those commits add lines to `results.jsonl` and change
  nothing else.
- **Deviations from the pre-registration:** none. The settings, arms and test
  are the registered ones, and `PREREG.md` is unchanged. A first launch
  failed in the shell before training anything and recorded nothing.

`guidance.py report` prints the tables above from
[`results.jsonl`](../../benchmarks/expert_guidance/results.jsonl), and
`guidance.py run` reuses every recorded batch.
