# Expert guidance on Pendulum: pre-registration

Registered on 2026-11-02, before any arm has run. Once this file is merged,
it is not edited: anything done otherwise is reported as a deviation with
the results.

## Question

Take a controller that swings Pendulum up well but not perfectly. Does
each of four published ways of using it make SAC learn faster than SAC
alone?

Each seed gets one score: the mean of its 20 evaluation returns over the
run, the area under its evaluation curve (`ajax.compare.auc`). A higher
score means the agent did better across the whole run, its early steps
included.

## Task and agent

- **Task.** gymnax's Pendulum-v1: 200-step episodes, a torque in [-2, 2].
  Each seed trains for 20,000 environment steps in one environment.
- **Agent.** SAC with the matched settings of the SAC reference comparison
  at spec version 2 (`benchmarks/reference/refcmp_specs.py`, RL Zoo's
  Pendulum SAC):
  - learning rates of 1e-3 for the actor, the critics and the temperature;
  - a 20,000-row replay, 100 warm-up steps and batches of 256;
  - tau 0.005, gamma 0.99;
  - a temperature starting at 1, with a target entropy of -1;
  - every update from the first one;
  - two 256-256 ReLU networks.

  `guidance.py` writes these settings out, and a test holds them equal to
  the spec.
- **Evaluation.** Every 1,000 steps the deterministic policy plays 10
  episodes, which gives 20 evaluations per seed. With an expert, each
  evaluation also logs the expert's return on the same episodes.

## The expert

`SwingUpPD` (`swing_up.py`) uses energy control away from the top and a PD
near it, saturated at ±2.

- **Swing-up.** This is Åström and Furuta's energy control ("Swinging up a
  pendulum by energy control", Automatica 36, 2000, pp. 287-295, Eq. 8),
  `u = sat(k (E0 - E) sign(thdot))`. Pendulum's energy is
  `E = thdot²/6 + 5 cos(theta)`, which is 5 upright at rest, and
  `dE/dt = u thdot`. The paper drives the pivot's acceleration, which moves
  the energy through `thdot cos(theta)`. A torque moves it through `thdot`
  alone, so the sign here is `thdot`'s.
- **Catch.** Near the top, the torque limit can still hold the pole up
  (`15 sin(theta) < 6`, so `|theta| < 0.41`), and a PD takes over. The
  paper advises a catching controller near the upright position but gives
  no design for it.
- **Gains**, set from the model before any measurement:
  - energy gain k = 1: the torque saturates for most of a swing anyway;
  - catch angle 0.4 rad, just inside the 0.41 the torque can hold;
  - kp = 10 and kd = 2: these turn the linearised pole
    `thddot = 15 theta + 3 u` into `thddot = -15 theta - 6 thdot`, a
    natural frequency of 3.9 rad/s with damping 0.77.
- **Its return**, measured with no training
  (`guidance.py expert`: 1,000 episodes from gymnax's resets, key 0): mean
  **-185.3**, sd 110.5, median -177.6. For scale, SB3's SAC ended at -141
  after 20,000 steps in an earlier 16-seed recording, and -200 counts as
  solved.

## Arms

Every arm but A0 passes the expert as `expert_policy=`, which the
extensions read. A0 passes it as `eval_expert_policy=`, which only logs its
return.

| Arm | Ajax | What it does | Source |
|---|---|---|---|
| A0 | SAC | SAC alone | Haarnoja et al., 2018 |
| A1 | `ExpertPrefill(n_steps=2000)` | The expert's 2,000 steps (10 episodes) go into the replay before training | DDPGfD: Vecerik et al., "Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards", arXiv:1707.08817, 2017 |
| A2 | `JSRLCurriculum(episode_length=200, decay_frac=0.5)` | The expert acts for the first H steps of each episode; H falls linearly from 200 to 0 over the first 10,000 steps | Uchendu et al., "Jump-Start Reinforcement Learning", ICML 2023 |
| A3 | `IBRL(expert_policy=expert)` | Acting and the TD target each take the better of the expert's and the policy's action, by the target critics | Hu, Mirchandani and Sadigh, "Imitation Bootstrapped Reinforcement Learning", RSS 2024 |
| A4 | `ResidualPolicy(scale=1.0)` | Executes `clip(expert + policy, -1, 1)`, in the agent's units | Johannink et al., "Residual Reinforcement Learning for Robot Control", ICRA 2019 (concurrent work: Silver et al., "Residual Policy Learning", arXiv:1812.06298) |

Each arm is the paper's idea as Ajax implements it on SAC, not the paper's
whole method:

- **A1.** DDPGfD also uses prioritised replay, n-step returns and an L2
  term. It keeps the demonstrations for the whole run, while this 20,000-row
  replay overwrites them in the last 2,000 steps.
- **A2.** The paper's curriculum lowers H when the evaluation return
  passes a threshold; Ajax lowers it on a schedule.
- **A3.** The paper's guide is a policy trained by behaviour cloning, and
  its agent is TD3-like. Here the guide is the controller itself and the
  agent is SAC.
- **A4.** It starts with SAC's policy spread, a standard deviation of about
  1 in [-1, 1] units, on top of the expert.

## Test

- **AdaStop** (`ajax.compare.AdaStop`), each arm against A0: 4 two-sided
  comparisons with a family-wise error of 0.05. Each interim adds n = 4
  seeds per arm, up to K = 12 interims, so at most 48 seeds per arm.
  Interim k uses seeds 4k to 4k+3, the same numbers for every arm. An arm
  stops training once its comparison is decided; A0 runs while any is
  open.
- **Reported for each comparison:**
  - AdaStop's decision (larger, smaller or equal);
  - the difference of the mean scores with its 95% bootstrap interval
    (`ajax.compare.difference`);
  - the resolution: the smallest gap this budget detects 80% of the time,
    simulated from the observed scores (`ajax.compare.resolution`).
- **Descriptive only, no test:**
  - each arm's final evaluation;
  - the median steps to its first evaluation at -200 or better, and how
    many seeds never got there;
  - the expert's return on the evaluations.
- **Not asked:** whether an arm ends worse than SAC. That would need the
  final return as a second registered metric, with the error split between
  the two.

## Running

```bash
uv run python benchmarks/expert_guidance/guidance.py run
```

Run from a clean tree on a commit that contains this file; every record
names the commit it ran on. It runs locally on the CPU, since it records
returns, not speed. Each batch appends one record to `results.jsonl`, and a
second run reuses the recorded batches. The results, the records and a page
under `docs/reproduction/` come in their own pull request.

**Cost.** At the cap, 240 runs of 20,000 steps (60 batches of 4 seeds),
about 8-12 CPU-hours. It is less if comparisons are decided early.
