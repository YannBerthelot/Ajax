# Offline learning and imitation: design note (2026-10-19)

Proposal on main 1751d82. One method per branch and PR. Reference files are
named from standing knowledge; each `@pin` commit is read and pinned before
that method's PR, and the details read from it are unverified until then.

## 1. Does `ImitationLoss` already do DAgger?

DAgger (Ross, Gordon and Bagnell, AISTATS 2011, Alg. 3.1): start with D = ∅.
At each iteration i, roll out π_i = β_i π* + (1 − β_i) π̂_i, label every
visited state with the expert's action, set D ← D ∪ D_i, then fit π̂_{i+1} on D
with a supervised loss only. The paper's parameter-free choice is
β_i = I(i = 1); the other is β_i = p^(i−1). Its bound needs the β_i to average
to 0 and a no-regret learner: Follow-the-Leader on D, or an online learner.

| DAgger | `ImitationLoss(expert)`, `expert` the agent's own, on SAC/TD3/REDQ/ASAC | Same |
| --- | --- | --- |
| labels π*(s) on the learner's states | `a_expert` stored at collection, from the expert's tracked state | yes |
| D, every iteration's data | the replay, sampled uniformly, FIFO past `buffer_size` | within capacity |
| β_1 = 1 | SAC's expert prefill; `PriorData(record(expert))` | yes |
| β_i = p^(i−1), mixed per step | `EDGEExploration(gate="fixed")` | SAC only |
| fit on D | one gradient step per env step | an online learner, which the bound allows |
| supervised loss only, no reward | added to the RL loss; a critic, a reward and exploration noise | no |

PPO, APO, APG and UDRL apply the term to their latest rollout only, and AVG
to its latest step. There is no D there, so it is not DAgger on those agents.

**Verdict.** The earlier judgement holds for the replay agents. Their
`ImitationLoss` is DAgger's aggregation with β_i = I(i = 1), plus an RL loss.
Pure DAgger needs no reward and no critic, so it is a separate agent (§2.1),
and it is the reward-free baseline. The docs will state this equivalence.

## 2. Candidates

### 2.1 DAgger: agent `DAgger`
- **Paper:** Ross, Gordon and Bagnell, AISTATS 2011. The authors released no
  code. The reference is HumanCompatibleAI/imitation
  `src/imitation/algorithms/dagger.py` (`ExponentialBetaSchedule`, per-env,
  per-step mixing in `InteractiveTrajectoryCollector`, `SimpleDAggerTrainer`) @pin.
- **Form:** an agent, because its update is the supervised loss alone. Surface:
  `DAgger(env, expert_policy, beta_decay=0.0, iteration_steps, actor_architecture,
  learning_rate, buffer_size, batch_size, n_envs, extensions=())`. β_i =
  beta_decay^(i−1), with 0^0 = 1, so the default 0 is I(i = 1). `expert_policy`
  is required: it is the algorithm's oracle.
- **TrainLoop:** `off_policy`, as UDRL. At each step, each env executes the
  expert's action with probability β_i, else the actor's deterministic action.
  `expert_collect_kwargs` stores `a_expert`, so stateful experts work. Each
  tick runs one update: MSE(μ(s), a_E) for continuous actions, cross-entropy
  to the expert's action for discrete ones. Evaluations log the expert's return,
  `Eval/expert bias`, `dagger/beta` and `dagger/loss`.
- **Data:** the replay is D. `PriorData(record(expert))` can seed it. Rows whose
  `a_expert` is NaN (logged data) get weight 0.
- **Capabilities:** UDRL's, `actor obs_networks off_policy replay`, with
  `agent_expert` always present. It runs CloningPretrain, ImitationLoss,
  PriorData and the metrics modules. Critic modules are refused: there is no critic.
- **Probe `dagger-off-path`:** a 4-state one-hot chain. From state 0, an action
  a > 0 leads to 1 and then 2; a ≤ 0 leads to 3, which only a learner's
  mistake reaches. The expert's labels are 0.8, 0.4, −0.2 and −0.6. The actor
  starts as a `CloningPretrain` clone of a wrong expert (−0.5 everywhere), and
  `iteration_steps=1`. Truth: a(0) = 0.8, a(1) = 0.4, a(3) = −0.6. Wrong
  answers: cloning only the expert's states, or β stuck at 1, gives
  a(3) = −0.5; labels taken from the executed action give −0.5 everywhere.
  The last two are the planted faults. A discrete twin cell follows.
- **Budget:** about 250 source lines (agent 70, train 160, state 20), 120 unit
  test, 70 probe and 80 docs.

### 2.2 TD3+BC offline: TD3 with two `ImitationLoss` options
- **Paper:** Fujimoto and Gu, NeurIPS 2021 (arXiv:2106.06860). Code: sfujim/TD3_BC
  `TD3_BC.py` (λ = α / mean|Q(s, π(s))|, α = 2.5, actor loss −λQ + MSE(π(s), a))
  and `main.py` (states standardised by the dataset, std + 1e-3) @pin.
- **Form:** no new agent. `target="replay"` clones the replayed action, so
  logged data without an expert works. `alpha` sets coef = mean|Q(s, μ(s))| / α,
  held constant, with no `/4`. That is the paper's loss divided by λ, which
  Adam nearly cancels (offline-proto `checks.py`). Replay agents put `actions`
  in their actor-loss batch. Recipe: `TD3(normalize_observations=True,
  extensions=(ImitationLoss(None, target="replay", alpha=2.5),))`, trained offline (§3).
- **Capabilities:** `alpha` needs `q_critic`; `target="replay"` needs `replay`.
- **Probe `td3bc-alpha-TD3`:** PriorData's `signed_rows`: one step with reward
  a·s, and logged actions on a grid of mean 0. Q = a·s, so the actor settles at
  π = λs/2 with λ = α/|π|. At α = 0.5 that gives a(±1) = ±√(α/2) = ±0.5.
  Wrong answers: no BC gives ±1; BC alone gives 0; an unnormalised λ = α gives ±0.25.
- **Deviation:** Ajax's running normaliser, sqrt(var + 1e-8), seeded with the
  dataset, replaces fixed statistics with std + 1e-3. The two differ on a
  constant column, such as a single setpoint.
- **Budget:** about 50 source lines, 60 unit test, 40 probe and 40 docs.

### 2.3 IQL: agent `IQL`
- **Paper:** Kostrikov, Nair and Levine, ICLR 2022 (arXiv:2110.06169). Code (JAX):
  ikostrikov/implicit_q_learning. `critic.py` holds the expectile loss and
  `update_v`/`update_q`; `actor.py` holds the weights exp(β(Q − V)) clipped at
  100; `configs/` give τ 0.7 and β 3 for locomotion, τ 0.9 and β 10 for
  antmaze, with a cosine actor schedule @pin.
- **Form:** an agent, because its V, Q and actor losses replace SAC's. Surface:
  the paper's hyperparameters and `extensions=()`.
- **TrainLoop:** `off_policy`. It trains offline (§3), then fine-tunes online
  with the same update (paper Sec. 5.3), with the offline data kept in its replay.
- **Capabilities:** `actor q_critic v_critic critic_ensemble discounted replay
  off_policy obs_networks`.
- **Probe `iql-expectile`:** `signed_rows` at τ 0.9 and β 10, where the clip
  binds. Q = a·s, and V(±1) is the τ-expectile of the 21 logged returns. The
  actor's mean is Σwa / Σw with w = min(exp(β(as − V)), 100), the weighted
  Gaussian likelihood's maximum at a state-independent std. Wrong answers:
  τ = 0.5 (V is the mean, 0), no weights (mean action 0), no clip.
- **Budget:** about 300 source lines, 150 unit test, 60 probe and 80 docs.

### 2.4 CQL and Cal-QL: Extension `ConservativeQ`
- **Papers:** Kumar, Zhou, Tucker and Levine, NeurIPS 2020 (arXiv:2006.04779),
  CQL(H); Nakamoto et al., NeurIPS 2023 (arXiv:2303.05479). Code:
  aviralkumar2907/CQL `d4rl/rlkit/torch/sac/cql.py`, which draws 10 uniform and
  10 policy actions at s and s′ with importance weights and has `min_q_weight`
  and Lagrange options; nakamotoo/Cal-QL `JaxCQL/conservative_sac.py`, which
  floors the policy actions' Q at the Monte-Carlo return @pin.
- **Form:** an Extension. `ConservativeQ(alpha=5.0, n_actions=10,
  calibrated=False)` adds a `critic_loss` term on SAC; `calibrated=True` is
  Cal-QL. One class keeps one source for the regulariser. No Lagrange variant
  until asked.
- **Wiring:** SAC's critic-loss batch gains `actor_state`, `next_observations`
  and `rng_key`. Cal-QL reads a replay field `return_to_go`, which
  `fill_replay` fills from the dataset. Deviation: online rows carry no floor,
  where Cal-QL computes their return when the episode ends.
- **Capabilities:** it needs `q_critic`, `actor` and `continuous_actions`. The
  discrete form (an exact logsumexp, on DQN) is a follow-up.
- **Probe:** `signed_rows`, with 100 sampled actions to shrink the estimator's
  bias. The critic's fixed point is Q = a·s − α(ρ/μ − 1), with ρ ∝ exp Q and μ
  the logged actions' density, solved on a grid. For Cal-QL, Q(s, π(s)) stays
  at or above the rows' return-to-go. Wrong answer: no penalty, Q = a·s.
  Tolerances come from a calibration ladder.
- **Budget:** about 150 source lines, 120 unit test, 70 probe and 50 docs.

### 2.5 ReBRAC: TD3 with TD3+BC's term, `TargetBC` and LayerNorm
- **Paper:** Tarasov, Kurenkov, Nikulin and Kolesnikov, NeurIPS 2023
  (arXiv:2305.09836). Code (JAX): tinkoff-ai/ReBRAC `src/algorithms/rebrac.py`
  (BC on the actor at β1; the critic target minus β2(π(s′) − a′)²; a LayerNorm
  critic; 3 × 256 layers; batch 1024) @pin. Also CORL `algorithms/offline/rebrac.py`.
- **Form:** 2.2's actor term, plus an `on_target` Extension `TargetBC(beta)`
  (y −= γ(1 − d)β(μ(s′) − a′)²), plus the `"layernorm"` token already on main,
  plus the paper's hyperparameters, documented as a recipe.
- **Data:** a′ comes from a new replay field `next_action`, the next row of the
  episode, which `fill_replay` fills. Online rows get it at the next step.
- **Capabilities:** `TargetBC` needs `q_critic`, `actor`, `continuous_actions`
  and `discounted`.
- **Probe:** a two-step chain with the actor frozen: a `CloningPretrain`, then
  actor learning rate 0. Then Q(s0, a) = r0(a) + γ(r1(π(s1)) − β(π(s1) − a1)²).
  Wrong answer: no penalty.
- **Budget:** about 90 source lines, 80 unit test, 50 probe and 40 docs.

### 2.6 GAIL: Extension `AdversarialReward` and a new `reward` phase
- **Paper:** Ho and Ermon, NeurIPS 2016 (arXiv:1606.03476). The authors' code is
  openai/imitation (TRPO, Theano), `policyopt/imitation.py`; maintained:
  HumanCompatibleAI/imitation `algorithms/adversarial/gail.py` (PPO) @pin.
- **Form:** an Extension. ext_state holds the discriminator D(s, a) and its
  optimiser. The rollout's rewards become the reference's −log D form. Uses PPO
  in place of the paper's TRPO (a deviation). It needs a new `reward` phase,
  which PPO folds first, before GAE.
- **Data:** expert demonstrations, as a `TransitionDataset` (obs, action).
- **Probe:** SignedActionEnv, whose own optimum is a = s, with the expert
  a_E = 0.6s. Truth: a(±1) = ±0.6. Wrong answers: the env's reward kept (±1);
  a frozen discriminator.
- **Budget:** about 250 source lines plus 80 for the phase, 150 unit test,
  60 probe and 60 docs.

## 3. Shared pieces

- **Offline mode** (needed by 2.2 to 2.5). Every replay agent gets
  `train_offline(dataset, n_updates, seed, logging_config)`. It fills the replay
  and the normalisers (PriorData's code), then scans `update`, evaluating every
  `log_frequency` updates. Online fine-tuning resumes from that state. The
  dataset enters as `build_resumable_train`'s `shared` input, as in
  TDMPC2MultiTask. About 100 source and 80 test lines. The alternative is no
  new code: `PriorData(offline_updates=N)` at `n_timesteps=0`, which gives no
  learning curve.
- **Replay fields** `return_to_go` (Cal-QL) and `next_action` (ReBRAC). Each is
  stored only when a module declares it, as `raw_next_obs` is today.
- **Docs.** A guide page, `docs/guide/offline-and-imitation.md`, on choosing a
  method and building datasets; algorithm pages for DAgger and IQL; README rows
  and the "Not in Ajax yet" line; `compatibility.md` regenerated.

## 4. Order (simplest first, DAgger first as asked)

| # | Method | Form | Source lines | Needs |
| --- | --- | --- | --- | --- |
| 1 | DAgger | agent | 250 | none |
| 2 | Offline mode | loop | 100 | none |
| 3 | TD3+BC | ImitationLoss options | 50 | 2 |
| 4 | IQL | agent | 300 | 2 |
| 5 | CQL / Cal-QL | Extension | 150 | 2 |
| 6 | ReBRAC | Extension and recipe | 90 | 2, 3 |
| 7 | GAIL | Extension and phase | 330 | none |

TD3+BC's term becomes ReBRAC's actor term. IQL comes before CQL because IQL's
probe is exact and CQL's estimator is approximate. GAIL comes last: it is the
only one that needs a new phase. The total is about 1,270 source lines.
