# Offline learning and imitation: design note (2026-10-19)

Proposal on main 1751d82; one method per branch and PR. References are from
standing knowledge, unverified until each `@pin` commit is read before its PR.

## 1. Does `ImitationLoss` already do DAgger?

DAgger (Ross, Gordon and Bagnell, AISTATS 2011, Alg. 3.1): from D = ∅, each
iteration i rolls out π_i = β_i π* + (1 − β_i) π̂_i, labels the visited states
with the expert's action, sets D ← D ∪ D_i and fits π̂_{i+1} on D, supervised
only. β_i = I(i = 1) is the parameter-free choice, β_i = p^(i−1) the other;
the bound needs mean β_i → 0 and a no-regret learner (FTL on D, or online).

| DAgger | `ImitationLoss(own expert)` on SAC, TD3, REDQ, ASAC | Same |
| --- | --- | --- |
| labels π*(s) on the learner's states | `a_expert` stored at collection, from the expert's tracked state | yes |
| D, every iteration's data | the replay, sampled uniformly, FIFO past `buffer_size` | within capacity |
| β_1 = 1, then p^(i−1), per step | expert prefill, `PriorData(record(expert))`; `EDGEExploration(gate="fixed")` | β_1 yes; mixing SAC only |
| fit on D | one gradient step per env step | online learner: allowed |
| supervised loss only, no reward | added to the RL loss: critic, reward, exploration noise | no |

PPO, APO, APG, UDRL (latest rollout) and AVG (latest step) keep no D.
**Verdict:** the earlier judgement holds for the replay agents: DAgger's
aggregation with β_i = I(i = 1), plus an RL loss. Pure DAgger, needing no
reward or critic, is a separate agent (§2.1); the docs will say so.

## 2. Candidates

**2.1 DAgger: agent `DAgger`.**
- *Reference:* no authors' code. HumanCompatibleAI/imitation
  `src/imitation/algorithms/dagger.py` (`ExponentialBetaSchedule`; per-env,
  per-step mixing in `InteractiveTrajectoryCollector`) @pin.
- *Form:* an agent; its update is the supervised loss alone. Surface:
  `DAgger(env, expert_policy, beta_decay=0.0, iteration_steps, actor_architecture,
  learning_rate, buffer_size, batch_size, n_envs, extensions=())`, β_i =
  beta_decay^(i−1) (0^0 = 1: the default is I(i = 1)). `expert_policy` is
  required: the algorithm's oracle.
- *TrainLoop:* `off_policy`, as UDRL. Each env executes the expert's action
  with probability β_i, else the actor's deterministic one; `a_expert` is
  stored from the expert's tracked state (stateful experts work). One update
  per tick: MSE(μ(s), a_E), cross-entropy for discrete actions. Logs
  `Eval/expert bias`, `dagger/beta`, `dagger/loss`. The replay is D;
  `PriorData(record(expert))` can seed it; NaN `a_expert` rows weigh 0.
- *Capabilities:* UDRL's `actor obs_networks off_policy replay` plus
  `agent_expert`; critic modules are refused.
- *Probe `dagger-off-path`:* one-hot chain: from 0, a > 0 leads to 1 then 2,
  a ≤ 0 to 3, reached only by a learner's mistake. Labels 0.8, 0.4, −0.2, −0.6.
  The actor starts as a CloningPretrain clone of −0.5 everywhere;
  `iteration_steps=1`. Truth a(0) = 0.8, a(1) = 0.4, a(3) = −0.6. Wrong:
  cloning the expert's states only, or β stuck at 1: a(3) = −0.5; labels from
  the executed action: −0.5 everywhere (both planted). A discrete twin follows.

**2.2 TD3+BC offline: two `ImitationLoss` options.**
- *Paper, code:* Fujimoto and Gu, NeurIPS 2021 (arXiv:2106.06860); sfujim/TD3_BC
  `TD3_BC.py` (λ = α / mean|Q(s, π(s))|, α = 2.5, loss −λQ + MSE(π(s), a)),
  `main.py` (states standardised by the dataset, std + 1e-3) @pin.
- *Form:* no agent. `target="replay"` clones the replayed action (logged data
  works); `alpha` sets coef = mean|Q(s, μ(s))| / α, held constant, no `/4`: the
  paper's loss over λ, which Adam nearly cancels (offline-proto `checks.py`).
  Replay agents add `actions` to the actor-loss batch. Recipe:
  `TD3(normalize_observations=True, extensions=(ImitationLoss(None,
  target="replay", alpha=2.5),))`, trained offline (§3). `alpha` needs
  `q_critic`, `target="replay"` needs `replay`.
- *Probe `td3bc-alpha-TD3`:* PriorData's `signed_rows` (one step, reward a·s,
  logged actions on a grid of mean 0): Q = a·s, π = λs/2, λ = α/|π|, so a(±1)
  = ±√(α/2) = ±0.5 at α = 0.5. Wrong: no BC ±1, BC alone 0, λ = α ±0.25.
- *Deviation:* Ajax's running normaliser (sqrt(var + 1e-8), seeded with the
  dataset), not std + 1e-3: they differ on a constant column (one setpoint).

**2.3 IQL: agent `IQL`.**
- *Paper, code:* Kostrikov, Nair and Levine, ICLR 2022 (arXiv:2110.06169); JAX
  ikostrikov/implicit_q_learning: `critic.py` (expectile loss, `update_v`,
  `update_q`), `actor.py` (weights exp(β(Q − V)) clipped at 100), `configs/`
  (τ 0.7, β 3 locomotion; τ 0.9, β 10 antmaze; cosine actor schedule) @pin.
- *Form:* an agent: its V, Q and actor losses replace SAC's. Surface: the
  paper's hyperparameters, `extensions=()`. On `off_policy`: offline (§3),
  then online fine-tuning with the same update, offline data kept (Sec. 5.3).
  Has `actor q_critic v_critic critic_ensemble discounted replay off_policy`.
- *Probe `iql-expectile`:* `signed_rows`, τ 0.9, β 10 (the clip binds): Q = a·s,
  V(±1) the τ-expectile of the 21 logged returns, the actor's mean Σwa/Σw
  with w = min(exp(β(as − V)), 100) (weighted Gaussian MLE, state-independent
  std). Wrong: τ = 0.5 (V = 0), no weights (0), no clip.

**2.4 CQL and Cal-QL: Extension `ConservativeQ`.**
- *Papers, code:* Kumar, Zhou, Tucker and Levine, NeurIPS 2020 (arXiv:2006.04779),
  CQL(H); Nakamoto et al., NeurIPS 2023 (arXiv:2303.05479).
  aviralkumar2907/CQL `d4rl/rlkit/torch/sac/cql.py` (10 uniform and 10 policy
  actions at s and s′, importance weights, `min_q_weight`, Lagrange option);
  nakamotoo/Cal-QL `JaxCQL/conservative_sac.py` (policy actions' Q floored at
  the Monte-Carlo return) @pin.
- *Form:* `ConservativeQ(alpha=5.0, n_actions=10, calibrated=False)`, a
  `critic_loss` term on SAC; `calibrated=True` is Cal-QL (one regulariser, one
  source; Lagrange only if asked). SAC's critic batch gains `actor_state`,
  `next_observations`, `rng_key`; Cal-QL reads a replay field `return_to_go`
  that `fill_replay` fills. Deviation: online rows get no floor (Cal-QL
  computes their return at the episode's end). Needs `q_critic`, `actor`,
  `continuous_actions`; the exact discrete form (DQN) is a follow-up.
- *Probe:* `signed_rows`, 100 sampled actions: Q = a·s − α(ρ/μ − 1), ρ ∝ exp Q,
  μ the logged density, solved on a grid; Cal-QL: Q(s, π(s)) ≥ the rows'
  return-to-go. Wrong: no penalty. Tolerances from a ladder.

**2.5 ReBRAC: TD3, 2.2's term, `TargetBC`, LayerNorm.**
- *Paper, code:* Tarasov, Kurenkov, Nikulin and Kolesnikov, NeurIPS 2023
  (arXiv:2305.09836); JAX tinkoff-ai/ReBRAC `src/algorithms/rebrac.py` (actor BC
  β1; target minus β2(π(s′) − a′)²; LayerNorm critic; 3 × 256; batch 1024)
  @pin; CORL `algorithms/offline/rebrac.py`.
- *Form:* an `on_target` Extension `TargetBC(beta)`: y −= γ(1 − d)β(μ(s′) − a′)²;
  with main's `"layernorm"` token and the paper's settings, a documented
  recipe. a′: replay field `next_action` (dataset: the episode's next row;
  online: the next step). Needs `q_critic actor continuous_actions discounted`.
- *Probe:* two-step chain, actor frozen (CloningPretrain, actor rate 0):
  Q(s0, a) = r0(a) + γ(r1(π(s1)) − β(π(s1) − a1)²). Wrong: no penalty.

**2.6 GAIL: Extension `AdversarialReward`, a `reward` phase.**
- *Paper, code:* Ho and Ermon, NeurIPS 2016 (arXiv:1606.03476); the authors'
  openai/imitation (TRPO, Theano) `policyopt/imitation.py`; maintained
  HumanCompatibleAI/imitation `algorithms/adversarial/gail.py` (PPO) @pin.
- *Form:* D(s, a) and its optimiser in ext_state; rewards become the reference's
  −log D form, in a new `reward` phase PPO folds before GAE; PPO for TRPO
  (deviation). Demonstrations: a TransitionDataset.
- *Probe:* SignedActionEnv (own optimum a = s), expert a_E = 0.6s: a(±1) = ±0.6.
  Wrong: env reward kept (±1), frozen discriminator.

## 3. Shared pieces

- **Offline mode** (2.2 to 2.5): every replay agent gets `train_offline(dataset,
  n_updates, seed, logging_config)`: fill the replay and normalisers
  (PriorData's code), scan `update` with evaluations every `log_frequency`
  updates; online fine-tuning resumes from its state. The dataset is
  `build_resumable_train`'s `shared` input, as in TDMPC2MultiTask. Alternative,
  no code: `PriorData(offline_updates=N)` at `n_timesteps=0`, without a curve.
- **Replay fields** `return_to_go`, `next_action`, stored only when declared.
- **Docs:** a guide page, DAgger and IQL pages, README rows, `compatibility.md`.

## 4. Order: simplest first, DAgger first as asked

Line budgets: source; unit tests, probe, docs.

| # | Method | Form | Source | Tests, probe, docs | Needs |
| --- | --- | --- | --- | --- | --- |
| 1 | DAgger | agent | 250 | 120, 70, 80 | none |
| 2 | Offline mode | loop | 100 | 80, 0, 30 | none |
| 3 | TD3+BC | ImitationLoss options | 50 | 60, 40, 40 | 2 |
| 4 | IQL | agent | 300 | 150, 60, 80 | 2 |
| 5 | CQL / Cal-QL | Extension | 150 | 120, 70, 50 | 2 |
| 6 | ReBRAC | Extension, recipe | 90 | 80, 50, 40 | 2, 3 |
| 7 | GAIL | Extension, phase | 330 | 150, 60, 60 | none |

TD3+BC's term is ReBRAC's actor term; IQL's probe is exact where CQL's
estimator is approximate; GAIL alone needs a new phase. About 1,270 lines.
