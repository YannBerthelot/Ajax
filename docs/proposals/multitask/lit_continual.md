# Sequential multi-task RL (continual / lifelong): literature check

Checked 2026-10-11. PDFs fetched from arXiv / NeurIPS proceedings and converted with pdftotext; section,
equation and table numbers below are from those texts. **[U]** = not opened or not verified in the body.

## 0. Short answer to Yann

Yes. "Task A for a block, then task B, ... while training on each" is the **continual (lifelong) RL**
setting. Its main continuous-control benchmark, Continual World, does exactly this, but the blocks are
measured in **environment steps** (1M per task), not episodes. Training on all tasks at once is the
**simultaneous / multi-task** regime. Both CLEAR and Continual World report it as a separate baseline
next to the sequential one. Between the two lie cyclic and interleaved schedules (block length →
1 step) and curricula (the order is chosen to help a target task). Meta-RL (RL², MAML) is a different
axis: it is about adapting *within* a few episodes, not about the training schedule.

## 1. Continual RL with a task sequence

### 1.1 Continual World (Wołczyk et al., NeurIPS 2021), https://arxiv.org/abs/2105.10919
- **Tasks.** 10 Meta-World tasks in a fixed order, chosen for varied forward transfer: hammer, push-wall,
  faucet-close, push-back, stick-pull, handle-press-side, push, shelf-place, window-close, peg-unplug-side
  (Sec 4.2). **CW20 = CW10 repeated twice** (Sec 4.2), so it already contains a revisit. There are also
  8 three-task "triplets" for fast iteration (Sec 4.2, App F).
- **Block length.** Δ = 1M steps per task, so T = 20M for CW20. Task i trains on t ∈ [(i−1)Δ, iΔ]
  (Sec 4.1). Episodes last 200 steps (App A.1), so the boundaries fall on episode ends.
- **Task ID.** Given. There is a separate output head per task (20 heads for CW20) for the actor and
  the critic (Sec 4.3, App A.4). One-hot input with a single head appears only as an ablation (App G.3).
  The observation is 12-D and does not reveal the task (Sec 4.4, "Input space").
- **Boundaries.** Known ("We rely on task boundaries", Sec 4.4). At every boundary the SAC replay buffer
  is emptied and Adam's statistics are reset. Each task starts with 10K uniform-random steps (App A.3,
  "Task handling" and "SAC training").
- **Metrics** (Sec 4.1). p_i(t) ∈ [0,1] is the success rate of task i. All tasks are evaluated every 20K
  steps on 10 stochastic trajectories (App A.3, "Policy evaluation").
  - Average performance P(t) = (1/N) Σ p_i(t) (eq. 1). P(T) is the tuning objective.
  - Forward transfer FT_i = (AUC_i − AUĈ_i) / (1 − AUĈ_i), against a **single-task reference run** (eq. 2).
  - Forgetting F_i = p_i(iΔ) − p_i(T) (eq. 3). Backward transfer was negligible (Sec 4.1).
  - A pairwise transfer matrix and a reference transfer RT = 0.46 (Sec 4.2, eq. 4).
- **Results** (Table 1, CW20, 20 seeds). PackNet 0.80. EWC 0.60. MT with PopArt 0.65, plain MT 0.51.
  L2 0.43. Perfect Memory 0.12. Fine-tuning 0.05 with forgetting 0.73. Most CL methods stop forgetting
  but have negative FT. Multi-task training needs PopArt because reward scales differ; the sequential
  setting sidesteps that problem (Sec 5, Sec 6, App H).
- **RL-specific findings** (Sec 6.1):
  - Regularise only the actor and let the critic adapt freely. A forgetful critic hurts revisits.
  - Emptying the buffer at a boundary plus uniform exploration lowers transfer even into the *same*
    task. Keeping the buffer, or exploring with the current policy, helps.
  - PackNet needs the task identity at evaluation and the number of tasks in advance (Sec 6).
- **Follow-up** (Wołczyk, Zając et al., NeurIPS 2022), https://arxiv.org/abs/2209.13900:
  - The critic carries most of the forward transfer: FT 0.20 vs actor 0.06 and exploration 0.09
    (Sec 4.1, Table 1). Freezing the critic body is very harmful (FT −1.29).
  - ClonEx-SAC = behaviour cloning on the actor + best-return exploration + no critic regularisation
    + multi-head (Sec 6).

### 1.2 Khetarpal, Riemer, Rish, Precup, JAIR 2022, https://arxiv.org/abs/2012.13490
- **General CRL problem** ⟨S(t), A(t), r(t), p(t), x(t), O(t)⟩: any element may vary in time (Def 1, eq. 8).
- **Form:** Lipschitz slow drift (Def 2), or **piecewise stationary with switch times t0, t1, ...** (Def 3,
  eq. 10). The task-block protocol is Def 3.
- **Taxonomy:** *scope* α ⊆ {S, A, r, p, x, O}, i.e. what changes (Def 4); *driver* β ∈ {stationary,
  passive, active, hybrid}, i.e. whether the change depends on the agent's actions (Def 5).
- A non-stationary MDP is dual to a POMDP whose hidden variable is time or the task (Prop 2), so an
  unknown task is a partial-observability problem.
- **Placement** (Sec 4.8): stationary p(z) is multi-task *and* meta-learning (4.8.1). Passive covers most
  continual-RL experiments, incl. switches at episode termination (semi-MDP) and change points (4.8.2).
  Active includes automatic curricula (4.8.3).
- **Methods** (Sec 5): explicit retention by parameter storage, distillation or rehearsal (5.1); shared
  structure (5.2); learning to learn, incl. context detection for unknown boundaries (5.3).
- **Metrics** (Sec 6.2): return alone is not enough; measure forward/backward transfer and skill reuse.
- **Newer survey:** Pan et al. 2025, https://arxiv.org/abs/2506.21872, a four-way method taxonomy by
  knowledge storage and transfer. Abstract only; the category names are **[U]**.

### 1.3 Other protocols in use (block length, and whether the agent sees the switch)
| Protocol | Domain | Block | Boundary/ID given to agent? | Source |
|---|---|---|---|---|
| CW10/CW20 | Meta-World, SAC | 1M steps | yes (heads), buffer+Adam reset | 2105.10919 Sec 4.1, 4.3, App A |
| Switching ALE (S-ALE) | Atari, Rainbow/DQN | 20M frames per visit, cyclic, 5-10 games; also 10M/50M | no reset of weights or buffer at switch | 2303.07507 Sec 3.1-3.2 |
| Game-mode loop | Atari modes | cyclic | milder non-stationarity | 2303.07507 Sec 3.3 |
| Slippery Ant | MuJoCo Ant-v3, PPO | friction changes every 2M steps (`change_time` default) | not signalled to the agent | github.com/shibhansh/loss-of-plasticity `lop/rl/run_ppo.py` (read via summary, about l.100-126) |
| Continual DMC (Plasticine) | DMC chains, e.g. dog stand→walk→run→trot | 1M steps | not stated in paper **[U]** | 2504.17490 App E.3 |
| Continual Procgen (Plasticine) | Procgen | new level every 2M steps | **[U]** | 2504.17490 (summary) |
| EWC Atari | 10 Atari games, random order with revisits | "extended periods" | no: an HMM task-recogniser infers it | 1612.00796 Sec 2.2 |
| CLEAR DMLab | 3 DMLab tasks, cyclic | equal experience per task | no (method needs none) | 1811.11682 Sec 1, 4.1 |

- Ajax note: Plasticine is PyTorch/CleanRL (2504.17490, summary).
- The Nature paper by Dohare et al. (2024, doi:10.1038/s41586-024-07711-7) could not be opened
  (redirected to a sign-in page). The RL facts above come from its official code. The config
  `lop/rl/cfg/ant/cbp.yml` sets n_steps 1e8 (the README says 50M), replacement rate 1e-4, maturity
  10000, Adam β1 = β2 = 0.99 and weight decay 1e-4.

### 1.4 How task boundaries are defined
- **By steps** is the standard for continual control: CW 1M, Plasticine 1M, S-ALE 20M frames, Slippery
  Ant 2M. Switches land on episode ends only when the block is a multiple of the episode length (CW:
  200 | 1M). Slippery Ant recreates the env at a switch; whether that cuts an episode short is **[U]**.
- **By episodes or trials:** meta-RL's unit is a trial of n episodes on one MDP (RL² Sec 2.2).
  Khetarpal 4.8.2 allows switches at episode termination.
- **Known:** CW, task-ID heads, PackNet, progressive nets, EWC as usually implemented. **Unknown:** S-ALE,
  Slippery Ant, Permuted MNIST (Dohare preprint Sec 3: "no indication was given"), CLEAR (Sec 1, "No
  task information"). **Inferred:** the EWC Atari HMM recogniser (Sec 2.2).
- Many methods need the boundary *only during training* (to snapshot or prune), plus the task ID *at
  evaluation* (to pick a head, mask or column). These are two separate signals.

## 2. Plasticity loss in sequential tasks

- **Lyle, Rowland, Dabney, ICLR 2022 ("capacity loss")**, https://arxiv.org/abs/2204.09560. A network
  trained on a *sequence of targets* loses its ability to fit new ones. Measured as target-fitting
  capacity on random targets from agent checkpoints (Def 1, Sec 3.1). Metric: **feature rank**, the count
  of singular values of φ(X)/√n above ε (Def 2, eqs 4-5). Fix: InFeR, which regresses auxiliary heads
  toward their initial values (Sec 4.1). Single-task Atari: deep RL is already a sequence of targets.
- **Lyle et al., ICML 2023**, https://arxiv.org/abs/2303.01486. Abstract: tied to loss-landscape
  curvature, often *without* saturated units; layer normalisation helps; evaluated on ALE. Body **[U]**.
- **Dohare et al., Nature 632:768-774 (2024)**, read via the preprint https://arxiv.org/abs/2306.13812
  (v3) and Research Square rs-3256479.
  - Protocols: Continual ImageNet, thousands of binary tasks (Sec 2); Online Permuted MNIST, 800 tasks
    of 60k examples, no switch signal (Sec 3); Slowly-Changing Regression; RL from the code (see 1.3).
  - Correlates (Sec 4): % dead ReLU units (zero on all of 2000 samples, counted at each task start);
    weight magnitude; **effective rank** erank(Φ) = exp(H(σ/‖σ‖₁)) (eq. 1).
  - Existing fixes (Sec 5): L2 and shrink-and-perturb help; Adam and dropout make things worse.
    Incremental S&P adds noise "on each update".
  - **Continual backprop** (Sec 6, Alg 1): at every update, reinitialise a fraction ρ (replacement rate)
    of the lowest-utility units per layer. New units get zero outgoing weights (function unchanged) and
    are protected for m updates (maturity). Utility = running average of |activation| × Σ|outgoing
    weights|, mean-corrected (eqs 2-5).
- **Nikishin et al., ICML 2022 ("primacy bias")**, https://arxiv.org/abs/2205.07802. **Single-task**:
  the agent overfits its early data (Sec 3); no task sequence. Fix (Sec 4, 5.1): periodically
  re-initialise the last layer(s) and their optimizer state, and **keep the replay buffer**. SPR: last
  linear layer every 2×10⁴ steps; SAC: all networks every 2×10⁵ steps; DrQ: last 3 layers. This
  enables higher replay ratios (Sec 5.2, Tables 1-2).
- **Shrink-and-perturb (Ash & Adams, NeurIPS 2020)**,
  https://proceedings.neurips.cc/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf. At each
  round of new data: θ_t ← λ·θ_{t−1} + p_t, with p_t ~ N(0, σ²) and 0 < λ < 1 (Sec 4). Supervised
  warm-starting, not RL.
- **Dormant neurons / ReDo (Sokar et al., ICML 2023)**, https://arxiv.org/abs/2302.12902. τ-dormant:
  mean |activation| normalised by the layer mean ≤ τ (Def 3.1-3.2); the count grows under non-stationary
  targets (Sec 3). ReDo, every F steps: reinitialise incoming weights and zero outgoing weights of
  dormant units (Sec 4, Alg 1). Tested on DQN, DrQ(ε) and SAC.
- **Kumar et al., ICLR 2021**, https://arxiv.org/abs/2010.14498 (abstract only). The **srank** of value
  features collapses under bootstrapped updates ("implicit under-parameterization").
- **Abbas et al. 2023**, https://arxiv.org/abs/2303.07507 (S-ALE). Weight change, gradients and
  activations collapse (Sec 4). CReLU mitigates (Sec 5). Forgetting stays unsolved (Sec 6). Compared
  with an idealised reset-at-every-switch agent (Sec 3.2).
- **Takeaway.** Plasticity work uses *long* sequences: hundreds of switches, or cycles with revisits. Its
  diagnostics (dormant fraction, feature/effective/stable rank, weight norm, gradient norm) are cheap
  per-layer statistics on a fixed probe batch. Plasticine bundles six (2504.17490 App D.1-D.6).

## 3. Methods against forgetting: what they need, and where they hook in

| Method | Needs | Acts at | Task ID at eval? | Source |
|---|---|---|---|---|
| L2 | anchor θ* (one snapshot; CW keeps only the last) | loss term every update; snapshot at boundary | no (but CW uses heads) | 2105.10919 App B.1 eq 6-7 |
| EWC | θ* + diagonal Fisher F, computed **at each task switch** from samples (CW: 2560 from the buffer). CW stores ΣF and the last θ, so memory is O(1) in tasks | λ/2 Σ F_i(θ_i−θ*_i)² added to the loss every update (EWC eq 3); computed at boundary | original Atari version used task-specific gains and biases | 1612.00796 Sec 2, 2.2; 2105.10919 App B.1 |
| PackNet | per-task binary masks (≤log2 N bits per param), N fixed in advance, task boundary, extra fine-tune (CW: 100K steps on stored data) | at boundary: prune by magnitude (CW keeps 5%/task), fine-tune, freeze; every update: mask gradients | **yes** (mask + head) | 1711.05769 Sec 3; 2105.10919 App B.2, Sec 6 |
| Progressive nets | a new column per task (parameters grow), frozen old columns, lateral adapters | at boundary: freeze and add a column; forward pass uses laterals (eq 2) | **yes** ("requires knowledge of the task label") | 1606.04671 Sec 2, "Limitations" |
| Replay: Perfect Memory / reservoir | a large buffer, task-tagged in CW | buffer policy: don't clear at boundary; reservoir insertion | no | 2105.10919 Sec 5, App B.3 |
| CLEAR | buffer of all past experience (reservoir; 1-in-200 still works) + stored behaviour policy µ and value V_replay | each batch mixes new/replay 50-50 (75-25 vs P&C); V-trace on replay + KL(µ‖π) and value-L2 cloning losses | no boundaries, no ID | 1811.11682 Sec 3, 4.4-4.6 |
| Resets (whole or partial) | step counter, or a boundary | periodically (Nikishin), or at each switch (S-ALE reference) | no | 2205.07802 Sec 5.1; 2303.07507 Sec 3.2 |
| CBP / ReDo / S&P | per-unit utility statistics (CBP), or nothing | inside every update (CBP, incremental S&P) or every F steps (ReDo) | no | 2306.13812 Sec 5-6; 2302.12902 Sec 4 |

- **Which network to protect is agent-specific.** CW protects only the actor and leaves the critic free
  (Sec 6.1). CLEAR clones both policy and value (Sec 3). For DreamerV3 / TD-MPC2 (world model, actor,
  critic) I found no source in this pass **[U]**.
- **Replay on on-policy agents** (PPO, APO, APG) needs an off-policy correction; CLEAR uses V-trace on
  IMPALA (Sec 3).
- **What CW resets at a boundary** (App A.3): buffer, Adam state, initial uniform exploration. These are
  protocol choices that move FT, not just method choices.

## 4. Neighbouring settings

- **Simultaneous vs sequential.** CLEAR Fig 1 (Sec 4.1) compares separate, simultaneous and
  sequential-cyclic training *with equal experience per task*: simultaneous ≈ separate on DMLab;
  sequential forgets.
- **Interleaved or cyclic.** CW20 is one revisit; S-ALE cycles 5-10 games with 10M/20M/50M-frame
  visits (Sec 3.2); EWC Atari uses a random order with returns (Sec 2.2). The block length interpolates
  from per-batch mixing (multi-task) to long blocks.
- **Curriculum** (Narvekar et al., JMLR 2020, https://arxiv.org/abs/2003.04960, abstract only): orders
  tasks to speed up a *target* task, in three parts (task generation, sequencing, transfer). Success is
  judged on the target, not on retaining all tasks. An automatic curriculum makes the schedule depend on
  the agent: Khetarpal's "active" driver (4.8.3).
- **Meta-RL, fast adaptation.**
  - **RL²** (https://arxiv.org/abs/1611.02779, Sec 2.2): a trial is n episodes of one sampled MDP; the
    RNN gets (s, a, r, done); its hidden state persists across episodes and resets between trials. The
    objective is trial return, optimised with TRPO (Sec 2.4).
  - **MAML-RL** (https://arxiv.org/abs/1703.03400, Alg 3, Sec 5.3): the inner step is a policy gradient
    on K rollouts (20 per step, H = 200; 40 for ant fwd/back); the outer step differentiates through it.
  - Meta-training samples from a **stationary** p(z) (Khetarpal 4.8.1): the all-at-once regime plus an
    adaptation-shaped objective, *not* a sequential schedule.

## 5. Placement in the layered view

Layers: (L1) task set and interface; (L2) data regime; (L3) per-task balancing;
(L4) gradient interference; (L5) architecture.

| Layer | Simultaneous multi-task | Sequential / continual adds |
|---|---|---|
| L1 interface | task ID usually observed (one-hot) | ID/boundary may be hidden (S-ALE, Slippery Ant) or train-only; eval-time ID is a separate switch (PackNet, heads, columns) |
| L2 data regime | per-slot task assignment | **schedule over time** (block length, order, revisits); buffer policy at the boundary (clear / keep / reservoir); optimizer-state reset |
| L3 balancing | reward scales across tasks (PopArt is essential in CW MT, App H) | largely sidestepped, since one task at a time (CW Sec 5); returns with revisits |
| L4 interference | conflicting gradients at the same time (Khetarpal 4.8.1) | interference spread **across time** = forgetting; handled by L2 / EWC / PackNet / CLEAR penalties and masks |
| L5 architecture | shared trunk + heads or task embedding | multi-head, masks, growing columns; plasticity tools (resets, CBP, ReDo, CReLU, LayerNorm) |

**Does "a task schedule over time" cover these settings as one axis?** For the *data regime*, mostly
yes. Simultaneous, sequential blocks, cyclic, interleaved and a fixed curriculum are all a
function `task_id(t, env_slot, sampler_state)`:
- simultaneous: depends on the slot only;
- blocks: ⌊t/Δ⌋;
- cyclic: ⌊t/Δ⌋ mod K;
- fixed curriculum: a chosen order.

In a compiled JAX loop this can be a pure function of the step counter, with the sampler state in the
carry. With K known at compile time, there is no recompilation. Switches should happen at env resets,
so a switch does not change the MDP mid-episode. That is my inference; CW gets this for free because
200 divides 1M.

**What else the literature needs beyond the schedule:**
1. **Boundary event and task-ID exposure, set separately for training and evaluation.** The event
   drives EWC/L2 snapshots, PackNet pruning, column freezing, boundary resets and CW's buffer/Adam
   resets. Exposure modes: none (CLEAR, CBP), boundary-only, or ID at train and eval.
2. **Boundary hooks on the training state:** snapshot anchor and Fisher (needs a buffer sample); clear
   or keep the buffer; reset the optimizer or networks; start a fine-tune phase (PackNet, 100K steps).
   These are Extension-shaped, not agent kwargs. Some run *outside* normal env stepping (PackNet
   fine-tunes on stored data).
3. **Per-task state with static shape:** heads (K), masks (K), anchors (1 summed, CW eq 7), columns (K,
   preallocated, since progressive nets grow). PackNet and multi-head both need K up front (CW Sec 6).
4. **An evaluation matrix p_i(t):** evaluate *every* task every E steps (CW: 20K steps, 10 episodes).
   Average performance, forgetting, FT and the transfer matrix come from it plus **separate single-task
   reference runs** (FT's AUĈ_i). It is the protocol's defining output; current-task return hides
   forgetting.
5. **Plasticity probes:** dormant fraction, effective/feature/stable rank and weight norm on a fixed
   probe batch, per layer.

**Where the view breaks:**
- **Plasticity loss happens without a task schedule**, in single-task deep RL too (Nikishin, Kumar,
  Sokar, Lyle 2022). Its remedies are periodic interventions keyed to the step count, not to tasks. They
  belong at L5 / in Extensions, orthogonal to the schedule axis.
- **Unknown or drifting boundaries.** Lipschitz drift (Khetarpal Def 2) has no blocks. A hidden switch
  makes the problem a POMDP (Prop 2) that may need context inference (Sec 5.3.1; the EWC HMM). The
  schedule then lives in the env, invisible to the agent, but the protocol must still record it.
- **Adaptive curricula** close the loop: the schedule depends on agent metrics (active driver), so it is
  not a fixed function of t.
- **Meta-RL** is off this axis. It changes the *objective* (trial return), the *interface* (r and done fed
  back, memory across episodes) and the *episode/trial structure*; its training data is stationary.
- **The metric is the setting.** The same schedule reads as "multi-task" or "continual" depending on
  whether you measure the final average, forgetting or FT. The schedule axis does not carry this.
- **Protocol knobs that move the numbers** but are not part of the schedule: buffer clearing and
  uniform restart exploration (CW Sec 6.1, App D.1); critic regularisation on or off (CW Sec 6.1;
  2209.13900 Sec 4.1); Adam reset (CW App A.3); Adam's own effect on plasticity (Dohare Sec 5).

## Sources read
2105.10919; 2209.13900; 2012.13490; 2204.09560; 2306.13812 (+ researchsquare rs-3256479);
github.com/shibhansh/loss-of-plasticity (`lop/rl/README.md`, `cfg/ant/cbp.yml`, `run_ppo.py` via summary);
2205.07802; 2302.12902; 2303.07507; 1910.08475 (NeurIPS 2020 PDF); 1612.00796; 1711.05769; 1606.04671;
1811.11682; 1611.02779; 1703.03400; 2504.17490 (HTML, via summary).
Abstract only: 2303.01486, 2010.14498, 2003.04960, 2506.21872. Not opened: the Nature version of Dohare
et al. 2024.
