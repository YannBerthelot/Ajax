# Multi-task RL literature: state of the art, model-based, offline and distillation

Scope: checks the five-layer hypothesis (1 task interface, 2 data regime, 3 per-task balancing,
4 gradient interference, 5 architecture) against TD-MPC2 and its follow-up, recent online
multi-task continuous control, offline/generalist models and distillation. Read 2026-10-11.

How each claim was checked:
**[V]** I read the code lines or the full paper text myself (TD-MPC2 code, Newt PDF and code).
**[S]** I read the paper's HTML through the web-fetch summariser. The section numbers are the
ones it reported; spot-check them before quoting them on.
**[U]** Unverified (my recollection, or the source would not open).

Sources (the keys used below):
- TD2-P: TD-MPC2 paper, https://arxiv.org/html/2310.16828
- TD2-C: https://github.com/nicklashansen/tdmpc2 at `main@e9f5932` (fetched today). Ajax pins the
  paper-era `5f6fade`, so its line numbers may differ by a few.
- NEWT-P: Hansen, Su, Wang, "Learning Massively Multitask World Models for Continuous Control",
  ICLR 2026, https://www.nicklashansen.com/NewtWM/newt.pdf (openreview MPabX9LEds)
- NEWT-C: https://github.com/nicklashansen/newt at `1d3fc05`
- BRC: https://arxiv.org/html/2505.23150 · MOORE: https://arxiv.org/html/2311.11385v2
- PACO: https://ar5iv.labs.arxiv.org/html/2210.11653 · CARE: https://ar5iv.labs.arxiv.org/html/2102.06177
- MRQ-MT: https://arxiv.org/html/2606.05555 (Obando-Ceron, Li, Fujimoto, Bacon, Courville, Castro, June 2026)
- MTB (MTBench): https://arxiv.org/html/2507.23172 · MW+ (Meta-World+): https://arxiv.org/html/2505.11289
- CDS: https://ar5iv.labs.arxiv.org/html/2109.08128 · GATO: https://ar5iv.labs.arxiv.org/html/2205.06175
- MGDT: https://ar5iv.labs.arxiv.org/html/2205.15241 · TDOPT: https://arxiv.org/abs/2507.01823 (abstract only)
- PD (Policy Distillation): https://ar5iv.labs.arxiv.org/html/1511.06295
- AM (Actor-Mimic): https://ar5iv.labs.arxiv.org/html/1511.06342 · DISTRAL: https://ar5iv.labs.arxiv.org/html/1707.04175

## 1. TD-MPC2 multi-task (Hansen, Su, Wang, ICLR 2024)

| Item | Fact | Source |
|---|---|---|
| Task sets | mt30 is 19 original DMC tasks plus 11 custom ones. mt80 is mt30 plus the 50 Meta-World tasks. | [V] TD2-C `common/__init__.py:26-60`; [S] TD2-P App. C Table 2 |
| Dataset origin | Replay buffers of 240 single-task TD-MPC2 agents, holding behaviour "from random to expert". | [S] TD2-P Sec. 4.1 |
| Dataset size | mt80 holds 545M transitions and mt30 holds 345M. The loader's buffers are 550,450,000 = 5.45M episodes × 101 (mt80) and 345,690,000 = 690k × 501 (mt30), i.e. 100-step and 500-step episodes plus the first observation. That matches the paper's numbers. | [S] TD2-P Sec. 4.1; [V] TD2-C `trainer/offline_trainer.py:53-54,59` |
| Seeds per task | 240 / 80 = 3 agents per task in mt80. That is plausible, but I did not find it stated. It is also unclear how DMC's 500-step episodes became 101-step rows in mt80. | [U] |
| Regime | Offline only. `train.py` picks `OfflineTrainer` whenever the task set is multi-task, and the trainer asserts mt30 or mt80. The batch is 4× larger (1024) and nothing else changes. | [V] TD2-C `train.py:52`, `offline_trainer.py:69-70`; [S] TD2-P Sec. 4.1 |
| Task embedding | Learned `nn.Embedding(len(tasks), task_dim, max_norm=1)`, concatenated to the input of the encoder, dynamics, reward, policy and Q. The paper uses dim 96. The code uses 64 for mt30 at 5M, 19M and 48M, which only serves the released checkpoints (Ajax records this as deviation T14). | [V] TD2-C `world_model.py:21,26-30,88-101`, `parser.py:75`; [S] TD2-P Sec. 3.3, Table 9 |
| Padding and masks | Observations are zero-padded at the end to the largest dim. The action space is the largest one, and task i uses its leading `action_dims[i]` dims: mask on the policy mean, std and noise, mask on the planner samples, mean and std, and the env receives only that prefix. | [V] TD2-C `envs/wrappers/multitask.py:44-47,56`, `world_model.py:22-24,158-162`, `tdmpc2.py:180-181,195-197` |
| Per-task discount | γ_i = clip((T_i/5 − 1)/(T_i/5), [0.95, 0.995]), with T_i the episode length after action repeat. It is used in both the TD target and planning. | [V] TD2-C `tdmpc2.py:35-37,57-70,131,256`; [S] TD2-P App. H Eq. 7 |
| Model sizes | 1, 5, 19, 48 and 317M parameters, set by enc/mlp/latent width and the number of Qs. The mt30 19M model has latent 512. | [V] TD2-C `common/__init__.py:1-24`, `parser.py:67-68` |
| Evaluation | Every task in turn, `eval_episodes=10` per task, with planning (`mpc: true`). The score averages DMC returns scaled to [0,100] with Meta-World success rates. | [V] TD2-C `offline_trainer.py:22-40`, `config.yaml:11,34`, `tdmpc2.py:114-115`; [S] TD2-P Sec. 4.1 |
| Online multi-task | **None in this repo.** The only online stage is fine-tuning the 19M multi-task model *single-task* on 10 held-out tasks for 20k steps, with an empty buffer and the new task's embedding initialised from a semantically similar task. Single-task evaluation of multi-task checkpoints is "not supported". | [S] TD2-P App. E, Fig. 8; [V] TD2-C `README.md:108` |

**Follow-up by the same authors, which is the online answer: Newt (ICLR 2026).** It extends TD-MPC2 to
*online* RL on 200 tasks at once [V NEWT-P Sec. 3, 2.1]:
- **Task interface.** A frozen CLIP-ViT/B text embedding (512 dims) of a per-task instruction replaces
  the learned ID. It is loaded into a frozen embedding table [V NEWT-P Sec. 2.2, 3.1 Eq. 1;
  NEWT-C `common/world_model.py:22-28`, `config.py:87`]. States are padded to 128 dims and actions to
  16 dims, with prefix action masks [V NEWT-P Sec. 4 "Implementation details"; NEWT-C
  `envs/wrappers/vectorized_multitask.py:36-43`, `world_model.py:30-35`].
- **Data regime.** There is one env per task (`num_envs = num_tasks`), 100M env steps in total, one shared
  10M buffer sampled uniformly, UTD 0.075 and batch 1024 [V NEWT-C `vectorized_multitask.py:65,75`,
  `config.py:159-160`; NEWT-P Sec. 4.1, Table 7]. Demonstrations are central: 10-40 per task (4,020 in
  total), filtered expert rollouts of 200 single-task TD-MPC2 agents trained for 5M steps each
  [V NEWT-P Sec. 2.3, App. C]. Newt uses them in four ways [V NEWT-P Sec. 3.2]:
  1. model-based pretraining with the Q term off;
  2. a planner biased toward the pretrained policy, annealed over the first 12%;
  3. separate demo and online buffers sampled 50/50;
  4. a BC term in the policy loss (Eq. 3).
- **Balancing.** Per-task γ: the domain default if there is one, otherwise the TD-MPC2 heuristic
  [V NEWT-P Sec. 3.1, App. F Eq. 4; NEWT-C `config.py:184-193`]. The cross-entropy (two-hot)
  reward and value losses are there because "reward distributions may be drastically different
  between tasks" [V NEWT-P Sec. 3.1].
- **Gradients.** No surgery is mentioned [V, full text grep].
- **Architecture.** The TD-MPC2 MLPs at 20M parameters by default. Model and batch scaling help more
  as the number of tasks grows [V NEWT-P Sec. 4.1 (1), Fig. 8].
- **Baselines.** PPO (cleanRL) and FastTD3 had to be *extended* with language conditioning and per-task
  γ. PPO also needed tanh-Normal, 3.5M parameters, 128-step rollouts, rewards scaled by 0.1 and retuning.
  FastTD3 needed 8-step returns [V NEWT-P Sec. 4, App. D]. Newt beats both [V NEWT-P Sec. 4.1, Fig. 7].

## 2. Online multi-task continuous control, the strongest methods

| Method | What changes | Base, benchmark, result | Source |
|---|---|---|---|
| CARE (Sodhani 2021) | L1: RoBERTa embedding of a task description (768 dims, then projected). L5: k encoders mixed by attention on the context, with the context concatenated to the mixed encoding. L3: per-task SAC α. | SAC. MT10 0.84, MT50 0.54 at 2M steps. | [S] CARE Sec. 3.1-3.2, Sec. 4, Table 9 |
| PaCo (Sun 2022) | L5: θ_τ = Φ w_τ, K shared parameter vectors and a per-task composition vector picked by task ID. L3: per-task α, per-task buffers, *loss mask-out* of unstable tasks and w-reset (Eq. 3). | SAC, 10 parallel envs. MT10-rand 85.4, MT50-rand 57.3. | [S] PACO method section, Eq. 3 |
| MOORE (Hendawy 2024) | L5: k experts orthogonalised by Gram-Schmidt and mixed by per-task weights. The one-hot is concatenated after the representation, single- or multi-head. L3: per-task α, one buffer per task sampled equally. Compared against PCGrad. | SAC (Meta-World), PPO (MiniGrid), MushroomRL. MT10-rand 88.7, MT50-rand 72.9 at 20M. | [S] MOORE Sec. 4-5, Tables 1-2 |
| BRC (Nauman 2025) | L5: a *bigger* residual LayerNorm MLP (BroNet, 1M to 1B parameters). L3: categorical cross-entropy critic and per-task return normalisation r·V_max/Ḡ_i. L1: a learned task embedding concatenated to the observation for actor and critic. L2: all tasks in parallel, shared buffer, 4× batch. L4: none. Conflict rates fall as the parameter count grows. | SAC (DrQ for Atari). 283 tasks: MW50, DMC, HumanoidBench, ShadowHand, Atari. Beats PCGrad, PaCo, CTPG and SimbaV2. Embedding-only tuning transfers to new tasks. | [S] BRC Sec. 3 (Fig. 5), Sec. 4, App. C.2, C.4 |
| MR.Q multitask (Obando-Ceron 2026) | L1: learned e(τ). L5: MR.Q's predictive (model-based) representation plus a high-capacity value function. No surgery. | Claims a 5-8% lead over Newt on MMBench at 10M steps and better zero-shot transfer. This is the authors' claim. | [S] MRQ-MT Sec. 2, 4, Fig. 2, 8 |
| MTBench (2025) | Benchmark. MT-PPO, MT-GRPO, MT-SAC and MT-PQN, each × {PCGrad, CAGrad, FAMO, SM, CARE, PaCo, MOORE}. Blocks of envs per task, with a one-hot appended to the observation. | Massively parallel PPO wins on wall clock. The critic is the bottleneck. Gradient methods help consistently. | [S] MTB Sec. 3.3, 4.1-4.2, Fig. 1, 4-6 |
| Meta-World+ (2025) | Benchmark audit: results depend on the reward version (V1 vs V2). Re-ran MTMHSAC, SM, MOORE, PaCo and PCGrad with 10 seeds. | SM and MOORE are above MTMHSAC on V2, with the gap narrowing on bigger task sets. | [S] MW+ Sec. 5.1, Table 1 |

**Strongest simple recipe (online, state-based, no demonstrations): BRC.** It is plain SAC with
- a wide residual LayerNorm critic,
- a categorical critic,
- per-task return normalisation,
- a learned task embedding concatenated to the input,
- a shared buffer stepped on all tasks in parallel,
- a 4× batch.

The modular architectures (CARE, PaCo, MOORE) are now baselines that the scaled networks beat [S BRC
Sec. 4]. With demonstrations, the model-based state of the art is Newt, and MR.Q-MT reports beating
it (one claim, not reproduced). None of the 2025-26 leaders uses gradient surgery.

## 3. Offline multi-task RL and generalist sequence models

- **CDS** (Yu 2021). The data regime is per-task offline datasets. Transitions from task j are
  *relabelled* with task i's reward, which needs the reward function. A transition is shared only if
  its conservative Q value for task i is above a top-k quantile of task i's own data. The base is CQL,
  and the policy is π(a|s,i): one-hot ID, or separate networks. Naive sharing "performs surprisingly
  poorly" [S CDS abstract, method section].
- **Gato** (Reed 2022). Offline behaviour cloning on episodes from SoTA specialist agents (MPO, Muesli),
  keeping only those with return ≥ 80% of the expert's. There is **no task ID**. 25% of sequences
  carry a *prompt*, an episode from the same source agent. Continuous values go through mu-law and
  1024 bins. The loss covers only actions and text. 1.2B parameters
  [S GATO Sec. 2.1, 2.3, 3.1].
- **Multi-Game DT** (Lee 2022). Offline on DQN-Replay from 41 Atari games (4.1B steps), with a shared
  18-action space and **no game ID** (inferred from pixels). It is conditioned on return, and inference
  samples "high-but-plausible" returns. 10M, 40M and 200M models. Fine-tuning to new games is offline,
  with 1% of the data [S MGDT].
- **TD-MPC-Opt** (2025). Distils the mt30 317M TD-MPC2 into 1M parameters (normalised score 28.45 vs
  18.93) with FP16 quantisation. The loss and the regime are unread [S TDOPT abstract; rest U].

## 4. Distillation

| Method | Stages | Who acts during collection | What the student sees and fits | Source |
|---|---|---|---|---|
| Policy Distillation (Rusu 2016) | 1. One DQN per game. 2. Distil. There is also an "online" variant that tracks a teacher still training. | **The teacher** (greedy, 5% random) | Frames and the teacher's Q for every action. Loss: KL to softmax(q/τ), τ = 0.01 (also MSE and NLL). The multi-task student is a shared trunk with **one head per game**, selected by game ID. | [S] PD |
| Actor-Mimic (Parisotto 2016) | 1. One expert DQN per game. 2. The multi-task AMN. 3. Optional: the AMN initialises a DQN on a new game. | **The student AMN**, ε-greedy | Cross-entropy to the Boltzmann of the expert's Q, plus regression onto the expert's features. **A single 18-way output** with invalid actions masked. 8 source games. | [S] AM |
| Distral (Teh 2017) | **One stage, joint.** The task policies π_i and a shared π_0 train at the same time. | Each π_i in its own task | Each π_i maximises return minus KL to π_0 plus entropy (α, β). π_0 is the maximum-likelihood "centroid" of the π_i's visitation distributions. A two-column parameterisation: π_0 logits plus task logits. A3C and soft Q. | [S] DISTRAL; I did not check whether π_0 sees the task [U] |

## 5. Layers 1 to 5 for each method

"Pad" means observations zero-padded and actions prefix-masked. "Same body" means one embodiment, so no padding.

| Method | L1 interface | L2 data regime | L3 balancing | L4 grad | L5 architecture |
|---|---|---|---|---|---|
| TD-MPC2 MT | learned ID embedding (≤1 norm), pad | offline, pooled single-task buffers | per-task γ, two-hot CE reward and value, 4× batch | none | one MLP world model, 1-317M |
| Newt | frozen CLIP language, pad | demos → pretrain → online on all tasks (1 env per task), demo/online 50/50 | per-task γ, CE losses, BC term | none | TD-MPC2 MLP, 20M |
| CARE | RoBERTa task description | online, all tasks | per-task α | none | mixture of k encoders |
| PaCo | ID → w_τ | online, 10 envs | per-task α and buffer, loss mask-out, w-reset | none | parameter composition Φw |
| MOORE | one-hot (weights and concat) | online | per-task α, one buffer per task sampled equally | (PCGrad compared) | orthogonal experts, SH or MH |
| BRC | learned embedding, same body | online, all tasks in parallel, shared buffer | per-task return norm, categorical critic, 4× batch | none (conflicts fall with size) | BroNet, up to 1B |
| MR.Q-MT | learned e(τ) | online | not read | none | predictive representation plus a big critic |
| CDS | one-hot or separate networks | offline + relabelled sharing | conservative routing | none | per method |
| Gato | prompt (demo), tokens | offline BC on filtered expert data | return filter ≥80% | none | 1.2B transformer |
| MGDT | none (pixels), shared 18 actions | offline DQN-Replay | return conditioning | none | DT, 10-200M |
| Policy Distillation | ID → head | 2-stage, teacher collects | KL(τ) to teacher Q | none | trunk plus heads |
| Actor-Mimic | shared 18-way, masked | 2-stage, student collects | CE plus feature regression | none | one network |
| Distral | per-task networks + π_0 | joint online, one policy per task | KL to π_0 plus entropy | none | two-column |

## 6. Where the hypothesis is wrong or incomplete

1. **"Every agent then becomes its published multi-task baseline unchanged" is false.** Every
   multi-task recipe I read changes the update using the task index:
   - per-task γ (TD-MPC2, Newt, and Newt's PPO and FastTD3 baselines);
   - per-task α (CARE, PaCo, MOORE);
   - one buffer per task with equal sampling (MOORE, PaCo);
   - per-task return normalisation (BRC);
   - a 4× batch (TD-MPC2, BRC).

   PPO additionally needed reward scaling and retuning (Newt App. D). So the environment must hand the
   agent the **task index as its own field, plus static per-task metadata**: episode length (hence γ),
   action dims, and the normalised-score spec. A one-hot hidden in the observation is not enough.
   Ajax's `TDMPC2/multitask.py:TaskSet` already holds this metadata; it belongs in the shared layer.
2. **Layer 1 options are wider than one-hot or learned:** a frozen language embedding (CARE, Newt),
   prompt conditioning (Gato), or no ID at all (MGDT). The embedding is also the transfer handle:
   initialise it from a similar task (TD-MPC2 App. E) or tune only the embedding (BRC).
3. **Padding and masking is not universal.** BRC, CARE, PaCo and MOORE work on single-embodiment suites
   and need none. Padding matters only when embodiments differ, as in TD-MPC2 mt80, Newt, and a
   Playground set spanning DM Control, locomotion and manipulation.
4. **Distral is not distillation from per-task experts.** It is one joint online stage with N+1
   policies, so it is a multi-policy *agent*, not a protocol over a conditioned agent. Actor-Mimic (the
   student collects the data) and Policy Distillation (the teacher collects it) differ in that respect.
5. **Layer 2 is missing three regimes:**
   - demonstration-bootstrapped online (Newt);
   - cross-task relabelling (CDS, which needs reward functions; Playground exposes them);
   - held-out-task fine-tuning as a stage (TD-MPC2 App. E, MGDT, BRC).
6. **Layer 4 has weak support at the state of the art.** The 2025-26 leaders do not use it. BRC finds
   conflicts fall with scale, while MTBench finds gradient methods help in its massively parallel PPO
   and SAC regime. Keep it as an optional Extension and give it low priority.
7. **Layer 5:** the state of the art has moved from modularity to scale plus normalisation (BRC, Newt,
   MR.Q-MT).

**Axes the five layers miss:**
- **Env allocation and task sampling:** one env per task (Newt), blocks of envs per task (MTBench),
  all tasks in parallel (BRC).
- **Reward and return scale normalisation** as its own concern: BRC's Ḡ_i, PPO's 0.1 scale, and
  categorical critics.
- **Capacity and compute scaling with the number of tasks:** batch, UTD and model size (TD-MPC2 Sec. 4.1,
  BRC Sec. 4, Newt Fig. 8).
- **Evaluation protocol:** episodes per task, the normalised-score definition for each domain, and
  planning vs the policy prior at test time.
- **Task-level stability control:** PaCo's mask-out and reset.
- **Prior-data provenance:** expert demonstrations vs replay buffers vs filtered episodes.

**What this means for the granularity.** The hypothesis holds with three amendments:
- The central abstraction is the task-set environment. It emits `task_id` and per-task metadata, not
  only padded observations.
- Each agent's published multi-task form (γ_i, α_i, return normalisation, embedding input) is a small,
  task-aware piece on the agent side, written as Extensions or network options.
- "Protocols" are the multi-stage regimes:
  1. single-task runs → pooled dataset → offline multi-task (TD-MPC2);
  2. demonstrations → pretrain → online (Newt);
  3. teacher → student (Policy Distillation, Actor-Mimic, TD-MPC-Opt);
  4. multi-task → held-out fine-tuning.
