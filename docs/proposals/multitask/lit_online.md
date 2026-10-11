# Online multi-task RL baselines: checking the five-layer hypothesis

Scope: the online side (Meta-World baselines, MTRL, IMPALA/PopArt, on-policy recipe, PCGrad/CAGrad/GradNorm,
two recent SOTA online methods). Written 2026-10-11.
How it was read: papers as arXiv PDFs through `pdftotext` (section/equation/table numbers below are the paper's);
IMPALA through the ar5iv HTML and a summariser only (marked *summ.*). Code from each repo's default branch on
2026-10-11 (no commit pinned), so line numbers can drift. Short names used below:
`[MTRL]` = https://github.com/facebookresearch/mtrl/blob/main/ ,
`[garage]` = https://github.com/rlworkgroup/garage/blob/master/src/garage/ ,
`[mtenv]` = https://github.com/facebookresearch/mtenv/blob/main/mtenv/envs/metaworld/ .
Papers: MW = Meta-World https://arxiv.org/abs/1910.10897 (v2); PCG = PCGrad https://arxiv.org/abs/2001.06782;
CAG = CAGrad https://arxiv.org/abs/2110.14048; GN = GradNorm https://arxiv.org/abs/1711.02257;
CARE = https://arxiv.org/abs/2102.06177; SM = Soft Modularization https://arxiv.org/abs/2003.13661;
POP = PopArt multi-task https://arxiv.org/abs/1809.04474; IMP = IMPALA https://arxiv.org/abs/1802.01561;
DIS = Distral https://arxiv.org/abs/1707.04175; BRC = https://arxiv.org/abs/2505.23150;
NEWT = https://arxiv.org/abs/2511.19584.

## 1. Verdict on the hypothesis

Holds:
- **The env is the main abstraction for on-policy baselines.** garage's MT-PPO and MT-TRPO are unchanged
  PPO/TRPO run on a `MultiEnvWrapper` that switches task at each reset and gets a one-hot from a per-task wrapper
  ([garage]examples/torch/mtppo_metaworld_mt10.py:44-53, 72-79; envs/multi_env_wrapper.py:182-190).
  CARE §4.1 describes the same split: either extend a single-task baseline with task-specific parts, which
  "are generally algorithm-agnostic", or use specialised multi-task algorithms such as PCGrad.
- Layers 3-5 vary on their own. MTRL exposes them as separate config switches: per-task alpha, task encoder,
  multi-head, MoE/soft-mod, and the PCGrad/GradNorm agent wrappers
  ([MTRL]config/agent/components/metaworld_multitask.yaml:3-32).

Wrong or incomplete:
1. **Distral is not distillation from per-task experts.** Per-task policies and a shared distilled policy train
   together, online (DIS §2, "joint training"). In MTRL, the per-task agents act and each updates every step
   ([MTRL]mtrl/agent/distral.py:133-167). Their TD target uses log π0 from the distilled policy (:373-383), and
   π0 is trained by KL to them (:435-496). Policy Distillation and Actor-Mimic were not opened here. So Distral
   belongs in the online regime, with a per-task architecture.
2. **"Unchanged agent" fails for SAC as the field runs it.** Every MT-SAC I read learns a separate α per task:
   - garage MTSAC: [garage]torch/algos/mtsac.py:142-145, reading the one-hot back out of the obs at :180-189;
   - PCG: App. C and App. J.2 ("separate alpha term per task");
   - CARE: Table 9, "learned and disentangled";
   - MTRL: mtrl/agent/sac.py:73-80, :154-155.
   That is an algorithm-specific per-task parameter, and it needs the task id in the batch. The MW paper only says
   "automatic entropy tuning: True" (MW Table 7). The per-task α is in the code, not in the paper.
3. **The task id is not always an input.** IMPALA multi-task: "the model does not know which task" (IMP §5.3.1,
   *summ.*). In PopArt, only the value heads and their statistics use the task index; the policy does not, so the
   agent can run "in a fully task agnostic way" (POP §"Scale invariant updates for multi-task learning"). MTRL's
   "Multi-task SAC" has no task input at all without a task encoder ([MTRL]mtrl/agent/components/actor.py:193-195).
   Its task-specific parts are the per-task α and, separately, the heads.
4. **Layers 3-5 are not all Extensions with today's Ajax hook set.** PCGrad, CAGrad, GradNorm and
   Soft-Mod's task weights all need *per-task losses* and they replace the loss *reduction*:
   - MTRL forces `loss_reduction="none"` ([MTRL]mtrl/agent/grad_manipulation.py:42);
   - it scatter-adds per-sample losses into per-task means (:103-109);
   - it runs one backward per task ([MTRL]mtrl/agent/pcgrad.py:134-141).
   Ajax's Extension phases are additive loss terms, target transforms, action overrides and `post_update`
   (src/ajax/extensions/base.py:29-40, clone b84469b). None of them sees per-sample losses or the
   gradient-combination step. PopArt needs a per-task head (architecture, layer 5), per-task statistics
   (layer 3) and a rescale of the output weights after each update. It cuts across both layers.
5. **Layer 3 often sits in the env, not the agent.** Per-task reward normalisation is applied inside each task's
   env wrapper:
   - MTRL/mtenv: [mtenv]env.py:131-132, which divides by a running std ([mtenv]wrappers/normalized_env.py:107-108);
   - garage MT-SAC: [garage]examples/torch/mtsac_metaworld_mt10.py:50-56 and envs/normalized_env.py:164;
   - BRC: divides each task's rewards by that task's best Monte-Carlo return so far (BRC §3, App. C).
6. **MT-MH-SAC is not in the Meta-World paper text I read.** MW §5 and Table 1 list only MT-PPO, MT-TRPO,
   MT-SAC and TE. SM §5.2 and CARE Table 1 attribute MT-MH-SAC to Meta-World. MW v1 was not checked, so this
   attribution is *unverified*.

## 2. Meta-World (paper and garage)
- Task conditioning: "a one-hot vector indicating the current task" (MW §4.3). Objects and goals are fixed per task
  in MT10/MT50 (§4.3). The obs is 39-dimensional and zero-padded for absent objects (§4.2, via ar5iv *summ.*).
- Baselines (MW §5): MT-PPO, MT-TRPO and MT-SAC, all "using the one-hot task ID as input", plus TE (on-policy
  task embeddings). Max success, MT10/MT50: MT-SAC 68.3/38.5, MT-PPO 30.5/35.4, MT-TRPO 31.3/21.0,
  TE 20.9/11.8 % (MW Table 1).
- garage, the official baselines code (hyperparameter tables are titled "Garage experiments", MW Tables 5-8):
  - one-hot appended by `TaskOnehotWrapper` ([garage]envs/task_onehot_wrapper.py:102-114);
  - task schedule `round_robin_strategy` / `uniform_random_strategy`, chosen at each reset
    ([garage]envs/multi_env_wrapper.py:15-43, 182-183);
  - MT-PPO: one PPO on that wrapper with round robin, GAE and advantage centring
    (mtppo_metaworld_mt10.py:51-79). Its envs are wrapped with `normalize(env)` and no reward normalisation (:46).
  - MT-SAC: one shared replay buffer (`PathBuffer`, :83), one sampler worker per task env (:84-93), so collection
    is balanced; a 1280-sample batch (:110-123); per-task reward normalisation (:50-51); per-task α as in §1.2.
  - TE-PPO: a learned embedding from the one-hot plus an inference network, batched per task and round robin
    (te_ppo_metaworld_mt10.py:25-62).

## 3. MTRL (facebookresearch/mtrl), shared vs per-task
- Collection: one env per task, all stepped in lockstep ([MTRL]mtrl/env/builder.py:56-75; vec_env.py:83 sets
  `task_obs = arange(num_envs)`). Transitions go to one shared buffer with the task index stored
  (experiment/multitask.py:235-249).
- Sampling: uniform over the shared buffer (replay_buffer.py:72-76). It is balanced only because collection is.
  `sample_an_index` (:93-104) gives per-task batches for Distral.
- Per-task: `log_alpha[num_envs]` when `should_use_disentangled_alpha` (sac.py:73-80, 154-155); per-task alpha
  loss (:415-427); per-task reward normalisation in the env (mtenv, §1.5).
- Multi-head: a head per task, picked by an identity mask (components/actor.py:170-175, 416-420).
- Task encoder: an embedding of the index, optionally a pretrained-language init (task_encoder.py;
  metaworld_multitask.yaml:8-21). Soft-Mod routing multiplies obs by the task encoding
  (soft_modularization.py:139-145). CARE is an attention-weighted mixture of encoders, keyed by language metadata
  (metaworld_encoder.yaml:25-72).
- The baselines' exact flags are listed in docs_src/source/pages/tutorials/baseline.rst:
  - MT-SAC: per-task α, no task encoder (so no task input; §1.3) (:76-95);
  - MH-SAC: per-task α plus heads (:97-116);
  - PCGrad: **shared α, no task input** (:118-136). PCG App. J.2 instead used a per-task α and the one-hot.
- Deviations seen in code (paper vs code, not run):
  - GradNorm keeps "L_i(0)" but overwrites it with the current loss every step (gradnorm.py:104-106). The paper
    defines L_i(0) as the initial loss (GN §3, Algorithm 1).
  - I found no SM Eq. 10 task reweighting in mtrl/agent/sac.py (grep for softmax/weight). MTRL Soft-Mod seems to
    keep only the routing, *unverified* beyond the grep.

## 4. IMPALA multi-task and PopArt
- IMPALA DMLab-30: "a fixed number of actors to each task" (IMP §5.3.1, *summ.*), so collection is balanced by
  allocation. One shared network, no task-specific heads, no task id. Optimistic asymmetric reward clipping
  (Fig. D.1). Multi-task 49.4 % vs IMPALA experts 44.5 % (Table 3, *summ.*).
- PopArt (POP):
  - moments per task by EMA (Eq. 6);
  - a vector of N task-indexed value outputs (Eq. 10);
  - **both** the value target and the policy-gradient advantage normalised by that task's (μ_i, σ_i) (Eqs. 11-12);
  - output weights rescaled to keep predictions unchanged (Eq. 13);
  - the task index is used at training time only (quote in §1.3).
- Result: Atari-57 median 59.7 % → 110.7 %; unclipped 0.3 % → 107.0 %; DMLab-30 test 58.4 → 72.8 % (POP Table 1).
  A MultiHead-IMPALA ablation has task-specific value heads but no PopArt. It separates the gain from extra
  capacity (POP Experiments, Fig. 1-2).
- What PopArt needs from the agent: the task id for each sample in the loss, a value head per task, and a
  post-update rescale of that head. It is an on-policy-friendly balancing mechanism, not SAC-specific.

## 5. Is there a standard on-policy (PPO) multi-task recipe?
- No dedicated published MT-PPO algorithm turned up. The de facto recipe is the single-task algorithm plus:
  1. task identity in the obs: one-hot (MW §5; garage), or none (IMP);
  2. balanced collection: round robin per episode (garage), or fixed actors per task (IMP);
  3. per-task value and advantage normalisation (PopArt).
- NEWT's PPO baseline (CleanRL) adds only language conditioning and per-task discounts (NEWT §4, App. D/H).
- PPO-specific note (my inference, not from a paper): PPO's batch-wide advantage normalisation
  (`center_adv=True`, mtppo_metaworld_mt10.py:78) mixes tasks. PopArt Eq. 12 is the published per-task version.

## 6. Gradient-interference methods: what they need, and the reported effect in RL
- **PCGrad** (PCG Alg. 1):
  - needs per-task gradients on the shared parameters, one backward per task, and a random projection order (§3);
  - is applied to both actor and critic gradients (App. C);
  - RL setup: per-task replay buffers, equal paths per task, stratified batches of 128 per task, 600 samples
    per task per collection, per-task α, one-hot input, 6-layer MLP (App. C, J.2);
  - cost: about 2x wall-clock and memory vs SAC on MT10 and MT50 (App. J.2);
  - reported effect: "solves all of the 10 tasks in MT10 and about 70%" of MT50 vs under half for SAC/MH-SAC (§5.2).
- **CAGrad** (CAG §3, Eq. 3):
  - needs the per-task gradients g_i plus the average gradient g0, and solves a small dual problem each step;
  - CAGrad-Fast subsamples |S| tasks for O(|S|N+T) cost (§3.3, Eq. 4);
  - run on MTRL, with max-over-training reporting (App. B.3).
  - MT10/MT50 success (CAG Table 3), with CARE's identical baseline numbers (CARE Table 1):

    | Method       | MT10 | MT50 |
    |--------------|-----:|-----:|
    | MT-SAC       | 0.49 | 0.36 |
    | +TaskEnc     | 0.54 | 0.40 |
    | MH-SAC       | 0.61 | 0.45 |
    | PCGrad       | 0.72 | 0.50 |
    | Soft-Mod     | 0.73 | 0.50 |
    | CAGrad       | 0.83 | 0.52 |
    | CARE         | 0.84 | 0.54 |
    | One SAC/task | 0.90 | 0.74 |

- **GradNorm** (GN §3, Eqs. 1-2, Alg. 1):
  - needs per-task losses L_i(t), their initial values L_i(0), and the gradient norms of w_i·L_i on the last
    shared layer;
  - learns weights w_i renormalised to sum to T;
  - the paper reports no RL experiments (supervised only); MTRL has an RL port (§3).
- Common interface: all three, and SM's α-softmax weights (SM §4.2, Eq. 10, applied to actor **and** critic),
  need the per-sample loss with its task id before reduction. That is one hook, shared by every agent.
- BRC counterpoint (BRC §3, Fig. 5): gradient conflicts drop as the critic grows. Cross-entropy (categorical)
  critics plus per-task return normalisation and learned task embeddings beat separate heads. On this account,
  layer 4 matters less at scale.

## 7. Recent SOTA online methods (state, continuous control)
- **BRC** (SAC or DrQ-ε base):
  - learned task embeddings, trained by the TD loss, instead of heads;
  - a categorical cross-entropy critic;
  - per-task reward scaling by max MC return;
  - a big residual critic (§3, App. C);
  - budgets per task, e.g. 500k steps per task on MW (§4).
- **NEWT**, a TD-MPC2 extension, *online* multi-task:
  - per-task discount factors (§3.1);
  - padded observations, with "a mask" for invalid action dimensions (§2.2);
  - CLIP language-instruction conditioning (§3.1);
  - pre-training on demos, then online, with a 50/50 demo/online buffer mix (§3.2 (3));
  - baselines (§4): **TD-MPC2 trained with multitask online RL**, plus PPO and FastTD3, both extended with
    language and per-task discounts.
  - So an online multi-task TD-MPC2 is a published baseline. It is separate from the offline TDMPC2MultiTask
    that Yann keeps "No deviation from the paper".

## 8. Method × layer table (L1 interface, L2 data regime, L3 balancing, L4 interference, L5 architecture)

| Method (source) | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| MT-PPO / MT-TRPO (MW §5; garage) | one-hot in obs | online, round robin per episode | none (obs norm only) | none | shared MLP |
| TE-PPO (MW §5; garage) | one-hot → learned latent + inference net | online, round robin | none | none | embedding-conditioned policy |
| MT-SAC (MW; garage) | one-hot in obs | online, one worker per task, shared buffer | per-task α (code), per-task reward norm | none | shared MLP 400x400 |
| MT-SAC (MTRL) | **no task input** | online lockstep, shared uniform buffer | per-task α, per-task reward norm | none | shared |
| MT-MH-SAC (MTRL; SM; CARE) | task picks the head | as MTRL | per-task α | none | multi-head |
| MT-SAC + task encoder (MTRL) | learned embedding (opt. language init) | as MTRL | per-task α | none | shared + embedding concat |
| Soft Modularization (SM) | one-hot → embedding into the router | online | per-task α + softmax(−α_i) loss weights (Eq. 10) | (weights only) | routed modules |
| CARE (CARE §3) | language metadata → context | as MTRL | per-task α | none | attention mixture of encoders |
| PCGrad-SAC (PCG App. C, J.2) | one-hot | online, per-task buffers, stratified | per-task α | PCGrad on actor+critic | shared 6-layer MLP |
| CAGrad-SAC (CAG §5.3) | as MTRL setup | as MTRL | as MTRL | CAGrad / -Fast | shared |
| GradNorm (GN; MTRL port) | any | any | learned loss weights | grad-norm balancing | needs a "last shared layer" |
| Distral (DIS; MTRL) | per-task policy, task id picks it | online, **joint** per-task + distilled | entropy/KL coefficients | none (separate nets) | per-task nets + shared π0 |
| IMPALA multi-task (IMP §5.3) | **none** | online, fixed actors per task | asymmetric reward clipping | none | shared |
| PopArt-IMPALA (POP) | none for the policy; id for value heads only | as IMPALA | per-task value+advantage norm | none | multi-head value |
| BRC (BRC §3) | learned task embedding | online, budget per task | per-task max-return scaling; CE critic | none (argues unneeded at scale) | big residual critic |
| NEWT (NEWT §2-3) | language + padding + action mask | demos pre-train → online, 50/50 buffers | per-task discount | none | TD-MPC2 world model |

## 9. Axes the five layers miss
1. **What the deployed policy sees** (train-time vs test-time task id). PopArt uses the id at training time only;
   IMPALA never; MW at test too. This is a deployment-interface choice, separate from how L1 encodes the id.
2. **Assigning tasks to env slots**: round robin or uniform per episode (garage), fixed per actor (IMP), lockstep
   one env per task (MTRL). It decides balance before any replay trick, and in one compiled program it decides the
   batch layout (my inference).
3. **Where normalisation lives**: per-task reward normalisation sits in the env wrapper (mtenv, garage, BRC App. C)
   and can belong to the multi-task env. PopArt-style normalisation sits in the loss.
4. **Hybrid data regimes**: demos then online (NEWT §3.2). Distral is joint online, not two-stage. The L2 list
   needs "pre-train then online" and "joint distillation".
5. **Budget accounting**: steps per task (CARE "2 million steps (for each environment)", Table 2-3 captions; BRC
   §4) vs total steps (PCG App. J.2: 5M on MT10). Sample allocation across tasks is also a choice.
6. **Evaluation protocol**: max over training vs final policy. MW Table 1 reports the max; CARE §4.1 reports the
   max and shows eval frequency acts as a hyperparameter (App. Table 19); CAG App. B.3 also the max; SM §5.2
   evaluates the final policy over 100 episodes per task. Cross-paper numbers are not comparable. PCG's 100 %
   on MT10 (own setup) vs 0.72 for "PCGrad" in CAG/CARE (MTRL's variant, shared α, no task input).
7. **Per-task discount / horizon** (NEWT §3.1; Ajax `TaskSet`, src/ajax/agents/TDMPC2/multitask.py:9-28,
   100-125). This is a task-set property, not balancing.

## 10. What this suggests for Ajax's granularity (inference, to be checked against the code)
- The agent-agnostic multi-task env from the hypothesis is supported. It should own:
  task-slot assignment, padding/masks, per-task discount, per-task reward normalisation, and the task id carried
  alongside the obs (not only inside it), with the policy's view of it configurable.
  The existing TD-MPC2 `TaskSet` already holds padding, masks and discounts (multitask.py:60-125).
- Two agent-side contracts are needed beyond today's Extension phases:
  - (a) the batch carries `task_id` to the losses;
  - (b) a pluggable **per-task loss reduction**: default mean, plus PCGrad/CAGrad/GradNorm/SM weights.
  (b) is one new phase, not one Extension per method. In JAX, per-task gradients are a `jax.jacrev` over a
  K-vector of task-mean losses (my inference; PCG App. J.2 reports about 2x cost at K=10).
- Per-task α is SAC-family specific: SAC, ASAC, REDQ (and TD3 has no α). It needs the agent's temperature to
  index by `task_id`. That is an agent option, or an Extension only if a "temperature" phase exists.
- PopArt is the on-policy counterpart (PPO, APO): per-task value statistics, a per-task value head, and a
  post-update rescale.
- A "protocol" object is still warranted only for multi-stage regimes (offline-from-single-task, two-stage
  distillation, demos → online). Distral is joint online and does not need one.

## Unverified / not opened
- IMPALA facts came via the ar5iv summariser only (§5.3.1, Tables 3-4, Fig. D.1); the PDF text was not grepped.
- MW v1 (2019) baselines list, i.e. whether MT-MH-SAC was there.
- How MTRL's per-sample task id reaches the critic in multi-head mode (critic.py not read).
- Policy Distillation, Actor-Mimic, PaCo, MOORE: not opened here.
- garage `PathBuffer` sampling details (assumed uniform; not read).
- Whether MTRL's documented MT-SAC command runs as written: not run.
