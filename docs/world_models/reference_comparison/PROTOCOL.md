# Ajax DreamerV3 vs reference DreamerV3 (29eb964) on CartPole-v1: comparison protocol

> On 2026-10-04 the absolute local paths in this file were replaced by the
> placeholders `<scratch>` and `<repo>` (README.md, "Recorded locations");
> nothing else was changed.

Single source of truth for the builder and the auditors. Question answered: on
the exact setup of the failed learning check (`dreamerv3-cartpole`: 1m model,
16 envs, train ratio 512, batch 16 x 64, 20 000 rows, eval every 2 000 rows
with 10 episodes), does the **real reference code** learn CartPole-v1 better
than Ajax does (an end-to-end bug in Ajax), or is Ajax's curve (final eval
155; 22, 100, 378, 123, 182, 219, 186, 126, 121, 155) within what the
reference itself produces? The invented bar (> 400) plays no role here.

Path abbreviations used below:

| name | path |
|---|---|
| `SP` | `<scratch>` |
| `AJ` | `SP/wt-m7` (Ajax worktree, **read only**; Ajax sources are `AJ/src/ajax/...`) |
| `REF` | `SP/refs/dreamerv3_29eb964` (reference checkout, **read only**) |
| `RR` | `SP/refrun` (every new file goes here) |
| `GX` | `<repo>/.venv/lib/python3.12/site-packages/gymnax` (gymnax 1.0.0, git `YannBerthelot/gymnax@61ff068`, the version Ajax runs) |
| `RPY` | `SP/venvs/dreamerv3_ref/bin/python` (jax 0.4.26 CPU, numpy 1.26, optax, tfp 0.24, ruamel, psutil, zmq, cloudpickle; no tensorflow, no gym) |
| `APY` | `<repo>/.venv/bin/python` |

All `file:line` citations are to these trees as they are now. Ajax file
paths are relative to `AJ/src/ajax/`; reference paths relative to `REF/`.

Verified facts this protocol rests on (probes in `RR/inspect_ajax.py`,
`RR/inspect_ajax_metrics.py`, `RR/inspect_ajax_params.py`,
`RR/inspect_ref_config.py`, `RR/probe_ref_agent.py`):

* Ajax resolved config (instantiating `DreamerV3(env_id="CartPole-v1",
  model_size="1m")`): units = hidden = 64, deter 512, stoch 32, classes 4,
  blocks 8, train_ratio 512, batch 16 x 64, ring 1 250 rows/env for 20 000
  rows, gate step `t0 = 1040`, **9 481 updates** in 1 250 ticks, env = bare
  gymnax `CartPole` (no wrapper), `max_steps_in_episode = 500`, 2 discrete
  actions.
* The reference Agent built with the overrides of section 1 on the port's
  spaces has **637 511** optimised variables; Ajax's learner has **637 381** =
  637 511 - 130, exactly the two extra two-hot output columns (64 + 1 each) of
  the reference's reward head and critic (`dreamerv3/nets.py:437-443`;
  `docs/world_models/deviations.md` section 1 "Two-hot output width"). The
  dimensions therefore match.
* Reference train metric names (one `Agent.train` call) were listed by the
  probe; the name map of section 5.4 is built from that list and from
  Ajax's `train_metric_keys` (90 keys).
* Reference cost on this machine under the current load (Ajax test suite
  running): build + compile 195 s; one update 1.0-1.4 s; one 16-env policy
  call 2-5 ms. The builder re-measures (section 7).

---

## 1. Config mapping table

Ajax values are the resolved values of `DreamerV3(env_id="CartPole-v1",
model_size="1m")` with every other argument at its default
(`agents/DreamerV3/DreamerV3.py:148-203`, `DreamerV3Config.from_model_size`
`agents/DreamerV3/state.py:203-232`, `MODEL_SIZES` `state.py:46-54`,
`DreamerV3AgentConfig` `state.py:252-275`), as printed by
`RR/inspect_ajax.py`. Reference keys are `dreamerv3/configs.yaml` of 29eb964
(`defaults:` block lines 1-146) and how `main.py`, `agent.py`, `nets.py`,
`jaxutils.py`, `embodied/run/train.py` read them. 29eb964 has **no
`size1m` preset** (presets `size12m` ... `size400m`, `configs.yaml:148-176`);
rows 3-6 give the explicit overrides that reproduce Ajax's `1m`
(`d = 64`: units = hidden = d, deter = 8d, classes = d/16, `state.py:225-231`),
verified by `RR/inspect_ref_config.py` (the regex key `.*\.units` is applied
by `embodied.Config.update` exactly as the stock presets' `.*\.units`).

Status: **EQUAL** = the reference default already equals Ajax's value;
**OVERRIDE** = set in this run (value in the "this run" column);
**UNMATCHABLE** = cannot be made identical without changing reference code
(listed again in section 6 with impact).

| # | Ajax setting = resolved value | Ajax source | Reference key | 29eb964 default (source) | This run | Status |
|---|---|---|---|---|---|---|
| 1 | `n_envs` = 16 | `DreamerV3.py:151` | `run.num_envs` | 16 (`configs.yaml:44`) | 16 | EQUAL |
| 2 | `model_size` = `'1m'` (d = 64) | `DreamerV3.py:152`; `state.py:46-54, 225-231` | none (no `size1m` in 29eb964) | `defaults` = size200m dims (`configs.yaml:113-129`) | rows 3-6 | OVERRIDE |
| 3 | `units` = 64 (enc, dec, rew, con, actor, critic MLP width) | `state.py:227` | `.*\.units` (= `enc.simple.units`, `dec.simple.units`, `rewhead.units`, `conhead.units`, `actor.units`, `critic.units`) | 1024 (`configs.yaml:117, 121, 122, 123, 128, 129`) | 64 | OVERRIDE |
| 4 | `hidden` = 64 (RSSM input/hidden/prior/posterior width) | `state.py:228` | `dyn.rssm.hidden` | 1024 (`configs.yaml:113`) | 64 | OVERRIDE |
| 5 | `deter` = 512 | `state.py:229` | `dyn.rssm.deter` | 8192 (`configs.yaml:113`) | 512 | OVERRIDE |
| 6 | `classes` = 4 | `state.py:230` | `dyn.rssm.classes` | 64 (`configs.yaml:113`) | 4 | OVERRIDE |
| 7 | `stoch` = 32 | `DreamerV3.py:157` | `dyn.rssm.stoch` | 32 (`configs.yaml:113`) | 32 | EQUAL |
| 8 | `blocks` = 8, block-GRU core | `DreamerV3.py:158`; `networks.py:145-180, 211-250` | `dyn.rssm.cell`, `.blocks`, `.block_fans`, `.block_norm` | `blockgru`, 8, False, False (`configs.yaml:113`) | same | EQUAL |
| 9 | RSSM depths: prior 2 layers, posterior 1, one core hidden layer; posterior input `concat(deter, token)` | `networks.py:84-85, 300-320`; `networks.py:204, 242` | `dyn.rssm.imglayers`, `.obslayers`, `.dynlayers`, `.absolute` | 2, 1, 1, False (`configs.yaml:113`; used `nets.py:66-67, 115, 129`) | same | EQUAL |
| 10 | `enc_layers` = 3, symlog input | `DreamerV3.py:159`; `networks.py:387-395` | `enc.typ`, `enc.simple.layers`, `enc.simple.symlog` | simple, 3, True (`configs.yaml:116-117`; `nets.py:254-260`) | same | EQUAL |
| 11 | `dec_layers` = 3, symlog-MSE, tolerance 1e-8 | `DreamerV3.py:160`; `distributions.py:50-51, 144-150` | `dec.simple.layers`, `dec.simple.vecdist` | 3, `symlog_mse` (`configs.yaml:121`; `nets.py:315-326, 450-453`; `jaxutils.py:179-207` tol 1e-8) | same | EQUAL |
| 12 | decoder output outscale 0.1 | `networks.py:77-80, 362` | (vector `Dist` built without `outscale`, `nets.py:326`) | `Dist.outscale` 0.1 (`nets.py:413`); `dec.simple.outscale: 1.0` applies to image heads only (`nets.py:358`) | same | EQUAL |
| 13 | `rew_layers` = 1, two-hot (symexp bins), output outscale 0 | `DreamerV3.py:161`; `networks.py:365` | `rewhead.layers`, `.dist`, `.outscale` | 1, `symexp_twohot`, 0.0 (`configs.yaml:122`; `nets.py:466-476`) | same | EQUAL |
| 14 | `con_layers` = 1, Bernoulli, outscale 1 | `DreamerV3.py:162`; `networks.py:366` | `conhead.layers`, `.dist`, `.outscale` | 1, `binary`, 1.0 (`configs.yaml:123`) | same | EQUAL |
| 15 | `actor_layers` = 3, discrete one-hot actor with `actor_unimix` = 0.01, output outscale 0.01 | `DreamerV3.py:163, 168`; `networks.py:90, 410-440`; `distributions.py:233-253` | `actor.layers`, `actor_dist_disc`, `actor.unimix`, `actor.outscale` | 3, `onehot`, 0.01, 0.01 (`configs.yaml:128, 130`; `nets.py:523-530`) | same | EQUAL |
| 16 | `critic_layers` = 3, two-hot, output outscale 0 | `DreamerV3.py:164`; `networks.py:91, 444-454` | `critic.layers`, `.dist`, `.outscale` | 3, `symexp_twohot`, 0.0 (`configs.yaml:129`) | same | EQUAL |
| 17 | `bins` = 255 (reward head, critic) | `DreamerV3.py:165` | `rewhead.bins`, `critic.bins` | 255 (`configs.yaml:122, 129`) | 255 | EQUAL (output width 256 vs 255, row 61) |
| 18 | `minstd` 0.1 / `maxstd` 1.0 (continuous actor only; inactive here) | `DreamerV3.py:189-190` | `actor.minstd`, `actor.maxstd` | 0.1, 1.0 (`configs.yaml:128`) | same | EQUAL |
| 19 | hidden layer = Dense (bias) -> RMSNorm(eps 1e-4) -> SiLU | `networks.py:23-24, 204, 242` | `*.norm`, `*.act` (rssm, enc, dec, heads, actor, critic) | `rms`, `silu` (`configs.yaml:113-129`); `Norm(eps=1e-4)` (`nets.py:733`) | same | EQUAL |
| 20 | latent `unimix` = 0.01 | `DreamerV3.py:167` | `dyn.rssm.unimix` | 0.01 (`configs.yaml:113`; `nets.py:212-215`) | 0.01 | EQUAL |
| 21 | kernel init `variance_scaling(outscale^2, fan_in, truncated_normal)` | `networks.py:73-76` (deviation D23) | `*.winit` | `normal`: `1.1368 sqrt(1/fan_in) TruncNormal(-2,2)` (`nets.py:821-860`, `:848`) | `normal` | UNMATCHABLE (constant factor, 4.16e-5 relative) |
| 22 | `free_nats` = 1.0 | `DreamerV3.py:166`; `world_model.py:131-155` | `rssm_loss.free` | 1.0 (`configs.yaml:125`; `nets.py:97-105`) | 1.0 | EQUAL |
| 23 | `rec_scale`, `rew_scale`, `con_scale`, `dyn_scale`, `rep_scale` = 1, 1, 1, 1, 0.1 | `DreamerV3.py:169-173` | `loss_scales.dec_mlp`, `.reward`, `.cont`, `.dyn`, `.rep` | 1, 1, 1, 1, 0.1 (`configs.yaml:98`; `agent.py:92-97`) | same | EQUAL |
| 24 | `actor_scale`, `critic_scale`, `repval_scale` = 1, 1, 0.3 | `DreamerV3.py:174-176` | `loss_scales.actor`, `.critic`, `.replay_critic` | 1, 1, 0.3 (`configs.yaml:98`) | same | EQUAL |
| 25 | continue target `gamma (1 - is_terminal)` | `world_model.py:230` | `contdisc` | True (`configs.yaml:124`; `agent.py:242-245`) | True | EQUAL |
| 26 | `return_horizon` = 333 (gamma = 1 - 1/333) | `DreamerV3.py:181`; `state.py:239-249` | `horizon` | 333 (`configs.yaml:136`) | 333 | EQUAL |
| 27 | `imag_horizon` = 15, imagination from every posterior state, 1 repeat | `DreamerV3.py:180`; `world_model.py:118`; `learner.py:282` | `imag_length`, `imag_start`, `imag_repeat`, `imag_unroll` | 15, `all`, 1, False (`configs.yaml:132-135`) | same | EQUAL |
| 28 | `lam` = 0.95 | `DreamerV3.py:182` | `return_lambda` | 0.95 (`configs.yaml:137`) | 0.95 | EQUAL |
| 29 | `repval_lam` = 0.95; replay critic on, with gradient, bootstrapped from imagination | `DreamerV3.py:183`; `actor_critic.py:21, 27-31, 270-280` | `return_lambda_replay`, `replay_critic_loss`, `replay_critic_grad`, `replay_critic_bootstrap` | 0.95, True, True, `imag` (`configs.yaml:104-106, 138`) | same | EQUAL |
| 30 | `actent` = 3e-4 | `DreamerV3.py:184` | `actent` | 3e-4 (`configs.yaml:144`) | 3e-4 | EQUAL |
| 31 | `slowreg` = 1.0; targets from the online critic | `DreamerV3.py:185`; `actor_critic.py:211` | `slowreg`, `slowtar` | 1.0, False (`configs.yaml:145-146`) | same | EQUAL |
| 32 | `slow_rate` = 0.02, every update, hard copy at the first | `DreamerV3.py:186`; `learner.py:519-565` | `slow_critic_fraction`, `slow_critic_update` | 0.02, 1 (`configs.yaml:139-140`; `jaxutils.py:737-762`) | same | EQUAL |
| 33 | return normaliser: EMA of P5/P95, `retnorm_rate` 0.01, `retnorm_limit` 1.0 | `DreamerV3.py:187-188`; `normalizers.py:51-113`; `learner.py:209` | `retnorm` | `{impl: perc, rate: 0.01, limit: 1.0, perclo: 5, perchi: 95}` (`configs.yaml:141`; `jaxutils.py:301-396`) | same | EQUAL |
| 34 | no value / advantage normalisation; actor-critic inputs stop-gradiented; reward head trains the RSSM; no context reset | `actor_critic.py:27-31`; `learner.py:289`; `world_model.py:194` | `valnorm.impl`, `advnorm.impl`, `ac_grads`, `reward_grad`, `reset_context` | off, off, `none`, True, 0.0 (`configs.yaml:102-103, 107, 142-143`) | same | EQUAL |
| 35 | `learning_rate` = 4e-5, **one** optimizer over world model + actor + critic | `DreamerV3.py:191`; `learner.py:190` | `opt.lr`, `separate_lrs` | 4e-5, False (`configs.yaml:99-100`; `agent.py:83-91`) | same | EQUAL |
| 36 | LaProp: `beta1` 0.9, `beta2` 0.999, `eps` 1e-20 | `DreamerV3.py:194-196`; `optim.py:121-139` | `opt.scaler`, `opt.momentum`, `opt.beta1`, `opt.beta2`, `opt.eps` | `rms`, True, 0.9, 0.999, 1e-20 (`configs.yaml:99`; `jaxutils.py:436-444, 677-713`) | same | EQUAL |
| 37 | AGC `agc` 0.3, `agc_pmin` 1e-3; no global clip | `DreamerV3.py:192-193` | `opt.agc`, `opt.pmin`, `opt.globclip` | 0.3, 1e-3, 0.0 (`configs.yaml:99`; `jaxutils.py:431-434, 660-674`) | same | EQUAL |
| 38 | `warmup` 1000 updates, linear from 0, constant afterwards, no weight decay | `DreamerV3.py:197`; `optim.py:98-110` | `opt.warmup`, `opt.schedule`, `opt.anneal`, `opt.wd` | 1000, `constant`, 0, 0.0 (`configs.yaml:99`; `jaxutils.py:508-524`) | same | EQUAL |
| 39 | `train_ratio` = 512 | `DreamerV3.py:177`; `train_DreamerV3.py:140-182` | `run.train_ratio` | 32.0 (`configs.yaml:52`) | **512.0** | OVERRIDE |
| 40 | `batch_size` = 16 | `DreamerV3.py:178` | `batch_size` | 16 (`configs.yaml:91`) | 16 | EQUAL |
| 41 | `batch_length` = 64 trained rows + 1 context row | `DreamerV3.py:179`; `state.py:262-265` | `batch_length`, `replay_context` | 65, 1 (`configs.yaml:92, 96`); `batch_steps = 16 (65 - 1)` (`embodied/run/train.py:26`) | same | EQUAL |
| 42 | training-start gate at step `t0 = 1040`, then `1 + floor((t - t0)/2)` updates (9 481 by 20 000) | `train_DreamerV3.py:147-182` (deviation D26) | `run.train_fill`; gate `len(replay) >= batch_size` | 0 (`configs.yaml:53`); `train.py:81-84`; `when.Ratio` (`embodied/core/when.py:26-42`) | 0 | EQUAL (r = 1/2: D26 exact) |
| 43 | `replay_capacity` 5e6 rows -> ring of 1 250 rows/env for 20 000 rows (never overwrites) | `DreamerV3.py:198, 288-298` (D27) | `replay.size` | 5e6 items (`configs.yaml:12`; `main.py:164, 187`) | 5e6 | EQUAL in effect (neither binds) |
| 44 | online queue first, then uniform with replacement | `agents/DreamerV3/replay.py:29-36` | `replay.online`, `replay.fracs` | True, `{uniform: 1.0, ...}` (`configs.yaml:13-14`; `embodied/replay/replay.py:140-144, 178-191`) | same | EQUAL |
| 45 | replay sampler seeded from the run seed | `agents/DreamerV3/replay.py` (D2) | none (code) | `selectors.Uniform(seed=0)`, never derived from `config.seed` (`embodied/replay/selectors.py:29-32`; `embodied/replay/replay.py:20, 27`; `main.py:187`) | seed 0 for every reference seed | UNMATCHABLE |
| 46 | float32 compute and params | deviation D4 | `jax.compute_dtype`, `jax.param_dtype` | `bfloat16`, `float32` (`configs.yaml:24-25`; `jaxagent.py:292-293`) | **`float32`**, `float32` | OVERRIDE |
| 47 | CPU | `JAX_PLATFORMS=cpu` | `jax.platform` | `gpu` (`configs.yaml:22`; `jaxagent.py:286`) | **`cpu`** | OVERRIDE |
| 48 | synchronous: acting with the latest parameters, metrics of the current update, no prefetch | `train_DreamerV3.py:546-572` (D1) | `jax.sync_every` + `JAXAgent` pending sync / pending outs and metrics; `Prefetch` threads | 1 (`configs.yaml:32`); `jaxagent.py:147-153, 188-202`; `embodied/core/prefetch.py:9-41` | stock | UNMATCHABLE |
| 49 | all updates of a vector step run after its 16 rows are added | `train_DreamerV3.py:559-572` (D3) | none (code) | `train_step` after each transition (`train.py:69-72, 81-92`) | stock | UNMATCHABLE |
| 50 | discrete action = int32 index; one-hot inside the model; raw action zeroed on `is_last` rows | `train_DreamerV3.py:248, 262`; `row_collector.py:329` | act space | port `Space(np.int32, (), 0, 2)`; `jaxutils.onehot_dict` (`jaxutils.py:723-734`, `agent.py:135, 233`); driver mask (`embodied/core/driver.py:70-74`) | same | EQUAL |
| 51 | the policy samples (actor and posterior) in every mode | `train_DreamerV3.py:243-264` | `mode` argument | ignored by `Agent.policy` (`agent.py:129-164`) | ignored | EQUAL |
| 52 | `action_repeat` = 1 | `DreamerV3.py:200` | env repeat | port steps once per action | 1 | EQUAL |
| 53 | episode limit 500 agent steps from gymnax params (`episode_length` = 1000 is ignored for gymnax, `DreamerV3.py:107-109`) | `DreamerV3.py:199`; `environments/utils.py:175-215` | `wrapper.length`, port | 0 = no `TimeLimit` (`configs.yaml:77`; `main.py:238-239`); the port truncates at 500 | 0 + port | EQUAL |
| 54 | one flat vector observation (4) into encoder and decoder | `train_DreamerV3.py:297-303` | `enc.spaces`, `dec.spaces` | `'.*'` (`configs.yaml:115, 119`) selects the port's only non-flag key `vector` (`agent.py:36-43`) | `'.*'` | EQUAL (single key: D5 moot) |
| 55 | `n_timesteps` = 20 000 rows | `benchmarks/learning_checks.py:128` (`AJ/benchmarks/...`) | `run.steps` | 1e10 (`configs.yaml:42`; loop `train.py:107`) | **20000** | OVERRIDE |
| 56 | log + eval every 2 000 rows, 10 episodes | `benchmarks/learning_checks.py:113-114, 129` | `run.log_every` (seconds) | 120 (`configs.yaml:47`; `when.Clock` `when.py:69-89`) | **-1** (stock wall-clock log block off; section 3.4 instrumentation logs at multiples of 2 000) | OVERRIDE |
| 57 | no report stream | D2 | `run.eval_every` (seconds) | 180 (`configs.yaml:49`; `train.py:111-113`) | **-1** | OVERRIDE |
| 58 | no periodic checkpoints | n/a | `run.save_every` (seconds) | 900 (`configs.yaml:48`; `train.py:125-126`) | **-1** (the stock initial save at `train.py:100` stays; harmless) | OVERRIDE |
| 59 | envs stepped in-process | `row_collector.py:510-551` | `run.driver_parallel` | True (`configs.yaml:73`; `driver.py:16-28`) | **False** | OVERRIDE (same observations) |
| 60 | seed `s` (0, 1, 2) | `DreamerV3.train(seed=...)` (`DreamerV3.py:391-423`) | `seed` | 0 (`configs.yaml:3`; param init `[seed, 0]` `jaxagent.py:382`; `self.rng = default_rng(seed)` `jaxagent.py:39`) | **s** | OVERRIDE |
| 61 | two-hot: 255 outputs; bins = float32 rounding of float64 values; expectation exactly 0 for uniform p | `distributional.py:11-45`; `networks.py:44-52` (D22) | `rewhead.bins`, `critic.bins` | 256 outputs, last dropped (`nets.py:437-443`); bins `symexp(linspace(-20, 0, 128))` in float32 under jit (`nets.py:466-476`); a zero-init head predicts 0.07-0.16 | stock | UNMATCHABLE (layout: none; values: numerics) |
| 62 | (no counterpart) | n/a | `jax.prealloc`, `jax.transfer_guard` | True, True (`configs.yaml:26, 34`; `jaxagent.py:264-289`) | **False, False** | OVERRIDE (no computational effect; lets the evaluation of section 4 pass host arrays) |
| 63 | RNG: one JAX key per seed, split per tick / update / eval | `train_DreamerV3.py:368, 425`; `row_collector.py:317` | numpy `Generator(seed)` drawing per-call seeds, shared by the policy and two prefetch threads | `jaxagent.py:39, 236-237, 391-394` | stock | UNMATCHABLE (independent streams) |
| 64 | (exploration flag unused) | n/a | `run.expl_until` | 0: `when.Until(0)` is always true, so the policy runs with `mode='explore'` (`train.py:27, 104-105`; `when.py:57-66`), ignored (`agent.py:129`) | 0 | EQUAL |
| 65 | env checks | n/a | `wrapper.checks`, `wrapper.discretize` | True, 0 (`configs.yaml:77`; `main.py:228-245`: discrete action -> no `NormalizeAction`/`ClipAction`; `ExpandScalars`; `CheckSpaces`) | same | EQUAL |
| 66 | task label | n/a | `task` | `dummy_disc` (`configs.yaml:5`) | **`gymnax_cartpole`** (label only: `make_env` is replaced, section 3.2) | OVERRIDE |
| 67 | (output location) | n/a | `logdir` | `/dev/null` (`configs.yaml:6`) | **`RR/full/ref_s{s}`** (must not exist beforehand, section 3.5) | OVERRIDE |

Complete override set for the reference run (a Python dict applied with
`embodied.Config(agt.Agent.configs['defaults']).update(OVERRIDES)`, then the
stock `main.py:35-38` update):

```python
OVERRIDES = {
    "seed": s, "task": "gymnax_cartpole", "logdir": f"{RR}/full/ref_s{s}",
    "jax.platform": "cpu", "jax.compute_dtype": "float32", "jax.param_dtype": "float32",
    "jax.prealloc": False, "jax.transfer_guard": False,
    "dyn.rssm.deter": 512, "dyn.rssm.hidden": 64, "dyn.rssm.classes": 4,
    r".*\.units": 64,
    "run.train_ratio": 512.0, "run.steps": 20000, "run.num_envs": 16,
    "run.log_every": -1, "run.eval_every": -1, "run.save_every": -1,
    "run.driver_parallel": False,
}
```

Nothing else is overridden; every other key keeps its 29eb964 default. The
builder dumps the resolved config to `RR/full/ref_s{s}/config.yaml` (stock
`main.py:54`) and `RR/compare.py` asserts the values of the "This run"
column against it.

---

## 2. Environment

### 2.1 What Ajax's learner sees (gymnax CartPole-v1 through the row collector)

* **Env stack.** `DreamerV3(env_id="CartPole-v1")` builds the bare gymnax
  `CartPole` (wrapper chain `['CartPole']`, `RR/inspect_ajax.py`), default
  `EnvParams` (`GX/environments/classic_control/cartpole.py:21-33`: gravity
  9.8, masscart 1.0, masspole 0.1, total_mass 1.1, length 0.5,
  polemass_length 0.05, force_mag 10.0, tau 0.02, theta threshold
  `12 * 2 * pi / 360` = 0.20943951 rad, x threshold 2.4,
  `max_steps_in_episode` 500).
* **Dynamics** (`cartpole.py:52-99`): `force = 10 a - 10 (1 - a)` for action
  index `a` in {0, 1}; explicit Euler with the pre-step velocities
  (`x += tau x_dot`, `x_dot += tau xacc`, `theta += tau theta_dot`,
  `theta_dot += tau thetaacc`); `reward = 1 - is_terminated(previous state)`
  (`:80-81`), which is 1.0 on every step of a live episode;
  `time += 1`. Observation `[x, x_dot, theta, theta_dot]` float32
  (`:115-117`).
* **Termination vs truncation** (gymnax 1.0 six-value step,
  `GX/environments/environment.py:63-100`): `terminated =
  is_terminated(post-step state)` = `x < -2.4 or x > 2.4 or theta < -thr or
  theta > thr` (`cartpole.py:119-131`, strict inequalities); `truncated =
  post-step time >= 500` (`environment.py:150-152`). Ajax takes the pair at
  face value (`environments/interaction.py:238-241`) and the row collector sets
  `is_last = terminated or truncated`, **`is_terminal = is_last and
  terminated`** (`row_collector.py:529, 545-546`): a time-limit end at step
  500 has `is_terminal = False` unless the pole also falls on that step.
  An episode has at most 500 steps, return = number of steps.
* **Reset** (`cartpole.py:101-113`): `uniform(-0.05, 0.05)` on all four state
  variables, time 0. Ajax's dynamic mode uses gymnax's auto-reset observation
  drawn at the ending step (`environment.py:82-92`; stashed
  `row_collector.py:548`), emitted one tick later.
* **Rows** (obs-aligned, `row_collector.py:9-27, 270-399`): for an episode of
  `L` steps the env emits `L + 1` rows: row 0 = reset observation,
  `is_first = 1`, reward 0 (`row_collector.py:543-544`); rows `1..L` =
  observation after step `k`, reward of that step (1.0); row `L` has
  `is_last = 1` and `is_terminal` as above; the env is *held* on the
  `is_last` tick (`row_collector.py:517-551`), the policy still filters the
  final observation and its action is zeroed in the row
  (`row_collector.py:329`); the next row is the next episode's reset row.
  First row of the run: `is_first = 1`, reward 0
  (`row_collector.py:221-253`).
* **Action.** The policy returns an int32 index (`train_DreamerV3.py:262`);
  `agent_action_to_env` passes discrete actions through
  (`environments/utils.py:252-253`); the env receives int32
  (`interaction.py:204-205`). The replay stores the raw index, zeroed on
  `is_last` rows.

### 2.2 Reference environment: `RR/cartpole_env.py`

A numpy port of the installed gymnax CartPole-v1 as an `embodied.Env`
subclass (`REF/embodied/core/base.py:47-82`, pattern of
`embodied/envs/dummy.py`), **not** via `from_gym` (no gym in the venv, and a
direct `embodied.Env` gives full control of the flags).

```text
class CartPolePort(embodied.Env):
  __init__(self, seed, index, recorder=None)
      self._rng = np.random.default_rng([seed, index])   # reset draws of this env
      self._recorder = recorder   # list to append finished-episode records to (training envs), or None
      self._row = -1              # row index of the last returned obs (0-based, per env)
      self._state = None; self._time = 0; self._score = 0.0; self._length = 0
  obs_space = {'vector': Space(np.float32, (4,)), 'reward': Space(np.float32),
               'is_first': Space(bool), 'is_last': Space(bool), 'is_terminal': Space(bool)}
  act_space = {'action': Space(np.int32, (), 0, 2), 'reset': Space(bool)}
  step(action):
      self._row += 1
      if action['reset']:                       # driver sends reset = previous is_last (driver.py:73)
          state = self._rng.uniform(-0.05, 0.05, 4).astype(np.float32); time = 0
          return obs(state, reward=0.0, is_first=True, is_last=False, is_terminal=False)
      assert self._state is not None and not self._done   # never step past is_last without reset
      a = int(action['action'])                 # 0 or 1
      prev_terminal = terminated(self._state)
      ... gymnax step_env arithmetic in np.float32, same operation order (cartpole.py:61-78) ...
      reward = np.float32(1.0 - prev_terminal)
      time += 1
      term = terminated(new_state); trunc = time >= 500
      is_last = term or trunc; is_terminal = term
      score += reward; length += 1
      if is_last and recorder is not None: recorder.append({worker: index, row_last: self._row,
                                                           length, score, terminal: bool(term)})
      return obs(new_state, reward, is_first=False, is_last, is_terminal)
  obs(...) -> {'vector': float32[4] = [x, x_dot, theta, theta_dot], 'reward': np.float32,
               'is_first': bool, 'is_last': bool, 'is_terminal': bool}
```

Constants: all float32 (`np.float32(9.8)`, `1.0`, `0.1`, `1.1`, `0.5`,
`0.05`, `10.0`, `0.02`, `np.float32(12 * 2 * np.pi / 360)`, `2.4`); the
threshold comparisons use the float32 state, as gymnax does under jit.
`score`/`length` reset at every reset; `score` = sum of the step rewards of
the episode = `length`.

Wrapping: the stock `wrap_env` (`main.py:228-245`, imported from
`dreamerv3.main`) on the discrete action space adds only `ExpandScalars` and
`CheckSpaces` (`embodied/core/wrappers.py:94-124, 226-250`), exactly as for
the stock envs.

Equivalence with 2.1, row by row: the driver steps env `w` once per vector
step with the previous action (`driver.py:55-65`); the first call carries
`reset = True` (`driver.py:34-38`), so row 0 is the reset row (reward 0,
`is_first`); a step that terminates or truncates returns the final
observation with `is_last` and `is_terminal = terminated`; the driver
zeroes that row's action and sets `reset` (`driver.py:70-74`), so the next row
is a fresh reset row. Same observation, reward-on-entering, flags, action
index and `L + 1` rows per episode as Ajax. Only the numbers of the reset
draws differ (numpy vs JAX PRNG; same distribution) and the float32
arithmetic may differ in the last ulp (numpy vs XLA `sin`/`cos`).

**Port self-test** (builder, before any run; `RR/test_cartpole_env.py`, run with
`APY`, which has both gymnax and numpy): for 200 random float32 states inside
the thresholds and both actions, the port's single step equals gymnax
`CartPole().step_env` (`cartpole.py:52-99`) on obs to 1e-6 absolute, and
`terminated`, `truncated`, `reward` exactly; one 600-step random-action
rollout with resets reproduces the `L + 1` row structure and the
`is_terminal` rule (truncation at time 500 with `is_terminal = False`,
forced by starting a state at `time = 499`).

---

## 3. Training loop (reference side)

### 3.1 Entry point

`RR/run_reference.py` reproduces `dreamerv3/main.py:22-66` for `script == 'train'`:
the config built from `agt.Agent.configs['defaults']` with `OVERRIDES`
(section 1), then the stock update `replay_length = batch_length` (65),
`replay_length_eval = batch_length_eval` (`main.py:35-38`), the stock
`args = embodied.Config(**config.run, logdir, batch_size, batch_length,
batch_length_eval, replay_length, replay_length_eval, replay_context)`
(`main.py:39-47`), `config.save(logdir / 'config.yaml')` (`main.py:54`),
the timer initialiser (`main.py:56-59`), and then calls
`RR/train_instrumented.train(make_agent, make_replay, make_env, make_logger,
args)` with:

* `make_agent(config)`: stock `main.py:136-144` with `make_env` below
  (builds `agt.Agent(env.obs_space, env.act_space, config)` from a port
  instance that is never stepped, then closes it).
* `make_replay = bind(dreamerv3.main.make_replay, config, 'replay')`: the
  **stock** function imported from `dreamerv3.main` (`main.py:162-188`):
  `Replay(length=65, capacity=5e6, directory=logdir/'replay', online=True,
  chunksize=1024)`, uniform selector (fracs uniform 1.0), no rate limiter.
* `make_env(config, index)`: `wrap_env(CartPolePort(config.seed, index,
  recorder=EPISODES), config)` with the stock `dreamerv3.main.wrap_env`.
  `EPISODES` is a module-level list in `RR/run_reference.py`.
* `make_logger(config)`: `embodied.Logger(embodied.Counter(),
  [embodied.logger.JSONLOutput(config.logdir, 'metrics.jsonl')],
  multiplier=1)`. Replaces the stock `main.py:147-159` only because its
  `TensorBoardOutput` needs tensorflow (absent); the logger is passive here
  (section 3.4).

If importing `dreamerv3.main` is impossible, `make_replay` and `wrap_env`
are copied **verbatim** into `RR/run_reference.py` with a comment giving the
source lines.

### 3.2 `RR/train_instrumented.py`

A copy of `REF/embodied/run/train.py` (lines 1-128) whose stock lines are
unchanged; additions are marked `# INSTRUMENTATION` and are exactly:

1. (after the stock `while step < args.steps:` body, i.e. after
   `train.py:125-126`, inside the loop) `if int(step) % args.record_every ==
   0: record(int(step))` with `args.record_every = 2000` (section 3.4).
2. (after the loop, before `logger.close()`) the **flush step**: with the
   driver's pending actions `driver.acts` (the actions the next
   `Driver._step` would send, `driver.py:56-64`), call
   `env.step({k: v[i] for k, v in driver.acts.items()})` once for each
   training env `i` in `driver.envs` (available because `driver_parallel =
   False`, `driver.py:26-28`). No policy call, no `replay.add`, no training,
   no callbacks: its only purpose is to let the env port record episodes
   whose `is_last` row is row 1 250 (the env step Ajax takes inside its last
   tick 1 249, see 5.2). Then write `episodes.jsonl` and `records.jsonl`
   finally.
3. `record(N)` (defined in the copy; reads only):
   * `train = agg.result()` (stock `embodied.Agg`, `train.py:20, 91`; reset
     on read), keeping the scalar `train/<name>` entries (drop `*/dist`);
   * `updates = int(agent.updates)` (`jaxagent.py:75, 186`);
   * `retnorm`: `lo = float(np.asarray(agent.params['agent/retnorm/low/value']))`,
     `hi` likewise with `high` (names from `RR/probe_ref_agent.py`),
     `scale = max(1.0, hi - lo)` (`jaxutils.py:382-386`);
   * the evaluation of section 4;
   * append one line to `RR/full/ref_s{s}/records.jsonl` (section 5.1);
     rewrite `episodes.jsonl` from `EPISODES`;
   * `logger.write()` (flushes the stock `episode/score`, `episode/length`
     entries added by the stock `log_step`, `train.py:56-61`, into
     `metrics.jsonl`: a cross-check of the port's episode log).

Nothing else changes: driver, callbacks (`train.py:69-72, 92`), gate and
`when.Ratio` (`train.py:26-28, 81-91`), datasets and prefetch
(`train.py:74-79`), policy lambda (`train.py:104-106`), checkpoint object
(`train.py:94-101`) are the stock lines.

### 3.3 Step counting and stopping

* `step` (`train.py:18`, `embodied.Counter`) is incremented once per
  transition (one env's row) by the first callback (`train.py:69`); the
  driver adds 16 transitions per `_step` (`driver.py:75-80`), so a reference
  step is an Ajax row and reset rows are counted on both sides.
* `driver(policy, steps=10)` (`train.py:109`) runs exactly **one** vector
  step per call: its local counter starts at 0 and reaches 16 >= 10 after one
  `_step` (`driver.py:50-53`). The loop test `step < args.steps`
  (`train.py:107`) is therefore evaluated after every vector step and the run
  stops at exactly **20 000** steps = 1 250 vector steps = rows 0..1 249 of
  each env, like Ajax's `n_timesteps // n_envs = 1250` ticks
  (`train_DreamerV3.py:618`).
* 2 000 rows = 125 vector steps: `record(N)` fires after the driver call
  that brings `step` to `N = 2000 k`, `k = 1..10`. At that point every train
  call triggered by those 16 transitions has run (`train_step` is a driver
  callback, `train.py:92`), as Ajax's log at tick `125 k - 1` runs after
  that tick's updates (`train_DreamerV3.py:559-583`;
  `log.py:120-125` with `per_update = n_envs`, `train_DreamerV3.py:673`).
* Expected updates: gate at step 1 040 (`len(replay) >= 16` once env 15
  inserts its first 65-row item, `embodied/replay/replay.py:130-138`),
  `Ratio(0.5)` afterwards: `1 + floor((20000 - 1040) / 2) = 9 481` at 20 000,
  equal to Ajax (`TrainRatio.total_updates(1250)`). `record` stores
  `updates` so the auditors can check it at every `N`.

### 3.4 Logging

The stock wall-clock blocks are off by config (`run.log_every = -1`,
`run.eval_every = -1`, `run.save_every = -1`: `when.Clock` returns False for
negative periods, `when.py:77-78`), so `agg` is read and reset only by
`record` and each window's means are over the updates of that window, as
Ajax's `MetricsAccumulator` (`state.py:278-316`; reset after each log
`train_DreamerV3.py:575-583`). Reference caveat (stock D1): `JAXAgent.train`
returns the metrics of the **previous** call (`jaxagent.py:199-202`), so a
window holds updates `n0 - 1 .. n1 - 1` instead of `n0 .. n1`.

### 3.5 Checkpointing and logdir

`logdir = RR/full/ref_s{s}`; `run_reference.py` **asserts it does not exist** before
the run (with an existing `checkpoint.ckpt`, the stock `load_or_save`,
`embodied/core/checkpoint.py:98-102`, would *load* it and resume). The stock
initial save (`train.py:100`) writes `checkpoint.ckpt` once (empty replay);
`save_every = -1` prevents any later save. The replay directory
`logdir/replay` is created by the stock `Replay` and stays empty.

### 3.6 Commands

The orchestrator starts all four processes with **`RR/launch.sh`** (absolute
paths only; refuses to start if `RR/full/` exists; CPU only). Per process it
runs, under `nohup` and in parallel:

```bash
# reference, one process per seed S in 0 1 2 (cwd = REF, as stock main.py)
cd $REF && env JAX_PLATFORMS=cpu XLA_FLAGS="$REF_XLA_FLAGS" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$REF:$RR $RPY $RR/run_reference.py --seed S --rows 20000 \
  --eval-every 2000 --eval-episodes 10 --out $RR/full/ref_sS > $RR/full/ref_sS.log 2>&1
# Ajax, one vmapped run over the 3 seeds (cwd = RR: nothing written under AJ)
cd $RR && env JAX_PLATFORMS=cpu XLA_FLAGS="$AJAX_XLA_FLAGS" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$AJ/src $APY $RR/run_ajax.py --seeds 0 1 2 --rows 20000 \
  --log-every 2000 --eval-episodes 10 --out $RR/full/ajax > $RR/full/ajax.log 2>&1
```

When a process exits, its wrapper writes its exit code to
`RR/full/<name>.DONE` (`name` = `ref_s0`, `ref_s1`, `ref_s2`, `ajax`; via a
`.tmp` file and `mv`). `REF_XLA_FLAGS` / `AJAX_XLA_FLAGS` default to empty
(default threading, measured faster than single-thread Eigen even when two
reference processes share the machine; threading changes speed only).
`PYTHONDONTWRITEBYTECODE=1` keeps even bytecode caches out of `REF` and
`AJ`. `launch_info.txt` records the launch time, load, the `REF` and `AJ`
revisions and a hash of `AJ`'s uncommitted `src` diff. When the four DONE
files exist: `$APY $RR/compare.py --root $RR/full --seeds 0 1 2`.

`run_ajax.py`: `agent = ajax.DreamerV3("CartPole-v1", model_size="1m")`;
`state, metrics = agent.train(seed=[0, 1, 2], n_timesteps=20000,
num_episode_test=10, logging_config=LoggingConfig(config={},
log_frequency=2000, use_wandb=False, use_tensorboard=False))` (one vmapped
run, as `benchmarks/learning_checks.py:11` runs a check). It must not modify
anything under `AJ`. `run_ajax.py` and `run_reference.py` record the file
the code was imported from (`ajax.__file__`; `dreamerv3.agent.__file__`,
`embodied.__file__`), checked by `compare.py`.

---

## 4. Evaluation protocol (both sides)

### 4.1 Ajax (as implemented; nothing to build)

* **When**: on ticks with `(tick + 1) % 125 == 0` (`log.py:119-125`,
  `every = 2000 // 16`), at the end of the tick, after its updates
  (`train_DreamerV3.py:575-578`): `N = 2000, 4000, ..., 20000`.
* **What** (`train_DreamerV3.py:466-491`, `evaluate.py:661-762`):
  `num_episode_test = 10` freshly reset parallel envs, the bare gymnax
  `CartPole` (`evaluate.py:130`, `env.unwrapped`), **one episode per env**:
  a scan of `horizon = 500` steps (`evaluate.py:716`,
  `utils.agent_episode_length`), the return and length of each env summed
  only while it has not yet been done (`evaluate.py:736-739`); `done =
  terminated or truncated`. Only each env's first episode counts, so short
  episodes are not over-represented. Return = episode length, at most 500.
  Logged: the mean return over the 10 envs (`Eval/episodic mean reward`) and
  the mean length.
* **Policy**: the current parameters after the tick's updates
  (`bind_policy`, `train_DreamerV3.py:267-281, 483`); zero carry
  (`initial_policy_carry`, `train_DreamerV3.py:205-215, 484`), `is_first =
  True` on the first step only (`evaluate.py:755, 744`); **sampled**
  posterior and actor (`policy_step`, `train_DreamerV3.py:218-264`; no
  deterministic mode, as the reference).
* **Randomness**: `eval_key = split(agent_state.eval_rng)[0]`
  (`log.py:128`); `eval_rng` is set once at init
  (`train_DreamerV3.py:368, 377`) and never advanced (`maybe_eval_and_log`
  only replaces `n_logs`, `log.py:152-154`). **Every evaluation of a seed
  therefore uses the same 10 initial states and the same noise keys**; only
  the parameters change between evaluations.

### 4.2 Reference (instrumentation in `record`, `RR/train_instrumented.py`)

Mirror of 4.1 with the reference's own policy function:

* **When**: in `record(N)`, i.e. after the vector step that brings `step` to
  `N` and all its updates (3.3).
* **Envs**: 10 fresh `CartPolePort` instances wrapped with the stock
  `wrap_env`, `recorder=None`, separate from the 16 training envs; env `i`'s
  reset generator `np.random.default_rng([s, 7_000_001, i])` is **re-created
  at every evaluation** (mirrors Ajax's fixed eval key: same 10 initial states
  at every evaluation of seed `s`). First obs: `env.step({'action': 0,
  'reset': True})` (reset row, `is_first = True`).
* **Parameters**: the **latest trained** policy parameters,
  `params = {k: agent.params[k].copy() for k in agent.policy_keys}`
  (`jaxagent.py:77-78, 81-83`; `.copy()` because the next `JAXAgent.train`
  deletes the non-synced policy buffers, `jaxagent.py:188-192`). Not
  `agent.policy_params`, which lag by one to two vector steps (stock D1) and
  which Ajax's evaluation has no counterpart of.
* **Carry**: `carry = agent._init_policy(params, seed0, batch_size=10)`
  (`jaxagent.py:297-299, 352-353`: `RSSM.initial` zeros and zero previous
  action, `agent.py:114-118`).
* **Steps**: up to 500 iterations: `obs = agent._filter_data(stacked obs of
  the 10 envs)` (`jaxagent.py:396-397`); `acts, _, carry =
  agent._policy(params, obs, carry, seed_t, 'eval')` (`jaxagent.py:301-303,
  354-355`; the same `Agent.policy`, `agent.py:129-164`, which samples); for
  each env not yet done: `o = env.step({'action': acts['action'][i],
  'reset': False})`, `ret[i] += o['reward']`, `len[i] += 1`,
  `done[i] |= o['is_last']`; a done env keeps being fed its last observation
  (its outputs are ignored; the batch rows are independent). Stop when all
  are done or after 500 iterations (the port truncates at 500 anyway).
  `seed0` and every `seed_t` are `uint32[2]` drawn from
  `np.random.default_rng([s, 7_000_001, 1_000_000])`, re-created at every
  evaluation, and `jax.device_put` explicitly.
* **No side effect on training**: the evaluation calls the jitted `_policy`
  directly, not `JAXAgent.policy`, so it does **not** draw from `agent.rng`
  (`jaxagent.py:140, 391-394`) and does **not** trigger the pending
  parameter swap of the training policy (`jaxagent.py:147-153`). (Going
  through `JAXAgent.policy` would consume the agent's RNG stream, which alone
  would be acceptable, but it would also change the staleness of the
  training actor's parameters, which is not; hence the direct call.)
* **Outputs**: per evaluation, the 10 returns and lengths, their means
  (`eval_mean`, `eval_len_mean`).

Unavoidable differences: the draws themselves (numpy vs JAX streams), and
`eval` compiles the policy once for batch 10 (time only).

---

## 5. Recording and statistics

### 5.1 Output files

Directory `RR/full/` (smoke runs: `RR/out_smoke/`):

* `ref_s{s}.DONE`, `ajax.DONE` (exit code of each process, written by
  `launch.sh`), `ref_s{s}.log`, `ajax.log`, `*.nohup`, `pids.txt`,
  `launch_info.txt`;
* `ajax/config.json`: Ajax's resolved config (constructor arguments,
  `DreamerV3Config`, agent config, schedule, ring rows, run arguments,
  `ajax_file`); `ref_s{s}/run_args.json` (run arguments, overrides, imported
  file locations) and `ref_s{s}/timing.json` (written by `Instr.finish`
  after the flush step: `flush_episodes`, `t_end`, timings);

* `ref_s{s}/records.jsonl` and `ajax_records.jsonl` (Ajax: one file, a
  `seed` field per line): one JSON object per log point `N`, same schema on
  both sides:

  ```json
  {"side": "ref|ajax", "seed": 0, "rows": 2000, "updates": 481,
   "eval_mean": 22.0, "eval_len_mean": 22.0, "eval_returns": [..10..] | null,
   "s2_logged": 21.3 | null,
   "train": {"<ajax name>": mean over the window, ...},
   "train_raw": {"<native name>": mean over the window, ...},
   "retnorm": {"lo": .., "hi": .., "scale": ..} | null,
   "wall_s": 123.4}
  ```

  Ajax: `rows` = `timestep`, `updates` = `Train/n_updates`, `eval_mean` =
  `Eval/episodic mean reward`, `eval_len_mean` = `Eval/mean episodic length`,
  `eval_returns` = null (Ajax returns only the mean,
  `evaluate.py:762`), `s2_logged` = `Train/episodic mean reward`, `train` =
  `train_raw` = every `Train/<name>` with the prefix stripped, `retnorm` =
  null; selected as `diag.py` does (ticks where the eval is finite,
  `SP/m7/lc/diag.py:31-32`). Reference: `s2_logged` = null (S2 is computed
  by `compare.py` from `episodes.jsonl`), `train` = `train_raw` renamed with
  the map of 5.4, `train_raw` = the `agg.result()` scalars with the `train/`
  prefix stripped.
* `ref_s{s}/episodes.jsonl`: one line per finished training episode,
  `{"seed", "worker", "row_last", "length", "score", "terminal"}`, from the
  port's recorder, including the flush step (3.2).
* `ref_s{s}/metrics.jsonl` (stock logger, cross-check), `config.yaml`,
  `checkpoint.ckpt`; `ref_s{s}.log`, `ajax.log` (stdout).
* `ajax_final.npz`: per seed, the final
  `state.collector_state.episodic_return_state` (`buffer [seeds, 10, 16, 1]`,
  `count`, `index`, `sum`), `last_episode_length`, `state.retnorm.lo/hi`,
  `n_updates` (state reads only), for the S2 audit.
* `summary.json` written by `RR/compare.py`, plus its printed tables.

### 5.2 Pre-registered statistics (fixed; not to be changed after results exist)

* **S1** (gate statistic), per seed: the mean of `eval_mean` at `N = 16000,
  18000, 20000` (3 evaluations x 10 episodes; equivalently the mean of the 30
  episode returns).
* **S2**, per seed, at `N = 20000` (and recorded at every `N`): Ajax's
  `Train/episodic mean reward` definition, made explicit:
  for env `w` let `E_w(N)` be its finished training episodes whose `is_last`
  row has per-env row index `row_last <= N / 16` (0-based; the episode is
  finished by the env step Ajax takes in tick `N/16 - 1`), in completion
  order, `k_w = |E_w(N)|`, and `m_w` the mean return of its **last
  `min(10, k_w)`** episodes; `S2(N) = (1/16) sum_w m_w` if every `k_w >= 1`,
  else NaN.
  Ajax: by construction of `update_episodic_return` /
  `update_rolling_mean` (`interaction.py:79-101, 492-551`: per-env ring of
  window 10, `count = min(count + 1, 10)`, mean over envs of `sum / count`,
  NaN while any env has `count = 0` since `0/0`), the window size 10
  (`row_collector.py:211, 235-239`), the episode counted when the env step
  ends it (`ended = is_last & stepped`, `row_collector.py:367-374`) and the
  return = sum of the stepped rewards (`row_collector.py:368`). The Ajax value
  is read from the log (`s2_logged`); `compare.py` **audits** it at 20 000
  against the final buffer in `ajax_final.npz` (`mean_w sum_w / count_w`,
  tolerance 1e-4).
  Reference: computed by `compare.py` from `episodes.jsonl` with the formula
  above (score = sum of the episode's step rewards, as Ajax's return and as
  the stock `episode/score`, `train.py:37-42, 56-61`). The flush step (3.2)
  supplies the episodes with `row_last = 1250`.
* Also recorded, not part of the decision: the eval curves (`eval_mean`,
  `eval_len_mean`) and `S2(N)` at every `N = 2000 k`, the training-metric
  window means (5.4), `updates` at every `N`, the reference's retnorm state
  at every `N` and Ajax's at 20 000.
* **Seeds**: 0, 1, 2 on each side; the seed values index independent runs
  (the two codebases' RNGs differ anyway).
* **Reported rule** (printed by `compare.py`; the decision is the user's):
  `A = mean over Ajax seeds of S1`, `[lo, hi] = [min, max]` over reference
  seeds of S1. `LEARNS = (A >= lo)`, `WITHIN_RANGE = (lo <= A <= hi)`. Print
  both booleans, `A`, `lo`, `hi`, the per-seed S1 and S2(20000) tables for
  both sides, the eval and S2 curves for both sides, and a side-by-side
  table of the mapped training metrics per window (mean over seeds, with
  min/max).

### 5.3 `RR/compare.py`

Reads the files of 5.1 and asserts (exit 1 if any fails; full mode unless
`--smoke`):

* every process finished with exit code 0 (`RR/full/<name>.DONE` holds `0`;
  required in full mode, checked when present in smoke mode);
* reference, per seed: the `config.yaml` values of section 1's "This run"
  column; `run_args.json` = seed, `tiny` False, rows 20 000, eval every
  2 000, 10 evaluation episodes (full mode); every evaluation has that many
  episodes; the reference code was imported from `REF`;
* Ajax: `ajax/config.json` = every `DreamerV3Config` value of the protocol
  (widths 64 / 64 / 512 / 4 / 32 / 8 and all algorithm values, as audited;
  the `--tiny` widths in a tiny smoke run), agent config train ratio 512,
  batch 16 x 64, replay capacity 5e6, constructor `CartPole-v1` / `1m` / 16
  envs / no extensions, ring rows = rows / 16, gate 1 040, total updates =
  `1 + floor((rows - 1040) / 2)`, seeds = the compared seeds, Ajax imported
  from `AJ/src`; full mode also `tiny` False, rows 20 000, log every 2 000,
  10 evaluation episodes;
* 10 records per seed per side with `rows = 2000 k`; `updates` at every `N`
  equal to `1 + floor((N - 1040) / 2)` on both sides (9 481 at 20 000);
* the Ajax S2 audit, and Ajax's final state rows / updates;
* the reference episode log consistency (`length == score`, every
  `length <= 500`, `terminal == False` only when `length == 500`; per
  worker, the episodes tile rows `0..row_last` with `L + 1` rows each);
* the reference run finished and its flush step ran: `timing.json` has
  `t_end` and `flush_episodes`, and `flush_episodes` equals the number of
  episodes with `row_last = rows / 16` (only the flush step can produce
  them), so S2(20 000) is never computed from a log without the flush.

Then prints and writes `compare.md`, `summary.json` and `curves.png` as
specified in 5.2, with a "Known differences" section listing D1 / U1
(stale acting parameters, prefetched first batch), D2 / U2 (shared
reference replay-sampler stream: the reference S1 range comes from three
not fully independent runs), the report prefetch stream (6.1), and the NaN
semantics of the two window means (5.4) together with any non-finite Ajax
window means of the run.

### 5.4 Training-metric name map

Ajax names (`train_metric_keys`, `train_DreamerV3.py:316-356`; values
`learner.py:419-470, 553-580`) to reference names (`agent.py:355-389`;
optimizer `jaxutils.py:540-549`, prefix `opt_` from the optimizer's name
`opt`, `agent.py:88`; Agg prefix `train/`, `train.py:91`). Both sides log the
**unscaled** per-term loss means (`agent.py:355`, `learner.py:431-440`).

| Quantity | Ajax `Train/<name>` | Reference `train/<name>` |
|---|---|---|
| decoder (vector) loss | `rec_loss`, `rec_loss_std` | `vector_loss`, `vector_loss_std` |
| reward loss | `rew_loss`, `rew_loss_std` | `reward_loss`, `reward_loss_std` |
| continue loss | `con_loss`, `con_loss_std` | `cont_loss`, `cont_loss_std` |
| dynamics KL (free-bits) | `dyn_loss`, `dyn_loss_std` | `dyn_loss`, `dyn_loss_std` |
| representation KL | `rep_loss`, `rep_loss_std` | `rep_loss`, `rep_loss_std` |
| actor loss | `actor_loss`, `actor_loss_std` | `actor_loss`, `actor_loss_std` |
| critic loss (imagination) | `critic_loss`, `critic_loss_std` | `critic_loss`, `critic_loss_std` |
| replay critic loss | `repval_loss`, `repval_loss_std` | `replay_critic_loss`, `replay_critic_loss_std` |
| total loss, grad / update / param norms, optimizer steps | `opt_loss`, `opt_grad_norm`, `opt_update_norm`, `opt_param_norm`, `opt_grad_steps` | same names |
| policy entropy (imagination) | `ent/action/{mean,std,mag,min,max}` | same |
| normalised entropy | `rand/action/{...}` | same |
| imagined actions | `act/action/{...}` | same |
| advantages | `adv/{...}` | same |
| imagined rewards | `rew/{...}` | same |
| weights (cumulative continuation) | `weight/{...}` | same |
| values | `val/{...}` | same |
| lambda returns | `ret/{...}` | same |
| normalised returns `(ret - lo) / scale` | `ret_normed/{...}` | same |
| replay returns | `replay_ret/{...}` | same |
| prior / posterior entropy | `prior_ent/{...}`, `post_ent/{...}` | same |
| TD error, return rate | `td_error`, `ret_rate` | same |
| data / predicted reward | `data_rew/{max,mean,std}`, `pred_rew/{max,mean,std}` | same |
| embedding magnitude | `activation/embed` | same |
| return-normaliser scale | derived: window mean of `ret/std` / window mean of `ret_normed/std` (approximate, both sides identically); exact at 20 000 from `ajax_final.npz` | exact at every `N`: `retnorm.scale` of `records.jsonl` (state read), and the same derived ratio |
| updates so far | `n_updates` | `int(agent.updates)` (`updates` field) |
| reference only (recorded, not compared) | none | `rewstats/*`, `constats/*`, `opt_param_count`, `*/dist` (dropped) |

`{...}` = `{mean, std, mag, min, max}` (`jaxutils.py:52-66`; Ajax `learner.py:405-417`
without the random `dist` subsample).

Window means and NaN: the reference `Agg` (`embodied/core/agg.py:80-87,
107-117`) skips NaN values (a NaN does not count, and a NaN initial value
is replaced by the next finite one); Ajax's `MetricsAccumulator`
(`state.py:297-309`) adds every value, so one NaN makes its window mean NaN
(written as `null` by `run_ajax.py`). The two agree whenever all values are
finite. A NaN Ajax window against a finite reference window therefore
means "a NaN occurred in that Ajax window", not a divergence;
`compare.py` lists the non-finite Ajax windows.

---

## 6. Departures from the stock reference, and unmatchable settings

### 6.1 Everything the reference run adds to or changes in stock 29eb964

1. **Config overrides** (section 1, `OVERRIDES`): 1m dims (rows 3-6), train
   ratio 512, float32, cpu, `run.steps` 20000, `run.log_every` /
   `run.eval_every` / `run.save_every` = -1, `run.driver_parallel` False,
   `jax.prealloc` / `jax.transfer_guard` False, `seed`, `task` label,
   `logdir`. None changes an algorithmic code path: the driver computes the
   same transitions in-process, and `log_every` / `save_every` only read
   state (log) or write a checkpoint. `eval_every = -1` does change one
   input to training slightly, through the report stream (paragraph below).
2. **Environment**: `RR/cartpole_env.py` (a new `embodied.Env`, section 2.2)
   with an episode recorder (append-only list; reads the env's own values).
3. **Entry point** `RR/run_reference.py`: stock `main.py` config/args logic;
   `make_env` (port + stock `wrap_env`), `make_logger` (JSONL only, no
   tensorflow), `make_agent` (stock body, port spaces), stock `make_replay`.
4. **`RR/train_instrumented.py`**: stock `train.py` plus the three
   `# INSTRUMENTATION` additions of 3.2: `record(N)` at multiples of 2 000
   steps (reads `agg`, `agent.updates`, retnorm parameters; evaluation;
   file writes), the post-loop flush env step (reads episode ends only), and
   the final file writes.
5. **Evaluation** (section 4.2): separate env instances, separate numpy
   generators, direct calls of the jitted `_init_policy` / `_policy` with a
   copy of the latest policy parameters.
6. **Recording only**: `run_args.json` (arguments, overrides, jax version,
   XLA flags, the files `dreamerv3.agent` and `embodied` were imported
   from), `timing.json`, `progress.jsonl`.
7. **Launch** (`RR/launch.sh`, 3.6): `nohup` wrapper writing a DONE marker
   with the exit code; `PYTHONDONTWRITEBYTECODE=1`; no XLA flags by default.

The stock **report prefetch stream** is still created (`train.py:76-77`,
`dataset_report`) although `agent.report` never runs (`should_eval` is
`when.Clock(-1)`, always False, `when.py:76-78`). `agent.dataset` wraps it
in `embodied.Prefetch(amount=1)` (`jaxagent.py:233-238`), whose thread starts
at once (`prefetch.py:11-15`) and, as soon as the replay holds an item
(`limiters.MinSize(1)`; the first items appear around row 1 040, when each
worker has 65 rows), assembles one batch (16 `Replay._sample` calls, each
popping the online queue first, `replay.py:178-182`, else drawing from the
`Uniform` selector; plus one `agent.rng` seed draw), puts it in the queue,
assembles a second one and blocks on `put` for good. So it consumes **up to
32 samples, i.e. up to 32 of the earliest online-queue items** (around rows
1 025-1 100), which training then does not receive as online items (they
stay in the replay for uniform sampling). In a stock run (`eval_every` 180
s of wall time) it would also take 16 more samples (one report batch of
length 33) every 180 s, a timing-dependent perturbation the override
removes. Ajax has no report stream (D2). Impact: negligible; training
itself draws 128 samples per vector step against 16 new items, so it
drains the online queue every step. This is stock behaviour and is kept;
`compare.md` repeats it in its "Known differences" section.

### 6.2 UNMATCHABLE settings and expected impact

| # | What differs | Reference (stock) | Ajax | Expected impact on S1 / S2 |
|---|---|---|---|---|
| U1 | Asynchrony (D1): acting parameters, prefetch, metrics | policy parameters swapped in only after the next policy call and only the pre-update parameters of the first update since the last swap are queued (`jaxagent.py:147-153, 188-192`): the training actor lags by about 1-2 vector steps (~8-16 of 9 481 updates); train/report batches prefetched by threads before the gate and one batch ahead (`prefetch.py:32-41`); write-back and metrics one call late (`jaxagent.py:194-202`) | synchronous | Small: a lag of ~1 % of the run's updates; the first prefetched batches may come from the first items only. Measured on the real reference (`RR/audit_loop`, tiny, seed 0): acting parameters equal the latest at rows 400 / 800 (0 updates) and differ at 1 200 / 1 600 / 2 000; the first training batch had 11 distinct items of 16. Evaluation uses the latest parameters on both sides. Not expected to change whether CartPole is learned. |
| U2 | Replay sampler RNG (D2) | `Uniform(seed=0)` for every run seed (`selectors.py:29-32`, `main.py:187`): the three reference seeds share the index stream (the data differ) | seeded per run | Small (the replay contents differ between seeds); slightly reduces the independence of the three reference seeds, so the reference S1 [min, max] range of the reported rule comes from three not fully independent runs. Stock behaviour, kept (`make_replay` must stay stock); disclosed in `compare.md`. |
| U3 | Update placement within a vector step (D3) | an update can run between env `w` and env `w+1`'s adds (`train.py:81-92`) | after the tick's 16 adds | Negligible (same rate, intra-tick order). |
| U4 | Two-hot bins and expectation (D22); output width 256 vs 255 | float32 bins inside jit; zero-init heads predict 0.07-0.16; 256 logits, last dropped | float64-rounded bins; exactly 0 at init; 255 logits | Width: none. Values: a tiny initial reward/value offset; negligible after the first updates. |
| U5 | Kernel init constant (D23) | `1.1368 sqrt(1/fan_in)` truncated normal | `variance_scaling` truncated normal | 4.16e-5 relative scale: negligible. |
| U6 | RNG streams | numpy `Generator(seed)` seeds shared by the policy and prefetch threads (`jaxagent.py:39, 391-394`), params from `[seed, 0]`, port resets from `default_rng([seed, w])` | JAX keys from `seed` | Different draws on every seed: handled statistically by 3 seeds per side. The reference is not bitwise reproducible across reruns (thread timing of the prefetch seeds and online-queue pops). |
| U7 | Env arithmetic | numpy float32 | XLA float32 | Last-ulp differences; same distribution. |
| U8 | Evaluation draws | numpy streams (structure of 4.1 mirrored: same initial states at every evaluation of a seed) | fixed JAX `eval_rng` | None on expectations; per-evaluation numbers differ. |

---

## 7. Builder checklist (smoke and speed only; the orchestrator launches the long runs)

1. `RR/test_cartpole_env.py` (section 2.2) passes with `APY`
   (`RR/test_cartpole_env.log`).
2. Reference smoke, seeds 0 and 1 (cwd `REF`, environment as in 3.6):
   `run_reference.py --seed S --rows 1200 --eval-every 400 --eval-episodes 2
   --tiny --out $RR/out_smoke/ref_sS` (400 = 25 vector steps; 1 200 > 1 040
   so the gate opens and 1 + floor(160/2) = 81 updates run): 3 records with
   `updates` 0, 0, 81; fresh logdir (no checkpoint loaded).
3. Ajax smoke (cwd `RR`): `run_ajax.py --seeds 0 1 --rows 1200 --log-every
   400 --eval-episodes 2 --tiny --out $RR/out_smoke/ajax`: 3 records per
   seed, `updates` 0, 0, 81.
4. `compare.py --root $RR/out_smoke --seeds 0 1 --smoke` passes every check
   (checkpoints 400 k, S1 window = the last three checkpoints, smoke only).
5. Speed: full-size reference runs of about 1 200-1 500 rows
   (`RR/out_speed/ref_s*`, `progress.jsonl`): wall clock per update after
   the gate, build + compile time, minutes per seed for 9 481 updates; same
   for Ajax (`RR/out_speed/ajax_1200`).
6. No command over ~15 minutes; CPU only; nothing written outside `RR`;
   `launch.sh` is checked (`bash -n`, wrapper idiom tested separately) but
   **not run** by the builder: the orchestrator launches it, and its
   outputs go to `RR/full/`, separate from the smoke and speed outputs.
