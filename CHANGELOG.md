# Changelog

Versions follow semantic versioning; before 1.0, a breaking change bumps
the minor version. Each release is tagged `vX.Y.Z` on main and published to
PyPI as `ajax-rl`.

## Unreleased

- **RVI-SAC.** A new average-reward agent, `RVISAC` (Hisaki and Ono, ICML
  2024): SAC learned by RVI Q-learning, its target `r - c 1(terminated) -
  xi + V'(s')` going on through terminations, `xi` following the batch mean
  of the next values, and a reset cost `c` learned as a Lagrange
  multiplier on the termination rate. Its defaults, initialisation,
  update order and evaluation action are the authors'
  (yhisaki/average-reward-drl); one update matches theirs leaf by leaf
  (`tests/conformance/test_rvisac_original.py`).
- **ValueCentering and ResetCost.** Two extensions bring RVI-SAC's pieces
  to the discounted agents. `ValueCentering(rule)` centres a one-step
  target on a learned reward rate (Naik et al., RLC 2024) by one of three
  rules in `ajax.agents.centering`: `RVIBatchMean`, `TDErrorCentering`,
  `ReferencePoint`. `ResetCost()` has the target go on through
  terminations and charges them a learned cost (`ajax.agents.reset_cost`).
  RVI-SAC calls the same functions.
- **What `on_target` may learn from.** An extension's `on_target` may now
  return `(target, metrics, ext_state)` to update its own state, and its
  batch holds `next_values` (the value the target bootstrapped),
  `terminated`, `predictions()` and `target_value_at(obs, actions)`.
  `update_steps.critic_step` returns the agent state (its critics stepped,
  the extensions' states updated) in place of the critic state, and
  `ExtensionStack.apply_on_target` returns the states as a third value.
  The agents' target helpers (`compute_td_target`, `compute_td3_td_target`,
  `compute_redq_td_target`, `compute_tqc_td_target`,
  `compute_asac_td_target`, `compute_avg_td_target`, IQL's
  `compute_bellman_target`, DQN's two) also return the value they
  bootstrapped. DQN's `td_target_fn` may return `(target, next_actions,
  next_values)`; one returning the target alone still works. Two new
  capabilities: `one_step_target` and `continuing_target`. An
  `Extension.check_stack(stack)` hook refuses a stack that cannot hold an
  extension. The outputs of every existing agent and extension are
  unchanged.
- **Pretraining keeps the states its updates move.** A pretraining that
  runs the agent's updates (PriorData, BellmanPretrain) now keeps the
  states those updates gave the other extensions, and a later pretraining
  reads the states the earlier ones left, where it read their initial
  ones.
- **`NetworkConfig.squash_sigma_point_mean`.** A squashed policy's
  deterministic action can be the mean of `tanh` at the latent mean plus
  and minus its standard deviation (RVI-SAC's authors' evaluation action),
  in place of `tanh` of the mean.

## 0.5.0

What a user of 0.4.0 meets:

- **Two new agents**, CrossQ (Bhatt et al., ICLR 2024) and TQC (Kuznetsov
  et al., ICML 2020), each checked one training step at a time against its
  authors' code. Ajax has eighteen agents.
- **A trained agent acts and loads in one call**: `agent.policy`,
  `agent.predict` and `load_checkpoint`.
- **Logging goes through backends**, TensorBoard by default.
- **Each agent is one file on a shared family**, and the expert tools
  (residual RL, gain mode, observation features, EDGE, IBRL, JSRL, the
  warm-up and the prefill) work on every agent they fit.
- **Random numbers are stable**: keys are derived by name, so an option or
  an extension leaves every other draw as it was, and resets are keyed by
  episode.
- **Agents do what their authors' code does** where they differed: ASAC,
  REDQ, AVG, PQN, APO, IQL, APG's schedule and IBRL.
- **A replay reserves memory for the rows a run writes**, and the replay
  and incremental agents' updates run faster on a CPU.
- **The documentation site has a version per release.**

### Upgrading from 0.4.0

Each change below can break code or results that worked on 0.4.0, with
what to do about it. The sections after it give the details.

- **Every run draws other numbers from the same seed.** Keys are derived
  by name, collection and updates draw on two streams, resets are keyed by
  episode and evaluations by phase and timestep; several agents now follow
  their authors' code (next item); and the replay and incremental agents'
  update runs under a jit of its own, which XLA rounds differently.
  *What to do:* compare 0.5.0 results with 0.5.0 runs, not with 0.4.0's.
  `episode_keys=False` (`ajax.environments.create.build_env_from_id`, or
  `with_episode_keys(env, on=False)`) keeps 0.4.0's resets, but not its
  other draws.
- **Agents that now follow their authors' code**, so their results move:
  ASAC (a termination drops the next value), REDQ, AVG (its log-std head
  read as a log-variance), PQN and APO (time limits), IQL (its two
  critics' errors summed), APG (its learning-rate schedule steps the
  matrices alone), IBRL (its target selects on a fresh critic subset),
  EDGE (it scores the action executed). DQN draws its warm-up actions as
  the other agents do; PPO, APO and IQL start a PID actor's log-std at -1,
  as the others do; PPO, APO, PQN and AVG keep one set of observation
  statistics, in the collector. *What to do:* nothing; re-run results you
  compare against.
- **IBRL acts as its authors' code does by default**
  (`IBRL(hybrid_acting=True)`): collection and evaluation execute the
  expert's action where the target critics rate it higher. *What to do:*
  `IBRL(hybrid_acting=False)` keeps 0.4.0's IBRL, which changed the target
  alone.
- **SAC with an expert no longer adds `ExpertWarmup(fraction=0.7)` and
  `ExpertPrefill(n_steps=20_000)` on its own.** *What to do:* pass them in
  `extensions=` to keep them.
- **Runs log to TensorBoard by default**, under `./tensorboard/<run id>/`,
  where 0.4.0 logged to Weights & Biases. `use_wandb` and
  `use_tensorboard` still work, with a FutureWarning. *What to do:*
  `LoggingConfig(backends=(WandbBackend(project, group),))` for W&B (with
  `pip install "ajax-rl[wandb]"`), `backends=()` for no files.
- **Checkpoints are written in format 3.** One saved before resets were
  keyed by episode is refused by an agent whose environment keys them.
  *What to do:* resume it with the environment built with
  `episode_keys=False`. `load_checkpoint(agent, path)` restores a
  checkpoint without a 0-step `train` call.
- **Brax environments reset as Playground's do** (`FreshAutoResetWrapper`)
  when resets are keyed by episode, the default. *What to do:*
  `episode_keys=False` keeps 0.4.0's stack.
- **Public names removed or moved.** The 0.4.0 compatibility layer is gone
  (Removed, below): each agent's `train_<A>`, `state`, `utils` and `core`
  modules, the `<A>Config` views and `agent.agent_config`, SAC's
  deprecated keywords, and the old step and loss names. Besides:
  - `agent.dreamer_config`, and TD-MPC2's `agent.agent_config`, are
    `agent.model_config`;
  - `ajax.agents.DreamerV3` and `ajax.agents.TDMPC2` no longer re-export
    `DreamerV3Config`, `MODEL_SIZES` and `TDMPC2Config`: import them from
    `ajax.agents.DreamerV3.state` and `ajax.agents.TDMPC2.state`;
  - `PIDTermsWrapper` is the `PIDTermsInput` extension;
    `ActionPipelineResult.rng` is gone; `ajax.agents.steps` is
    `ajax.agents.update_steps`; the ring rules moved from
    `ajax.agents.DreamerV3.replay` to `ajax.buffers.ring`;
  - a name an agent module only imported for its own use (`ActorCritic`,
    `EnvParams`, `CNNSpec`, `AlphaConfig`, `run_config` and the like) no
    longer imports from that module: import it from its own module;
  - the helpers of SAC's removed observation keywords (`augment_obs`,
    `augment_from_batch`, `expert_state_dim`, `flatten_expert_state` and
    `maybe_append_train_frac` in `ajax.evaluate` and
    `ajax.environments.interaction`, `expert_next_actions`) are gone:
    `ExpertActionInput`, `ExpertStateInput` and `TrainingProgressInput` do
    their work.
- **Calls that ran on 0.4.0 and now raise**, each naming what to pass:
  `EDGEExploration` under gain mode (EDGE executes the expert's env action
  where the policy's gains go), refused when the agent is built; `BellmanPretrain` without a seeded replay
  (`ExpertPrefill` or `PriorData`), where it ran on an empty one;
  `ConditioningMetrics` and `BiasVoreDecomposition` on an agent that
  stores no steps (APG), where they logged nothing; IBRL's acting with an
  expert other than the agent's under a residual policy; a replay that has
  wrapped, resumed by an agent with a larger `buffer_size`.
- **No dependency moved into an extra.** In a checkout, `uv sync` no longer
  installs wandb: `uv sync --group wandb` does.
- **The benchmark** (`benchmarks/agent_bench.py`) adds CrossQ, TQC, IQL
  and DAgger and records each row's device; the other agents' runs are
  unchanged, and its baseline is re-recorded with five trials.

### New since 0.4.0, in brief

Not all of it has a section below:

- **CrossQ** and **TQC**: declared agents, with conformance checks against
  their authors' code; critics may predict quantiles
  (`NetworkConfig(critic_quantiles=N)`).
- **IBRL** acts as its authors' code does (`hybrid_acting`, on by default)
  and can choose softly (`soft_choice=True`, `soft_choice_beta`), as their
  `ibrl_soft`; its target selects on a fresh critic subset.
- **The expert tools reach every agent they fit**: EDGE's gates every
  continuous off-policy agent with an expert; residual RL and gain mode
  every continuous agent with one; the observation features
  (`ExpertActionInput`, `ExpertStateInput`, `TrainingProgressInput`,
  `PIDTermsInput`) every agent on the shared collector. Evaluations log
  the replay's expert share (`Eval/expert share`).
- **Evaluation acts through the agent's controller**, as collection does.
  Fixed: Brax and Playground evaluation now keeps the wrappers a user put
  over Ajax's environment; it ran without them.
- **The update's own jit**: the replay and incremental agents' update is
  compiled as a call of its own, which XLA's CPU pipeline runs about 8-10%
  faster (DQN, SAC, AVG; PERFORMANCE_REPORT.md), a deliberate workaround
  for its scheduling.

### Changes in detail

- **A replay reserves memory for the rows its run writes.** The replay
  agents' replays (SAC, TD3, REDQ, ASAC, TQC, CrossQ, IQL, DQN, DAgger,
  their recurrent forms too) held `buffer_size // n_envs` rows per env
  however short the run: 38 MB per seed on Pendulum. A run now holds the
  rows it writes, those written before training (ExpertPrefill, PriorData,
  an offline dataset) included, plus one, at most `buffer_size // n_envs`
  per env; a resumed run grows the replay to the uninterrupted run's, as
  the world models' do. The numbers are unchanged: the runs replay the
  same rows. A checkpoint saved before restores and resumes as it did.
  A replay that has wrapped and is resumed by an agent with a larger
  `buffer_size` now raises, naming `buffer_size`, as DreamerV3's does; it
  kept its smaller ring silently.
  The ring rules moved from `ajax.agents.DreamerV3.replay` to
  `ajax.buffers.ring`: `count_ring_rows` as it was, `size_ring` now
  returning the ring length alone (each replay grows its own state).
- **Logging backends.** `LoggingConfig(backends=(...))` names where every
  run logs its evaluations: `TensorBoardBackend(folder)`,
  `WandbBackend(project, group)`, or a backend of your own, any picklable
  object with `open(run)`, `log(run_id, step, metrics)` and `close()`
  (`LoggingBackend`; an optional `start(run)` runs in the caller before
  training). One backend object serves every seed's run. `ajax.logging` now
  exports `LoggingConfig`, `LoggingBackend`, `RunInfo`,
  `TensorBoardBackend` and `WandbBackend`; `ajax.logging.wandb_logging`
  still works. `backends=()` logs nowhere; train still returns the evaluations.
  Nothing default changes: without `backends`, `use_wandb` (on
  by default) and `use_tensorboard` decide as before. Setting either flag
  together with `backends` is refused. `config` defaults to an empty dict.
  `WandbBackend(steps=False)` creates each run and sends it no evaluation,
  which `use_wandb` with `sweep=True` names, as before.
- **The documentation site has a version per release.** It opens on the
  latest release's documentation; each release keeps its own as X.Y, and
  main's is the development version, dev, marked as such. A release's
  pages link to its source at its tag. The Release workflow deploys a
  release's version (mike).
- **A trained agent acts and loads in one call.** `agent.policy(state,
  seed_index)` returns a policy: `carry = policy.reset(n_envs)`, then
  `action, carry = policy.act(obs, carry, done, reward)` acts on the
  environment's own observations, in its units, exactly as the agent's
  evaluation acts (the observation normalised as the agent does, its
  features, a residual policy or gain mode, a recurrent actor's memory,
  UDRL's command, a world model's state). `agent.predict(state,
  observation)` returns one action for an agent without memory.
  `load_checkpoint(agent, path, **train_options)` restores a saved state
  into an agent built with the same arguments, its skeleton built with the
  run's train options where they shape the update (APG's schedule). TDMPC2MultiTask has no policy yet.
  Training and evaluation are unchanged: their programs are the same.
- A gymnax step no longer warns that float64 is unavailable: its
  terminated and truncated flags convert to float32 directly. Every
  number is unchanged.
- **ConditioningMetrics and BiasVoreDecomposition log on PPO, APO, PQN
  and AVG without `expose_recent_rollout`.** An extension declaring
  `reads_last_rollout` has an on-policy or incremental agent keep its
  latest rollout; before, these logged nothing. They now require
  `stored_steps` (a replay or such a rollout), which APG lacks.
  `expose_recent_rollout` still works. Runs that logged these metrics
  are unchanged.
- An extension passed bare (`extensions=X`) is refused with a TypeError
  that shows the tuple to write, `extensions=(X(...),)`, instead of
  saying only that it is not iterable.
- **Documentation**: the agent and extension totals match the code
  (thirty-two extensions; `PIDTermsInput` joins the catalogue), the memory
  table lists every agent, and Why Ajax adds TorchRL and Tianshou, drops
  TQC and CrossQ from what Ajax lacks and installs from PyPI. A fast test
  checks the totals and lists against the code.
- **Documentation examples run in the tests**, as the README's do: each
  page's Python blocks in order, the pages that train in the slow tier.
  Two examples did not run and are fixed: Environments used an undefined
  environment, and Comparing agents' `estimate` trained 100,000 steps, which
  ends inside a PPO rollout and leaves the last evaluation NaN.
- **Documentation site**: the world models' reference-comparison records
  (protocol, pre-registrations, diagnosis, results) leave the site and stay
  on GitHub, where the site's links to them now point; the design notes
  move under the Developer guide.
- The README says Ajax runs on a CPU or a GPU, and that its GPU speed is
  not yet measured, instead of putting every run on the GPU.
- **Resets are keyed by episode** (PEGASUS, Ng and Jordan 2000), on by
  default: env i's k-th episode starts on a key of (i, k), and a gymnax
  env's step t in it draws on a key of (i, k, t), so runs that differ by an
  option, an extension or their updates start every episode alike. gymnax
  envs get `EpisodeKeyedAutoResetWrapper` under their own wrappers, and
  brax stacks auto-reset as playground's (`FreshAutoResetWrapper`).
  `episode_keys=False` (`build_env_from_id`, or `with_episode_keys(env,
  on=False)`) reproduces earlier runs. Checkpoints are format 3; one from
  before is refused with episode keys on. Every run draws other numbers.
- **An evaluation's keys follow its training phase and timestep**
  (`ajax.log.evaluation_keys`), not its count: the log frequency no longer
  changes an evaluation's episodes, and the online evaluations after
  `train_offline` no longer replay the offline ones' keys. Every logged
  evaluation draws other episodes; training is unchanged.
- **No key is drawn twice.** The observation noise (gymnax and brax),
  InitialStateWrapper's initial state and ModelReferenceWrapper's reference
  draw on keys derived by name from the key the inner env is handed whole,
  so wrapping an env leaves the env's own draws as they were. The expert's
  collection (ExpertPrefill, MCPretrain, CloningPretrain, recorded datasets)
  splits its carry first instead of carrying an env's step key on, and
  TD-MPC2's task embeddings no longer draw on the policy's initialisation
  key. Runs with these draw other numbers.
- **Each extension draws on keys of its own, by name.** In every phase an
  extension's key is derived from its name and its occurrence among the
  stack's extensions of that name (`ajax.extensions.base.extension_key_names`),
  so adding, removing or moving another extension changes none of its keys.
  A pretrain or post_update running the agent's own steps (BellmanPretrain,
  PriorData, PhiRefresh, MCPretrain's rollout) draws on a key lent off the
  agent's, which then gets its own back. The action stage's extensions (EDGE's
  gates, IBRL's choice) and evaluation's choosers draw on keys derived from
  the step's instead of advancing the collection's or the evaluation's
  stream, and IBRL's subset and soft choice draw by name, so either option
  leaves the other's draw. `ActionPipelineResult.rng` is removed: a
  pipeline's key argument is the action stage's. Runs with extensions draw
  other numbers; runs without keep theirs.
- **Options and extensions no longer move the updates' keys.** A key
  that exists only under an option, a count or an extension is derived by
  name or index instead of drawn from the stream: the extensions'
  post_update, an actor that steps on some updates only (policy_delay,
  policy_update_start), several critic or gradient steps per update
  (num_critic_updates, n_gradient_steps), PPO's and APO's minibatches,
  ExpertWarmup's decision and a recurrent network's initial carry. Runs that
  differ by one of them share every other draw, and SAC, REDQ, TQC, ASAC
  and AVG no longer draw a dropped actor step's keys again before
  policy_update_start. The agents using any of these draw other numbers.
- **Updates and collection draw on two streams of keys.** Collection draws
  on its own key, `collector_state.rng`, and the updates on the agent
  state's, each handing out its keys after its carry
  (`ajax.random_keys`): an update drawing more or fewer keys leaves the env
  steps, the action noise and the warm-up as they were, and a forced reset
  draws on a key derived from the collection's. Every agent but APG and the
  world models draws other numbers from the same seed. Checkpoints are
  written in format 2; an older one restores with the collection's key
  derived from the updates'.
- **DreamerV3 is a declared agent**, read from `agents/DreamerV3/DreamerV3.py`
  on a new family, `OwnReplayAgent` (`ajax.agents.families`): each tick its
  row collector stores one row per env in a replay of its own, sized for
  each `train` call (`prepare_replay`), and it runs the updates its
  schedule gives the absolute tick. Its keywords are fixed fields,
  keyword-only after `env_id`; `gamma` is derived, no keyword.
  `agent.dreamer_config` is renamed `agent.model_config`. Deleted, without
  a deprecation path: `train_DreamerV3`, `DreamerV3AgentConfig` and
  `agent.agent_config` (its four fields are DreamerV3's) and
  `agent.train_kwargs`'s entries. The names moved: `TrainRatio`, `EnvSpec`,
  `env_spec` (now `measure_env`) and `DreamerV3State` to `DreamerV3.py`;
  `policy_step` (now `observe_and_act`), `PolicyCarry` and
  `initial_policy_carry` (now `make_initial_carry`) to `world_model.py`;
  `train_metric_keys` (now `learner.list_train_metrics`); `bytes_per_row`
  (now `replay.count_row_bytes`); `MetricsAccumulator` to `ajax.log`. Its
  extension context drops `agent_config` and `total_timesteps`, which no
  extension read. Training is unchanged, bitwise.
- **TD-MPC2 is a declared agent** on the own-replay family, read from
  `agents/TDMPC2/TDMPC2.py`: its keywords are fixed fields, keyword-only
  after `env_id`. The `gamma` and `seed_steps` fields keep the values
  passed (`None`: derived from the episode length); the values used are
  `agent.resolved_gamma` and `agent.resolved_seed_steps` (read
  `agent.gamma` for the derived discount no more). `agent.agent_config` is
  renamed `agent.model_config`, on `TDMPC2MultiTask` too, and
  `TDMPC2Config` no longer subclasses `BaseAgentConfig` (its
  `expose_recent_rollout` is gone). Deleted, without a deprecation path:
  `train_TDMPC2` and `agent.train_kwargs`'s entries. The names moved:
  `Schedule` (now `FixedLengthSchedule`), `EpisodicSchedule` and
  `TDMPC2State` to `TDMPC2.py`; `planner_policy` (now
  `planner.plan_for_envs`); `update_metrics_zeros` (now
  `core.make_zero_update_metrics`); `clip_grad_norm`, `make_world_model_tx`
  and `make_policy_tx` from `core` to `TDMPC2/optim.py`; the ring's creation
  to `buffer.make_episode_buffer`. `agents.TDMPC2` no longer re-exports
  the classes (import them from `ajax` or their modules). Its extensions
  are bound, to an empty context. Training is unchanged, bitwise.
- **UDRL runs the action stage.** `JSRLCurriculum` and EDGE's fixed gate
  hand its expert steps, and its segment keeps the choice executing the
  expert's action (the expert's own, or a residual of 0), which its actor
  fits; `StorePolicyAction` keeps the policy's own instead. The gates draw
  on a fourth split of the step's key, taken only with an action extension.
  Runs without one are unchanged, bitwise.
- **UDRL observes the observation features.** `ExpertActionInput`,
  `ExpertStateInput`, `TrainingProgressInput` and `PIDTermsInput` join each
  observation of its segments, before the command, in collection and
  evaluation; UDRL's collector calls the expert through the shared wrapper
  (the expert first, gain mode last). Runs without features are
  unchanged, bitwise.
- **The world models observe the training progress and PID terms.**
  `TrainingProgressInput` and `PIDTermsInput` join each observation DreamerV3
  and TD-MPC2 collect and evaluate: their encoders read them, and DreamerV3's
  decoder reconstructs them with the env's columns. `TDMPC2MultiTask`, which
  trains on logged observations, refuses them. Runs without features are
  unchanged, bitwise.
- **DQN and PQN take an expert.** `expert_policy=` (an index per row) and
  `eval_expert_policy=`: collection tracks the expert, evaluations log its
  return, and `JSRLCurriculum` and the expert's observation features run on
  both; DQN, which stores the expert's actions, also takes `ExpertPrefill`,
  `ExpertWarmup` and `ExpertValueMetrics` (Q at the stored index). PQN now
  fits Q at the actions its steps executed, the policy's own without an
  action extension. Runs without an expert are unchanged, bitwise.
- **IQL under a residual policy takes JSRL, EDGE and IBRL's acting.** Where
  one hands the agent's expert the step, IQL's replay keeps the residual
  that executes the expert's action, 0, beside the executed action, which
  its actor fits; IBRL acting with another expert is refused (a residual
  need not reach that expert's action). `StorePolicyAction` still stores
  the policy's choice. The action pipelines' result carries the new
  columns (`policy_action_for_expert`, `stored_policy_action`). Runs that
  were accepted before are unchanged, bitwise.
- **APG logs its expert's return.** Each evaluation also rolls out
  `eval_expert_policy`, else the agent's `expert_policy`, alone on the same
  systems, and logs `Eval/episodic mean expert reward` and `Eval/expert
  bias` (NaN without an expert), as every agent does; APG takes
  `eval_expert_policy`, which it refused before. Training is unchanged,
  bitwise; evaluations gain the two keys.
- **APG observes the observation features.** `ExpertActionInput`,
  `ExpertStateInput`, `TrainingProgressInput` and `PIDTermsInput` join the
  observation its controller reads in the differentiable rollout, their
  memory afresh at each episode start; the expert's features keep their
  gradient there (`ObserveInputs.differentiable`), as the expert a residual
  policy acts with does. `closed_loop_rollout` takes `observe` and its
  `Rollout` keeps `policy_observation`. Runs without features are
  unchanged, bitwise.
- **DQN takes StorePolicyAction.** Its replay, on the replay family's
  collector, stores the policy's index where the run picks it; DAgger
  refuses the ablation as missing a capability (`reward_learning`: it
  learns from its expert's labels alone, no stored action). Runs without
  it are unchanged, bitwise.
- **Fixed: EDGE scores the executed action.** Under `ResidualPolicy`,
  `EDGEExploration`'s scoring gates compared the expert's action with the
  policy's residual, while every critic is trained on the executed action
  (the residual through the expert). They now compare it with the action
  the step would execute, as `IBRL`'s acting does. Runs with EDGE and a
  residual policy train differently; other EDGE runs do not change.
- **Gain mode refuses `EDGEExploration`** when the agent is built, as it
  refuses `JSRLCurriculum` and `ValueBox`: EDGE executes the expert's env
  action where the policy's gains go. It ran unnoticed when the expert had
  as many gains as the env has action dimensions, and failed when training
  started otherwise.
- **APG's schedule steps its matrices alone**, as its reference does: with
  `lr_schedule="warmup_cosine"` (`contextual_controller`'s default) the
  biases, LayerNorm scales and PID gains keep `learning_rate`. Scheduled
  APG runs train differently from before. The shared optimiser takes the
  option as `OptimizerConfig.vector_learning_rate`.
- **DQN, DAgger, UDRL and APG are declared agents**, each read from
  `agents/<A>/<A>.py`: keyword-only after `env_id` (DAgger's
  `expert_policy` included), fixed fields, `<A>Config` a view of them.
  DQN and DAgger are on the replay family, UDRL and APG on a new one,
  `OwnIterationAgent` (`ajax.agents.families`): the agent writes its
  iteration, report and evaluation, the loop resumes and logs; a train
  call's options (APG's `reset_optimizer_on_resume`) go to its
  `configure_call`, and `train_offline` refuses it. Their old modules
  are deleted, without a deprecation path (`train_DQN`, `train_DAgger`,
  `train_UDRL`, `train_APG`, each agent's `state`, UDRL's `utils`): the 0.4.0 compatibility layer is removed in this release. Their names are in
  the agent files, some renamed: APG's `make_controller`,
  `make_agent_state`, `compute_rollout_returns` (its `key` and
  `n_systems`), `make_step_through_expert`; UDRL's `scale_command` and
  `compute_actor_loss`; DAgger's `compute_beta` and
  `choose_learner_action`; init, update and evaluation are methods. UDRL
  takes `cnn_extra_obs_dim` and `cnn_spec`; DAgger and UDRL take
  `eval_expert_policy`, which APG refuses (`NotWired`) until its evaluation
  rolls out an expert; APG refuses a bad `lr_schedule` when it is built.
  Training is unchanged, bitwise, but for DAgger's replay, which now
  stores the expert's next action (`next_a_expert`, unread): its pipeline
  reads the expert's action collection computes, as every pipeline does.
- **UDRL logs its eval_expert_policy's return.** UDRL takes
  `eval_expert_policy`, as every agent with an expert does: each evaluation
  logs that expert's return on its episodes (`Eval/episodic mean expert
  reward` and `Eval/expert bias`), the expert taking no other part in the
  run. Runs without one are unchanged, bitwise.
- **The row collector and evaluate_policy observe features.** With
  observation features (`env_args.observers`), the world models' row
  collector appends them to every row, fresh at each episode's start, and
  `evaluate_policy` takes `agent_state` for the frozen training progress, as
  `evaluate` does. Runs without features are unchanged, bitwise.
- **Discrete experts outside DAgger.** A discrete expert's action index, an
  int `(n, 1)` column, now steps the env in the expert's own rollouts (the
  replay prefill, the cloning and Monte-Carlo rollouts), enters
  `ExpertWarmup`'s mix as the policy's index, and enters `ExpertActionInput`
  one-hot (`ExpertInAgentUnits.action_count`): PPO and APO run with one on
  CartPole. Continuous runs are unchanged, bitwise.
- **A skipped actor step logs the last one that ran.** On the updates whose
  actor step is skipped (TD3's and CrossQ's `policy_delay`, any agent's
  `policy_update_start`), the actor's logged metrics are now those of its
  last real step, NaN before its first (`update_steps.run_actor_if`; the
  state keeps them in `last_actor_metrics`). TD3 logged zeros there,
  CrossQ and IQL NaN, and SAC, TQC, REDQ, ASAC and AVG the metrics of the
  step they computed and dropped before `policy_update_start`. Training
  is unchanged; only the logs move. A checkpoint saved without the field
  restores it as NaN. `run_if` lost its `fill` option, which TD3 alone
  used.
- **AVG is a declared agent**, `agents/AVG/AVG.py`, on a new family,
  `IncrementalAgent` (`ajax.agents.families`): one update on each step as
  it comes, no replay. Keyword-only after `env_id`, fixed fields,
  `AVGConfig` a view of them; `train_AVG`, `AVG.state` and `AVG.utils`
  are removed in this release, their names moved
  (`value_loss_function` and `policy_loss_function` are
  `compute_critic_loss` and `compute_policy_loss`; `init_AVG`,
  `update_AVG_values` (`update_step_statistics`), `update_agent`,
  `update_value_functions`, `update_policy` and `make_train` became
  methods). AVG claims the warm-up it always had before a positive
  `learning_starts`, so `ExpertWarmup` reaches it, and it takes
  `policy_update_start` and `eval_expert_policy`. Training is unchanged,
  bitwise.
- **ASAC is a declared agent**, `agents/ASAC/ASAC.py`, as SAC is:
  keyword-only after `env_id`, fixed fields, `ASACConfig` a view of them;
  `train_ASAC`, `ASAC.state` and `ASAC.utils` are removed in this release, their names moved (`value_loss_function` is `compute_critic_loss`;
  `init_ASAC`, `update_agent`, `update_value_functions`, `update_policy`
  and `make_train` became methods). ASAC now takes `policy_update_start`,
  `alpha_update_start`, `eval_expert_policy` and `StorePolicyAction`, and a
  scheduled `target_entropy_per_dim`. A recurrent ASAC given extensions is
  refused by the family's check, as every recurrent agent is, with its
  message. Training is unchanged, bitwise.
- **REDQ is a declared agent**, `agents/REDQ/REDQ.py`, as SAC is:
  keyword-only after `env_id`, fixed fields, `REDQConfig` a view of them
  (no longer a `SACConfig` subclass); `train_REDQ` and `REDQ.state` are removed in this release, their names moved (`value_loss_function` is
  `compute_critic_loss`; `init_REDQ`, `update_agent`,
  `update_value_functions`, `update_policy` and `make_train` became
  methods). REDQ now takes `policy_update_start`, `alpha_update_start`,
  `eval_expert_policy`, `BellmanPretrain` and `StorePolicyAction`, and
  `target_entropy_per_dim` takes a schedule besides a number and
  `"mbpo"`. Training is unchanged, bitwise.
- **TQC is a declared agent**, `agents/TQC/TQC.py`, as SAC is: keyword-only
  after `env_id`, fixed fields, `TQCConfig` a view of them (no longer a
  `SACConfig` subclass); `train_TQC` and `TQC.state` are removed in this release, their names moved (`init_TQC`, `update_agent`,
  `update_value_functions`, `update_policy` and `make_train` became
  methods). TQC now takes `policy_update_start`, `alpha_update_start`,
  `eval_expert_policy`, `BellmanPretrain` and `StorePolicyAction`, and
  a scheduled `target_entropy_per_dim`. Training is unchanged, bitwise.
- **SAC is a declared agent**, `agents/SAC/SAC.py` on the replay family, as
  IQL, CrossQ and TD3 are. What changes for a caller:
  - the constructor is keyword-only after `env_id`; its fields and
    `agent_config` are fixed once built; `SACConfig`, a view of the fields, is removed in this release;
  - `train_kwargs` is empty: read the fields (`agent.policy_update_start`);
    `expert_policy` holds the expert as passed, `agent.expert` the one in
    agent units;
  - SAC's equations are in `SAC.py`, renamed by their verb:
    `compute_soft_bellman_target`, `compute_soft_policy_loss`,
    `compute_soft_policy_objective`, `compute_temperature_loss`,
    `compute_alpha_temperature_loss` (`compute_td_target` and
    `sample_next_actions` keep their names); the temperature is in
    `ajax.agents.temperature`, the action pipeline in
    `ajax.agents.action_pipeline`;
  - the old modules (`train_SAC`, `SAC.state`, `SAC.core`,
    `SAC.action_pipeline`) are removed in this release, their names moved; the functions that became methods (`init_SAC`, `update_agent`,
    `update_value_functions`, `update_policy`, `make_train`) raise
    TypeError with the method's call form; `SAC.core` keeps
    `init_soft_actor_critic`, `soft_actor_step` and the 0.4.0
    `update_temperature`;
  - the deprecated keywords are fields of `SACDeprecatedKeywords`:
    `store_policy_action` builds `StorePolicyAction` where it acted (with
    an expert, outside gain mode), and `target_entropy_initial_per_dim`
    with `target_entropy_ramp_frac` builds the ramp `target_entropy_per_dim`
    now takes as a schedule; both warn.

  Training is unchanged: every final state is bitwise the same, and so is
  the training program's text.
- **StorePolicyAction** (`ajax.extensions.expert`), SAC's
  `store_policy_action` as an extension for every replay agent on the
  family: the replay stores the action the policy chose, not the one
  executed. It acts in the controller alone, through a new controller
  phase, `choose_replay_action`; TD3, CrossQ and IQL take it now, and the SAC family (SAC's `store_policy_action` keyword is removed in this release).
- **A soft actor-critic's target entropy may be a schedule**:
  `target_entropy_per_dim` takes a function of the timestep
  (`ajax.schedule`, as TD3's `exploration_noise`) on every agent with the
  shared entropy temperature (`EntropyTemperature`): CrossQ now, the SAC
  family as each is declared. SAC's `target_entropy_initial_per_dim` and `target_entropy_ramp_frac` are removed in this release: pass `target_entropy_per_dim=linear(...)`.
- **Every agent with an expert takes eval_expert_policy**, SAC's option:
  the expert whose return each evaluation logs, if not the training one
  (`ExpertGuided`). **Every agent on the shared loop reads
  LoggingConfig.sweep**, which only SAC read: a sweep evaluates only in the
  last fifth of training. Runs without either are unchanged.
- **BellmanPretrain runs on TD3**, as on SAC: a replay agent with a
  discounted Q critic and target critics gives it its critic step without
  extensions (`ReplayAgent.extension_values`); REDQ and TQC take it once
  declared. It now requires target critics, so CrossQ and DreamerV3 refuse
  it for that reason, as they refused it before for another.
- **Every replay actor-critic takes policy_update_start**, SAC's knob:
  TD3, CrossQ and IQL keep the actor's steps from that timestep on, and
  CrossQ, as every soft actor-critic, takes alpha_update_start for its
  temperature. Both default to 0, no wait: runs that leave them unset are
  bitwise the same.
- **IQL, CrossQ and TD3 are declared agents.** Each is one file,
  `agents/<AGENT>/<AGENT>.py`, on the replay family of the new
  `ajax.agents.families`: its hyperparameters are dataclass fields, declared
  once with the paper's defaults. What changes for a caller:
  - the constructor is keyword-only after `env_id`;
  - an agent's fields and `agent_config` are fixed once built
    (`dataclasses.replace(agent, gamma=0.9)` builds another), and
    `agent_config` is a view of the fields;
  - `TD3Config`'s `learning_starts` default (100) gives way to TD3's
    (25 000), which TD3 always used;
  - `IQLConfig`, `CrossQConfig` and `TD3Config` were views of the fields and are removed in this release;
  - `train_kwargs` is empty: read the agent's fields and attributes
    (`agent.buffer`, TD3's `agent.pipeline`, CrossQ's read-only
    `agent.alpha_args`); `make_train` takes only the keywords every agent's
    takes (no `buffer`, `expert_policy` or `action_pipeline`);
  - `expert_policy` holds the expert as passed, which the run config logs;
    `agent.expert` is the one read in agent units;
  - the old modules (`train_IQL`, `IQL.state`, `train_CrossQ`,
    `CrossQ.state`, `train_TD3`, `TD3.state`) are removed in this release, their names moved (`train_TD3.update_target_networks` is
    `TD3.update_targets`). Their functions that became methods (the update
    functions, `init_<AGENT>`, `make_train`) take other arguments: calling
    one raises TypeError with the method's call form.

  Training is unchanged: every final state is bitwise the same, and so is
  the compiled program.
- **PPO, APO and PQN are declared agents**, on the on-policy family
  (`OnPolicyAgent` in `ajax.agents.families`): each is one file,
  `agents/<AGENT>/<AGENT>.py`. What changes for a caller, as for IQL,
  CrossQ and TD3:
  - the constructor is keyword-only after `env_id`;
  - an agent's fields and `agent_config` are fixed once built, and
    `agent_config` is a view of the fields;
  - `PPOConfig`, `APOConfig` and `PQNConfig` were views of the fields and are removed in this release;
  - `train_kwargs` is empty: PQN's `epsilon_start`, `epsilon_end`,
    `epsilon_decay_frac` and `td_loss_fn` are its fields; `make_train`
    takes only the keywords every agent's takes;
  - PPO's and APO's `expert_policy` holds the expert as passed, which the
    run config logs; `agent.expert` is the one read in agent units;
  - PPO's `encoder_kernel_init` defaults to `"orthogonal(np.sqrt(2))"`,
    which None still means: `PPO(encoder_kernel_init=None)` trains as
    before but logs None in the run config;
  - the old modules (`train_PPO`, `PPO.state`, `PPO.utils`, `PPO.core`,
    `train_APO`, `APO.state`, `train_PQN`, `PQN.state`, `PQN.utils`) are removed in this release, their names moved.
    `policy_loss_function` and `value_loss_function` keep their
    parameters. The functions that became methods (`init_PPO`,
    `init_APO`, the `update_agent`s, APO's `update_networks`, `make_train`)
    take other arguments: calling one raises TypeError with the method's
    call form.

  Training is unchanged: every final state is bitwise the same, and so is
  the compiled program.
- **`RolloutSplit`.** `ajax.agents.minibatch.Geometry`, how an on-policy
  rollout is cut into minibatches, is now `RolloutSplit`, and
  `minibatch_geometry` is `choose_rollout_split`: same arguments and fields,
  `kind` still `"flat"`, `"time"` or `"recurrent"`. The split also says
  where the advantages are computed (`advantages_per_minibatch`) and cuts
  the rollout (`minibatches`). The old names are removed in this release.
- **Extension phases are applied.** `ExtensionStack.fold_reward`,
  `fold_on_target` and `fold_post_update` are now `apply_reward`,
  `apply_on_target` and `apply_post_update`, `fold_init`,
  `fold_init_states`, `fold_pretrain` and `fold_eval_metrics` are
  `apply_init`, `apply_init_states`, `apply_pretrain` and
  `apply_eval_metrics`, and the two loss terms, `fold_critic_loss` and
  `fold_actor_loss`, are computed: `compute_extensions_critic_loss` and
  `compute_extensions_actor_loss`. So are `ajax.agents.loop`'s
  `fold_on_target`, `fold_critic_loss` and `fold_post_update` (also in
  `ajax.agents.update_steps`): same arguments. The old names are removed in this release. `TrainContext.stack` is
  `TrainContext.extensions`, and so is `TrainLoop.stack`.
- `ajax.agents.minibatch.rollout_actions` (`ajax.agents.PPO.core`'s
  before) is `get_actions_and_log_probs`; the old names are removed in this release.
- `ajax.agents.SAC.SAC.augmentation_features` is removed in this release, with SAC's deprecated keywords.
- `ajax.agents.SAC.expert.collect_and_store_expert_transitions` lives in
  `ajax.extensions.pretrain`, for every replay agent, and writes its rows as
  the live collector does (`policy_action` and `next_action` too); the old path is removed in this release.
- The extension phases' `ExtensionContext` parameter is named `context`
  (was `ctx`). The stack passes it positionally, so an extension that
  still names it `ctx` keeps working.
- Writing a name on a module it moved out of (`monkeypatch.setattr` on an
  old path) raises AttributeError naming its new home, where it is read:
  the write would have changed nothing.
- `checkpoint.py` reads PPO's and APO's restore rules off their state
  classes (`joint_gradient_clip`, `restore_hint`); `PPOState` is a class of
  its own, no longer an alias of `BaseAgentState`.
- `BaseAgentState.get_keys_and_new_state(n=1)` draws `n` keys from the
  training key and returns them, then the new agent state:
  `shuffle_key, agent_state = agent_state.get_keys_and_new_state()`.
  `next_keys` returns the same, the state first, until the agents still
  calling it are converted.
- `ajax.agents.update_steps.compute_gradients(loss, params)` returns the gradients
  of a loss of the parameters alone, and its aux; `gradient_step` and PPO's
  and APO's minibatch steps take their gradients with it.
- `update_steps.joint_step` is now `apply_gradients_clipped_together`, which also
  returns each network's gradient norm before the clip. `joint_step` is removed in this release.
- The declared agents (PPO, APO, PQN, TD3, IQL, CrossQ) and the shared
  modules they use name their parameters and locals in full words: an
  agent's `state` parameter is `agent_state`; the on-policy `update`'s
  `start` (and `compute_values_from_rollout_sequence`') is
  `agent_state_at_rollout_start`; APO's `minibatch_step` takes
  `value_bias` (was `b`); `get_sequence_minibatches` takes `shuffle_key`
  (was `rng`); PPO's `advantages` method is `compute_advantages`. A caller
  passing them by position, as the loop does, sees no change.
- The functions added since 0.4.0 that these modules define are named by
  verbs: `steps.when` is `run_if`, `steps.extension_key` is
  `derive_extensions_key` (its tag `EXTENSIONS` is `EXTENSIONS_KEY_TAG`),
  TD3's and CrossQ's `critic_update` and `actor_update` are `update_critic`
  and `update_actor`, IQL's equations start with `compute_`,
  `RolloutSplit.minibatches` is `cut_into_minibatches`,
  `sequence_minibatches` is `get_sequence_minibatches`, `recurrent`'s
  `rollout_sequence_values` is `compute_values_from_rollout_sequence`,
  its `actor_dist_train` is `ajax.networks.networks`'
  `get_policy_updating_batch_statistics` (beside
  `predict_value_updating_batch_statistics`, was `predict_value_train`),
  and PQN's `epsilon_schedule` method is `make_epsilon_schedule`.
- `recurrent.py` is split into labelled sections, and the machinery that
  inserts the final observation of an episode a time limit cut is named
  for it: `stream_positions` is `row_positions_with_final_observations`
  (the old name is removed in this release), and `RecurrentCarries.positions` is
  `row_positions_with_final_observations`.
- **Verb names.** `ajax.agents.steps` is `ajax.agents.update_steps`. A
  function returning a loss or a target value starts with `compute_`, one
  building a function or a network with `make_`: `critics`'
  `critic_loss_fn`, `q_regression_loss`, `q_loss_fn`, `mse_td_loss` and
  `huber_1_td_loss` are `compute_mean_critic_loss`,
  `compute_q_regression_loss`, `compute_q_loss`, `compute_mse_td_loss` and
  `compute_huber_1_td_loss`; IQL's `expectile_loss` is
  `compute_expectile_loss`; PQN's `half_mse_td_loss` is
  `compute_half_mse_td_loss`; TD3's `critic_loss` and `policy_loss_function`
  are `compute_critic_loss` and `compute_policy_loss`, and
  `gaussian_exploration` is `make_gaussian_exploration`; CrossQ's
  `batch_renorm_mlp` is `make_batch_renorm_mlp`. The 0.4.0 paths
  (`train_DQN.huber_1_td_loss`, `train_DQN.mse_td_loss`,
  `train_DQN.q_loss_fn`, `train_IQL.expectile_loss`,
  `train_PQN.half_mse_td_loss`, `SAC.core.critic_loss_fn`,
  `train_TD3.policy_loss_function`) are removed in this release. A declared agent's hooks that build an object
  start with `make_`: `make_context(total_timesteps)` (was `context`),
  `make_collector` (was `collector`, also the `TrainContext` field),
  `make_replay_buffer` (was `replay_buffer`) and
  `TrainContext.make_actor_critic` (was `actor_critic`); `behaviour` is
  `choose_behaviour`. The hooks that declare settings keep their names.
- **PPO's memory handling lives in `recurrent.py`.** PPO's losses are
  `compute_value_loss` and `compute_policy_loss`: they take the minibatch
  and no `recurrent`, `resets` or `initial_hidden`, and their keywords are
  `critic_state`, `normalize_advantage`, `entropy_key` and
  `compute_extensions_loss` (were `critic_states`,
  `advantage_normalization`, `entropy_rng` and `extra_loss_fn`). The
  networks run through `ajax.agents.recurrent`'s `get_policy_on_minibatch`
  and `predict_values_on_minibatch`, and the values GAE reads come from
  `compute_rollout_values`; each serves a network with or without memory.
  `RolloutSplit.cut_into_minibatches` and `get_sequence_minibatches` take
  what a recurrent split's sequences replay from (the episode starts and
  each network's memory at every sequence's start) as one dict. APO's
  losses take the same names, with the keywords `critic_state` and
  `value_bias` (was `b`). The 0.4.0 forms (`train_PPO.value_loss_function` and `policy_loss_function`, `train_APO.value_loss_function` and `policy_loss_function`) are removed in this release: call `compute_value_loss` and `compute_policy_loss`. `PPO.py` and
  `APO.py` no longer define the old names. Training is unchanged: every
  final state is bitwise the same, and so is the program.

- **ASAC** now does what the authors' code does (`darer/ASAC.py`): a
  termination drops the next value in the critic's target (it bootstrapped
  through it); the termination penalty follows their rule, `p_0` times the
  larger reward of the batch's first two rows; `theta` averages the reward
  without the penalty, at the actions the actor step sampled, which the
  temperature step reads too; the entropy terms are measured from
  `log(1/nA)`; and the policy's log-probability takes stable-baselines3's
  `+1e-6` in the `tanh` correction. Runs with terminations learn other
  values; continuing tasks change less.
- **REDQ** now does what the authors' code does (`watchernyu/REDQ`): its
  first update's critic targets and actor loss read the temperature 0.2,
  which the code's constructor leaves until its first temperature step, and
  the policy's log-probability takes the code's `+1e-6` in the `tanh`
  correction. Later updates read `exp(log_alpha)` as before.
- **AVG** reads its log-std head as a log-variance, as the authors' code
  does (`gauthamvasan/avg`): the policy's standard deviation is
  `exp(out / 2)`, its clip bounding the log-variance. Runs start with a
  different spread and learn otherwise.
- **PQN** now does what purejaxql does: a time limit ends an episode in
  its returns (it bootstrapped there), epsilon is read once per update
  instead of at every step, and the rollout's second-to-last step reads
  the last step's bootstrap, as purejaxql's backward pass does. Its
  reproduction comparisons become version 2.
- **APO** ends an episode at a time limit, its next value dropped, and
  leaves that step out of the loss, as xtma/apo does (`bootstrap_timelimit`):
  it bootstrapped on the final observation and trained on the step. With
  `normalize_advantage`, the advantages use the other steps' mean and
  unbiased standard deviation, floored at `1e-6`.
- **IQL** sums its two critics' squared errors, as the reference does
  (`ikostrikov/implicit_q_learning`, `update_q`): it averaged them, which
  halved the critics' gradient against Adam's eps and `max_grad_norm`.
- **`ExpertWarmup(fraction)`** (`ajax.extensions.expert`): during the
  warm-up the agent's expert takes a share of the steps, uniform actions the
  others, through a new controller phase, `choose_warmup_action` (no state
  and no key in the learner). SAC, TD3, REDQ, ASAC, CrossQ and IQL (with
  `learning_starts` above 0) run it, under residual RL and gain mode too,
  where the expert's steps store the policy action that executes its own
  action. DQN (no expert yet) and TD-MPC2 (its own collector) refuse it as
  not wired yet; TQC and AVG, which have no uniform warm-up, refuse it, as
  does TD3 given its own `action_pipeline`. SAC's `expert_fraction` keyword is removed in this release, and SAC with an expert no longer builds `ExpertWarmup(fraction=0.7)` on its own: pass the extension.
- **`ExpertPrefill(n_steps)`** (`ajax.extensions.pretrain`): the agent's
  expert acts `n_steps` steps in every env before training and its
  transitions go to the replay as collection writes its own, on SAC, REDQ,
  ASAC, TQC, TD3, CrossQ and IQL. The rows count once in the observation
  and reward normalisers, as `PriorData`'s do (one shared function,
  `ajax.agents.loop.count_in_normalisers`), and in no timestep. The
  pretraining now runs the replay's seeders (`ExpertPrefill`,
  `PriorData`) first, in stack order; `train_offline` refuses a prefill
  that would overwrite the dataset's rows. SAC's `expert_buffer_n_steps` keyword is removed in this release, and SAC with an expert no longer builds `ExpertPrefill(n_steps=20_000)` on its own: pass the extension.
- **SAC's expert prefill runs in the pretraining**, as `ExpertPrefill`:
  `init_SAC` and SAC's `make_train` lose it and `expert_buffer_n_steps`.
  It is the same algorithm on other random numbers: the prefill draws the
  extension's key, and a run whose stack held no other learner extension
  now draws one more key per update. Under its old key the prefill's rows,
  statistics and timestep are bitwise the old ones. Under `train_offline`
  its rows now come after the dataset's, and a prefill that would overwrite
  them is refused. Its rows now count in the
  observation and reward normalisers, which they skipped: a run combining
  the prefill with normalisation starts from other statistics.
- **`BellmanPretrain` requires a seeded replay** (`ExpertPrefill` or
  `PriorData` in the stack) instead of the agent's expert: with an expert
  and `expert_buffer_n_steps=0` it ran on an empty replay, where it
  pre-trained on nothing, and now it is refused. Beside `PriorData` it
  runs without an expert.
- **Fixed: gain mode's expert columns.** Under `ExpertGains` the replay's
  `a_expert` and `next_a_expert` took the width of the gains, while they
  hold the expert's env action: an expert with fewer gains than action
  dimensions crashed collection, and one with more had its action copied
  into every column. They now take the env's action width. Gain mode
  refuses an extension that uses an expert's actions where the policy's go
  (`Extension.uses_expert_actions_as_policy_actions`: ImitationLoss cloning
  its expert, OnlineBC, IBRL, LCBGatedBootstrap, CriticBlend, PhiRefresh,
  ExpertValueMetrics, CloningPretrain, MCPretrain): those are env actions,
  the policy's are gains.
- **Fixed: the expert prefill under an action transform.** SAC's prefill
  (`expert_buffer_n_steps`, now `ExpertPrefill`) wrote the expert's env
  action into the action column, which gain mode keeps for the gains. A
  prefill row now stores what
  collection stores when the expert acts as itself, the policy action each
  transform names (`policy_action_for_expert`): the anchor gains (0) under
  `ExpertGains`, and under `ResidualPolicy` a residual of 0 as the policy's
  action (IQL fits those) beside the expert's executed action. SAC runs in
  gain mode with a prefill learn other values. A prefill row ending in a
  termination keeps the reset observation as its next one, as collection
  does (ASAC's differential target continues from it); it kept the final
  one.
- `NetworkConfig.squash_log_det_eps` sets the guard inside the `tanh`
  log-det, `log(1 - tanh(u)^2 + eps)`, for an agent whose reference uses
  one; 0, the exact log-det, by default. `NetworkConfig.log_variance_head`
  reads the log-std head as a log-variance, for the same reason.
- **Development**: pull requests run the fast tier alone: the tests not
  marked slow, each under 128 gradient steps, and the conformance checks.
  The probes and every other training run in the slow tier, daily on main,
  on a labelled pull request and before a release.

### Default changes

- **Runs log to TensorBoard by default.** A `LoggingConfig` that names no
  backends writes each run's TensorBoard events and `config.json` under
  `<folder>/tensorboard/<run id>/` (the working directory by default),
  where it logged to Weights & Biases before. W&B needs the `wandb` extra
  and an account; name it with `backends=(WandbBackend(project, group),)`.
- `use_wandb` and `use_tensorboard` are deprecated: set, they warn with a
  FutureWarning pointing to `backends`, and log as before (W&B on unless
  `use_wandb=False`).
- `uv sync` no longer installs wandb in a checkout: `uv sync --group wandb`
  does. `pip install ajax-rl` never did; `ajax-rl[wandb]` does.

### Removed

- **The 0.4.0 compatibility layer.** A project pinned to 0.4.0 moves to the
  new forms when it upgrades:
  - the modules that only served moved names: each agent's `train_<A>`,
    `state`, `utils` and `core`, DQN's and TD3's `networks`, SAC's `expert`
    and `action_pipeline`. Each name is in `ajax.agents.<A>.<A>` or in the
    shared module it moved to; `init_<A>`, `update_agent` and `make_train`
    are the agent's `init`, `update` and its family's `make_train`. SAC's
    `core.soft_actor_step` and `init_soft_actor_critic` and the 0.4.0 loss
    forms (`train_PPO.value_loss_function`, `policy_loss_function`,
    `train_APO.value_loss_function`) have no replacement: call
    `compute_value_loss` and `compute_policy_loss`;
  - the former names `Geometry`, `minibatch_geometry`, `rollout_actions`,
    `stream_positions`, `joint_step` and the `fold_<phase>` aliases (now
    `RolloutSplit`, `choose_rollout_split`, `get_actions_and_log_probs`,
    `row_positions_with_final_observations`,
    `apply_gradients_clipped_together` and `apply_<phase>`), and the step
    functions `ajax.agents.loop` re-exported (import them from
    `ajax.agents.update_steps`);
  - SAC's keywords `expert_buffer_n_steps`, `expert_fraction`,
    `use_pid_policy`, `use_train_frac`, `augment_obs_with_expert_action`,
    `augment_obs_with_expert_state`, `store_policy_action`,
    `normalize_obs_running` and `target_entropy_initial_per_dim` with
    `target_entropy_ramp_frac`: pass `ExpertPrefill`, `ExpertWarmup`,
    `ExpertGains`, `TrainingProgressInput`, `ExpertActionInput`,
    `ExpertStateInput.for_expert(expert)`, `StorePolicyAction`,
    `normalize_observations=True` and `target_entropy_per_dim=linear(...)`.
    `jsrl_curriculum` and `skip_critic_pretrain` had no effect. SAC with an
    expert no longer adds `ExpertWarmup(fraction=0.7)` and
    `ExpertPrefill(n_steps=20_000)` on its own: pass them to keep them;
  - each declared agent's `<A>Config` view and `agent.agent_config`, and
    the `agent_config` argument of their `make_train`: read the agent's
    fields. DreamerV3 and TD-MPC2 keep their configs;
  - the six-argument action pipeline TD3 adapted (a pipeline takes the
    step's `ActingState` second), `LoggingConfig.horizon`,
    `PIDTermsWrapper` (pass `PIDTermsInput`) and `ajax.deprecation`.

## 0.4.0

The first release since 0.3.1 (April 2025), which held SAC alone. Ajax has
since been reworked; the [documentation](https://yannberthelot.github.io/Ajax/)
describes the result. What a user of 0.3.1 meets:

- **Requirements**: Python 3.12 or newer; JAX 0.10 and flax 0.12.8 or newer,
  with no upper bounds; gymnax 1.0.1 from PyPI instead of a fork. Ajax
  declares only the packages it imports, so gymnasium, mujoco, mujoco-mjx,
  pandas, matplotlib, pydantic and scipy are no longer installed with it.
  pip installs Ajax from PyPI or from its repository.
- **Distribution name**: Ajax is on PyPI as `ajax-rl` (`pip install
  ajax-rl`); the import is unchanged, `import ajax`. A project that declares
  `ajax` from git or a path renames it `ajax-rl`.
- **Breaking**: Brax and MuJoCo Playground are no longer installed with
  Ajax. A project that uses their environments asks for `ajax-rl[playground]`
  (which brings Brax) or `ajax-rl[brax]`; without them, building one of their
  environments raises an error that names the extra.
- **Extras**: `ajax-rl[cuda]` (JAX with CUDA, Linux), `ajax-rl[brax]`,
  `ajax-rl[playground]`, `ajax-rl[compare]`, `ajax-rl[reproduction]` and
  `ajax-rl[wandb]`.
- **Agents**: sixteen, each from a published paper: SAC, ASAC, REDQ, TD3,
  AVG, DQN, PQN, PPO, APO, UDRL, APG, DreamerV3, TDMPC2, TDMPC2MultiTask,
  IQL and DAgger, on one training loop, with several seeds in one compiled
  program. Each starts from its reference code's initialisation and
  optimiser settings where its algorithm page says so. TD3 makes the update
  of sfujim/TD3, its critics reading actions in the environment's units, as
  REDQ's do.
- **Extensions**: research features passed as `extensions=(...)`; an agent
  refuses, when built, one it cannot run.
- **Learning from data**: `train_offline(dataset, n_updates)` trains a replay
  agent (SAC, TD3, REDQ, ASAC, DQN, IQL) on a transition dataset before it
  acts, then `train` fine-tunes it online. On it run TD3+BC
  (`ImitationLoss(None, target="replay", alpha=2.5)`, the paper's actor
  loss), ReBRAC (with `TargetBC`), CQL and Cal-QL (`ConservativeQ`, `CalQL`,
  for a policy with a density) and IQL. GAIL (`AdversarialReward` on PPO)
  and DAgger imitate with no reward, and the expert-guidance extensions
  learn from an existing controller.
- **New modules**: `ajax.compare` (AdaStop and confidence sequences),
  transition datasets (`ajax.buffers.dataset`), checkpoints
  (`ajax.checkpoint`), recurrent memory, and the reference comparisons
  under `benchmarks/reference`.
- **Development**: dependency floors without caps, tested weekly at the
  newest and the oldest versions allowed; `uv.lock` refreshed monthly.
  Conformance checks (`tests/conformance`) compare one update of an agent
  with its original implementation's: PQN, TD3, TD3+BC, REDQ and AVG so
  far. The tests come in
  two tiers (CONTRIBUTING.md, Testing): the slow tier, every test that
  trains, also runs daily on main, on a pull request labelled
  `run-slow-tier` and before each release; tags come only from the Release
  workflow, once the slow tier passed on main's head.

### Default changes

For projects that pin an Ajax commit: the merges since 2026-10-17 that move
default runs. Each pull request says how.

| Agent | What changed | Pull requests |
| --- | --- | --- |
| SAC, TD3, REDQ, ASAC, DQN, IQL, DAgger | the replay also draws each environment's newest row | #210 |
| SAC | Xavier-uniform layers, as softlearning; Adam eps 1e-8 | #193, #197 |
| ASAC | torch's layer init; temperature fixed at 0.2; learning rates 1e-4 and 5e-4, gradient clip 10; Adam eps 1e-8 | #193, #196, #197 |
| REDQ | Xavier-uniform layers; Adam eps 1e-8; critics read actions in the environment's units | #193, #197, #227 |
| TD3 | the 2019 code's settings (256-256, learning rate 3e-4, batch 256, 25,000 warm-up steps); torch's layer init; Adam eps 1e-8; summed critic losses, first actor step at the `policy_delay`-th update, critics in the environment's units | #193, #196, #197, #220, #224 |
| TD3+BC, ReBRAC | `ImitationLoss(alpha=...)` is the paper's actor loss, `λ` times the agent's own plus the cloning term, actions compared in the environment's units | #225 |
| AVG | orthogonal layers; Adam eps 1e-8 | #193, #197 |
| DQN | torch's layer init; centered RMSProp with a Huber loss, as Mnih et al. | #193, #197 |
| PQN | flax's `lecun_normal` layers; RAdam after a gradient clip of 10 | #193, #197 |
| PPO | flat minibatches, GAE once per rollout; ppo2's orthogonal layers; Adam eps 1e-5, one gradient clip over both networks, `vf_coef` 0.5 | #146, #159, #183 |
| PPO with memory | PureJaxRL's layer init; `num_minibatches` sets the sequence length | #170, #188 |
| PPO, APO, UDRL, discrete actions | the logits head starts at orthogonal(0.01) | #147 |
| APO | its official code's actor head, 64-64 tanh layers, joint optimiser step and entropy bonus 0.01 | #192 |
| APG | Adam eps 1e-8; `APG.contextual_controller` does not clip its gradient | #175 |
| twin and ensemble critics, with `max_grad_norm` | each critic clipped on its own | #196 |
