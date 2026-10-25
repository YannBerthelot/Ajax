# Changelog

Versions follow semantic versioning; before 1.0, a breaking change bumps
the minor version. Each release is tagged `vX.Y.Z` on main and published to
PyPI as `ajax-rl`.

## Unreleased

- **ASAC is a declared agent**, `agents/ASAC/ASAC.py`, as SAC is:
  keyword-only after `env_id`, fixed fields, `ASACConfig` a view of them;
  `train_ASAC`, `ASAC.state` and `ASAC.utils` import their names with a
  FutureWarning until 0.5.0 (`value_loss_function` is `compute_critic_loss`;
  `init_ASAC`, `update_agent`, `update_value_functions`, `update_policy`
  and `make_train` became methods). ASAC now takes `policy_update_start`,
  `alpha_update_start`, `eval_expert_policy` and `StorePolicyAction`, and a
  scheduled `target_entropy_per_dim`. A recurrent ASAC given extensions is
  refused by the family's check, as every recurrent agent is, with its
  message. Training is unchanged, bitwise.
- **REDQ is a declared agent**, `agents/REDQ/REDQ.py`, as SAC is:
  keyword-only after `env_id`, fixed fields, `REDQConfig` a view of them
  (no longer a `SACConfig` subclass); `train_REDQ` and `REDQ.state` import
  their names with a FutureWarning until 0.5.0 (`value_loss_function` is
  `compute_critic_loss`; `init_REDQ`, `update_agent`,
  `update_value_functions`, `update_policy` and `make_train` became
  methods). REDQ now takes `policy_update_start`, `alpha_update_start`,
  `eval_expert_policy`, `BellmanPretrain` and `StorePolicyAction`, and
  `target_entropy_per_dim` takes a schedule besides a number and
  `"mbpo"`. Training is unchanged, bitwise.
- **TQC is a declared agent**, `agents/TQC/TQC.py`, as SAC is: keyword-only
  after `env_id`, fixed fields, `TQCConfig` a view of them (no longer a
  `SACConfig` subclass); `train_TQC` and `TQC.state` import their names
  with a FutureWarning until 0.5.0 (`init_TQC`, `update_agent`,
  `update_value_functions`, `update_policy` and `make_train` became
  methods). TQC now takes `policy_update_start`, `alpha_update_start`,
  `eval_expert_policy`, `BellmanPretrain` and `StorePolicyAction`, and
  a scheduled `target_entropy_per_dim`. Training is unchanged, bitwise.
- **SAC is a declared agent**, `agents/SAC/SAC.py` on the replay family, as
  IQL, CrossQ and TD3 are. What changes for a caller:
  - the constructor is keyword-only after `env_id`; its fields and
    `agent_config` are fixed once built; `SACConfig` is a view of the
    fields, untyped (`Any`) to mypy until 0.5.0;
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
    `SAC.action_pipeline`) import their names with a FutureWarning until
    0.5.0; the functions that became methods (`init_SAC`, `update_agent`,
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
  phase, `choose_replay_action`; TD3, CrossQ and IQL take it now, the SAC
  family once declared, where SAC's keyword builds it until 0.5.0.
- **A soft actor-critic's target entropy may be a schedule**:
  `target_entropy_per_dim` takes a function of the timestep
  (`ajax.schedule`, as TD3's `exploration_noise`) on every agent with the
  shared entropy temperature (`EntropyTemperature`): CrossQ now, the SAC
  family as each is declared. SAC's `target_entropy_initial_per_dim` and
  `target_entropy_ramp_frac` build that ramp until 0.5.0.
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
  - `IQLConfig`, `CrossQConfig` and `TD3Config` are views that subclass
    `BaseAgentConfig` only (`TD3Config` is no longer a
    `RecurrentReplayConfig`), and are untyped (`Any`) to mypy until 0.5.0;
  - `train_kwargs` is empty: read the agent's fields and attributes
    (`agent.buffer`, TD3's `agent.pipeline`, CrossQ's read-only
    `agent.alpha_args`); `make_train` takes only the keywords every agent's
    takes (no `buffer`, `expert_policy` or `action_pipeline`);
  - `expert_policy` holds the expert as passed, which the run config logs;
    `agent.expert` is the one read in agent units;
  - the old modules (`train_IQL`, `IQL.state`, `train_CrossQ`,
    `CrossQ.state`, `train_TD3`, `TD3.state`) still import their names, with
    a FutureWarning, until 0.5.0 (`train_TD3.update_target_networks` is
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
  - `PPOConfig`, `APOConfig` and `PQNConfig` are views that subclass
    `BaseAgentConfig` only, and are untyped (`Any`) to mypy until 0.5.0;
  - `train_kwargs` is empty: PQN's `epsilon_start`, `epsilon_end`,
    `epsilon_decay_frac` and `td_loss_fn` are its fields; `make_train`
    takes only the keywords every agent's takes;
  - PPO's and APO's `expert_policy` holds the expert as passed, which the
    run config logs; `agent.expert` is the one read in agent units;
  - PPO's `encoder_kernel_init` defaults to `"orthogonal(np.sqrt(2))"`,
    which None still means: `PPO(encoder_kernel_init=None)` trains as
    before but logs None in the run config;
  - the old modules (`train_PPO`, `PPO.state`, `PPO.utils`, `PPO.core`,
    `train_APO`, `APO.state`, `train_PQN`, `PQN.state`, `PQN.utils`) still
    import their names, with a FutureWarning, until 0.5.0.
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
  the rollout (`minibatches`). The old names, also on `train_PPO`, import
  with a FutureWarning until 0.5.0.
- **Extension phases are applied.** `ExtensionStack.fold_reward`,
  `fold_on_target` and `fold_post_update` are now `apply_reward`,
  `apply_on_target` and `apply_post_update`, `fold_init`,
  `fold_init_states`, `fold_pretrain` and `fold_eval_metrics` are
  `apply_init`, `apply_init_states`, `apply_pretrain` and
  `apply_eval_metrics`, and the two loss terms, `fold_critic_loss` and
  `fold_actor_loss`, are computed: `compute_extensions_critic_loss` and
  `compute_extensions_actor_loss`. So are `ajax.agents.loop`'s
  `fold_on_target`, `fold_critic_loss` and `fold_post_update` (also in
  `ajax.agents.update_steps`): same arguments. The old names stay as
  aliases until 0.5.0. `TrainContext.stack` is
  `TrainContext.extensions`, and so is `TrainLoop.stack`.
- `ajax.agents.minibatch.rollout_actions` (`ajax.agents.PPO.core`'s
  before) is `get_actions_and_log_probs`; the old names import with a
  FutureWarning until 0.5.0.
- `ajax.agents.SAC.SAC.augmentation_features` lives in
  `ajax.agents.SAC.deprecated_keywords`, with SAC's other deprecated
  keywords; the old path imports with a FutureWarning until 0.5.0.
- `ajax.agents.SAC.expert.collect_and_store_expert_transitions` lives in
  `ajax.extensions.pretrain`, for every replay agent, and writes its rows as
  the live collector does (`policy_action` and `next_action` too); the old
  path imports with a FutureWarning until 0.5.0.
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
  returns each network's gradient norm before the clip. `joint_step` (also
  `train_PPO.joint_step`) returns the two networks alone, as before, until
  0.5.0.
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
  (the old name stays until 0.5.0), and `RecurrentCarries.positions` is
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
  `train_TD3.policy_loss_function`) serve the renamed functions with a
  FutureWarning until 0.5.0. A declared agent's hooks that build an object
  start with `make_`: `make_context(total_timesteps)` (was `context`),
  `make_collector` (was `collector`, also the `TrainContext` field),
  `make_replay_buffer` (was `replay_buffer`) and
  `TrainContext.make_actor_critic` (was `actor_critic`); `behaviour` is
  `choose_behaviour`. The hooks that declare settings keep their names.

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
  does TD3 given its own `action_pipeline`. **Deprecated**: SAC's `expert_fraction` builds it and warns. Its
  default is now `None`, which with an expert and warm-up steps still builds
  `ExpertWarmup(fraction=0.7)`, with a warning that this default will go;
  passing both raises. Runs are unchanged.
- **`ExpertPrefill(n_steps)`** (`ajax.extensions.pretrain`): the agent's
  expert acts `n_steps` steps in every env before training and its
  transitions go to the replay as collection writes its own, on SAC, REDQ,
  ASAC, TQC, TD3, CrossQ and IQL. The rows count once in the observation
  and reward normalisers, as `PriorData`'s do (one shared function,
  `ajax.agents.loop.count_in_normalisers`), and in no timestep. The
  pretraining now runs the replay's seeders (`ExpertPrefill`,
  `PriorData`) first, in stack order; `train_offline` refuses a prefill
  that would overwrite the dataset's rows. **Deprecated**: SAC's
  `expert_buffer_n_steps` builds it, after the other extensions, and warns.
  Its default is now `None`, which with an expert still builds
  `ExpertPrefill(n_steps=20_000)`, with a warning that this default will
  go; the keyword (even 0) next to the extension raises.
- **SAC's expert prefill runs in the pretraining**, as `ExpertPrefill`:
  `init_SAC` and SAC's `make_train` lose it and `expert_buffer_n_steps`.
  It is the same algorithm on other random numbers: the prefill draws the
  extension's key, and a run whose stack held no other learner extension
  now draws one more key per update. Under its old key the prefill's rows,
  statistics and timestep are bitwise the old ones. Under `train_offline`
  its rows now come after the dataset's, and a prefill that would overwrite
  them is refused. A run's config records `expert_buffer_n_steps` as passed
  (`None` by default, where it recorded 20000). Its rows now count in the
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
