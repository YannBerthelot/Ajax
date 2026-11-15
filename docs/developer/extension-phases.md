# Extension phases

The points of the training loop where an extension acts, what each hands
it, and what an extension declares. The [user guide](../guide/extensions.md)
shows how to use extensions, and [Writing an extension](writing-an-extension.md)
how to add one.

## The phases

An extension overrides one or more *phases*, the points of the training loop
where it may act. Every other phase keeps a default that changes nothing.

| Phase | When it runs | What it may do | Across a stack |
| --- | --- | --- | --- |
| `init_state` | once, on a fresh run | build the extension's state | one state per extension |
| `pretrain` | once, before training, on a fresh run | change the agent's state (clone the actor, pre-train a critic) | in list order, the replay's seeders (`ExpertPrefill`, `PriorData`, `RetainedDemonstrations`, `RLPD`) first |
| `reward` | after each rollout, before the agent's targets | replace the rollout's rewards, and update the extension's own state | in list order, each sees the previous result |
| `on_target` | at each critic update | change the TD or value target, and log metrics | in list order, each sees the previous result |
| `critic_loss` | at each critic update | add a term to the critic loss, and log metrics | summed |
| `actor_loss` | at each actor update | add a term to the actor loss, and log metrics | summed |
| `action` | at each collection step | replace the action the agent would take; after the warm-up's choice it may also write `replay_action`, the action the replay stores (`CHEQ`'s RL action) | the last proposal wins |
| `eval_action` | at each evaluation step, on the agents evaluating through the shared loop | replace the executed action | the last proposal wins |
| `transform_action` | at each collection step, before the action is chosen, in the losses and at evaluation | make the executed action a function of the policy's and the expert's; the replay stores the executed one, which the critics score | one per agent |
| `actuate` | at each collection step, after the action is chosen, and at evaluation | execute the chosen action through the expert (gain mode); the replay stores the chosen one | one per agent |
| `choose_warmup_action` | at each warm-up step (before `learning_starts`) | choose the warm-up's actions instead of the uniform ones, and say whether the expert took them (the replay flags those rows) | one per agent; the agents whose uniform warm-up takes one (<!-- agents: warmup-action -->SAC, ASAC, RVI-SAC, REDQ, CrossQ, TD3, IQL, DQN, AVG<!-- /agents -->) |
| `choose_replay_action` | at each collection step, as the replay stores the step | choose the action the replay row stores instead of the executed one | one per agent; the [replay agents](../algorithms/index.md#agent-families) but DAgger, and UDRL (<!-- agents: replay-action -->SAC, ASAC, RVI-SAC, REDQ, CrossQ, TQC, TD3, IQL, DQN, UDRL<!-- /agents -->) |
| `post_update` | after each update | change the agent's state (schedules, refreshes) | in list order |
| `eval_metrics` | at each evaluation | add metrics to the log | merged |
| `observe` | whenever an observation arrives, in collection and evaluation | append columns to it, stored with it (an observation feature) | in column order: the normalised ones, then the raw ones |

The agent hands its stack (an `ExtensionStack`) to its training step, which
makes one call per phase: `apply_init` (every initial state, then every
`pretrain`), `apply_on_target`, `compute_extensions_critic_loss`,
`compute_extensions_actor_loss`, `apply_post_update` and `apply_eval_metrics`.
Each call gets a random key of its own, on which the agent draws nothing else
(`init_state` and `eval_metrics` split it, one sub-key per extension). On an
empty stack a fold returns its input untouched and draws nothing. The `action`,
`transform_action` and `actuate` phases are the exceptions: the agent's
controller (its action pipeline, the shared collector and evaluation) calls
them directly (see [below](../guide/extensions.md#which-agent-runs-which-phase)).

## What a phase hands an extension

`on_target`, `critic_loss` and `actor_loss` hand the extension a `batch`
dictionary of the quantities at that point of the update. Every agent folding
a phase passes the keys of the first column; each agent lists the rest in its
`phase_batch_keys` (`ajax.SAC.phase_batch_keys`, for instance). The replay
actor-critics (SAC, TD3, REDQ, CrossQ, TQC, ASAC, AVG, IQL) share most of
theirs:

| Phase | Every agent folding it | The replay actor-critics add |
| --- | --- | --- |
| `on_target` | `observations`, `next_observations`, `rewards`, `gamma` | `actions`, `dones`, `reward_scale`, `raw_next_observations`, `next_a_expert`, `next_action` (None where nothing stores them), `next_actions`, `critic_subset` |
| `critic_loss` | `observations`, `targets` | `actions`, `next_observations`, `critic_params`, `critic_state`, `actor_state`, `return_to_go` |
| `actor_loss` | `actor_params` | `actor_state`, `observations`, `raw_observations`, `pi_mean`, `agent_loss`, `critic_state`; `actions` and `a_expert` on most (None without an `expert_policy`) |

PPO and APO add `terminated`, `truncated`, `gae` and `values` to `on_target`;
DQN and PQN add `actions`, `dones`, `reward_scale` and their `q_state`; APG's
`actor_loss` holds its `rollout` and `returns`. `pi_mean` is None for a
discrete policy.

`gamma` is `None` for an average-reward agent. `dones` marks where the
target cut the bootstrap. `targets` is the target, held constant. The critic
(`critic_params`, or DQN's and PQN's `q_state`) and the actor
(`actor_params`) are the parameters being differentiated. `pi_mean` is the
actor's mean action. `raw_observations` are the environment's own
observations, before any normalisation; a replay agent keeps them only in a
run with an expert, and passes `None` otherwise, except UDRL, whose segments
hold them anyway. `next_actions` are the next actions the target scored
(None for a target on a state value, IQL's V(s')). `critic_subset` holds
the indices of the target critics whose minimum the target took, where it
took a random subset of them (REDQ's), and None where it took all of them.
An agent's own keys are in its `phase_batch_keys`.

An extension never falls back to adding nothing, so a misplaced extension
cannot look like a result. One declaring the keys it reads (`batch_keys`) is
refused when the agent is built if the batch lacks them: `ImitationLoss` on
PPO, whose actor batch has no `pi_mean`. One reading a key it does not
declare raises when the program is built.

## What an extension declares

Besides its phases, an extension states what it needs as class attributes,
which the agent reads when it is built:

| Attribute | What it says |
| --- | --- |
| `buffer_fields` | replay fields beyond the agent's own that the extension reads. Today only `raw_next_obs`, the environment's own next observation; SAC, TD3, REDQ, ASAC and AVG store it when an extension declares it, and the other agents refuse such an extension. |
| `expert_fields` | the fields holding an expert. The agent converts each one to [agent units](../guide/expert-guidance.md#the-expert), and a run whose extensions hold an expert keeps the replay fields only experts read. |
| `experts()` | the experts the extension calls itself, without a state, on the observations it is given. |
| `reads_agent_expert_actions` | the extension reads the expert's actions the agent stored at collection when its expert is the agent's own `expert_policy` (SAC, TD3, REDQ, ASAC, AVG and APO store them). |
| `uses_expert_actions_as_policy_actions()` | the extension puts an expert's actions where the policy's go: as the actor's target, or as an action its critics score. Gain mode refuses it, since there the policy chooses gains. |
| `requires` | the capabilities it needs beyond those its phases imply (`needs()` returns them all; an extension whose needs depend on its options overrides it). |
| `provides` | the capabilities it gives the rest of its stack: `MCPretrain` provides `phi_star`. |
| `context` | the agent-context keys it cannot work without (`bind_to_agent`); an agent's `extension_context` lists those it binds. |
| `pretrain_context` | the `ExtensionContext` fields its `pretrain` reads (`update`, `buffer`: `PriorData`); an agent's `pretrain_context` lists those its loop fills. |
| `batch_keys` | per phase, the batch keys it reads beyond those every agent folding the phase passes (`reads()` returns them). |
| `reads_step_in_episode` | the extension reads each copy's step within its episode, which the agent then counts. |
| `reads_last_rollout` | the extension reads the agent's latest rollout (`state.last_rollout`), which an on-policy or incremental agent then keeps. |
| `action_slot` | where in SAC's action pipeline its `action` runs. |

`φ*` is a separate critic fitted to an expert's returns before training.
`MCPretrain` fits it (it provides `phi_star`); `OnlineBC`, `CriticBlend`,
`MCVarianceCorrection`, `ValueBox` and `PhiRefresh` read it (they require
`phi_star`). Without `MCPretrain` in the stack they would add nothing for the
whole run, or fail when the program is built, so the agent refuses them. `EDGEExploration` scores with `φ*` when it exists and with the
agent's own critic otherwise, so it needs no `MCPretrain`.
