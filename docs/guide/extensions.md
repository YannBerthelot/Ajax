# Extensions

An extension adds a research feature to an agent's training without changing
the agent: an extra term in a loss, a change to the TD target, another way to
pick actions while collecting, an expert to learn from, a pre-training step,
or measurements logged at each evaluation. Ajax ships twenty-four of them,
listed in the [catalogue](#catalogue) below.

Why extensions instead of agent options? An agent's constructor takes its
paper's hyperparameters plus `extensions=()`. The exceptions are
`expert_policy` (SAC, TD3, REDQ, ASAC, AVG, APO), whose return each evaluation
logs, and a few SAC options (its expert's prefill and warm-up mix, the
observation augmentations, the PID gain mode, `use_train_frac`,
`store_policy_action`), kept because they change
the network's input, the collection or the action pipeline, which
extensions do not reach yet ([SAC](../algorithms/sac.md#expert-options)). A
feature written into
an agent grows that agent's file and ties it to one experiment; SAC once had
about a hundred constructor keywords that way. Kept apart, a feature works
with every agent that runs the parts of training it touches, and each
agent's code stays readable next to its paper.

## Using extensions

Pass a tuple (or list) of extension objects as `extensions=`. Each object
holds its own settings:

```python
from ajax import PPO
from ajax.extensions.instrumentation import BiasVorePenalty, ConditioningMetrics
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO(
    "CartPole-v1",
    expose_recent_rollout=True,  # keep the latest rollout for ConditioningMetrics
    extensions=(
        BiasVorePenalty(alpha=0.3),  # an extra critic-loss term
        ConditioningMetrics(),  # measurements at each evaluation
    ),
)
logging = LoggingConfig(config={}, log_frequency=8_192, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=16_384, logging_config=logging)
print(evaluations["Cond/critic_srank"])  # one row per seed, one column per evaluation
```

A few rules follow from how Ajax compiles training:

- **Extensions are fixed for a run.** They are compiled into the training
  program, so changing a setting means a new agent and a new compilation.
  Two extensions with equal settings compare and hash equal, as static
  arguments must. A second agent built the same way traces the same program
  again, and the [compile cache](performance.md#the-compile-cache) loads it
  instead of compiling it, unless the run logs.
- **What changes during training lives in the agent's state**, in
  `state.ext_state` (one entry per extension). It is saved in checkpoints and
  carried over when you resume a run.
- **One-shot work runs once.** An extension's initial state and its
  `pretrain` step run on a fresh run only, not when a run resumes from
  `initial_state=`.
- **Measurements need evaluations.** Extensions that add metrics run when the
  agent evaluates, which needs a `logging_config`. Their keys then appear in
  the returned evaluations next to `Eval/episodic mean reward` (see
  [Logging](logging.md)).
- **Order matters only where it can.** Target changes apply one after the
  other, in list order. Loss terms add up, so their order does not matter.
  For collection actions, the last extension that proposes one wins.

An empty tuple costs nothing: the agent's program is the same as without the
argument.

## The phases

An extension overrides one or more *phases*, the points of the training loop
where it may act. Every other phase keeps a default that changes nothing.

| Phase | When it runs | What it may do | Across a stack |
| --- | --- | --- | --- |
| `init_state` | once, on a fresh run | build the extension's state | one state per extension |
| `pretrain` | once, before training, on a fresh run | change the agent's state (clone the actor, pre-train a critic) | in list order |
| `reward` | after each rollout, before the agent's targets | replace the rollout's rewards, and update the extension's own state | in list order, each sees the previous result |
| `on_target` | at each critic update | change the TD or value target, and log metrics | in list order, each sees the previous result |
| `critic_loss` | at each critic update | add a term to the critic loss, and log metrics | summed |
| `actor_loss` | at each actor update | add a term to the actor loss, and log metrics | summed |
| `action` | at each collection step | replace the action the agent would take | the last proposal wins |
| `eval_action` | at evaluation | replace the evaluation action | no agent runs it yet |
| `transform_action` | at each collection step, in the losses and at evaluation | make the executed action a function of the policy's and the expert's | the first one acts |
| `post_update` | after each update | change the agent's state (schedules, refreshes) | in list order |
| `eval_metrics` | at each evaluation | add metrics to the log | merged |

The agent hands its stack (an `ExtensionStack`) to its training step, which
makes one call per phase: `fold_init` (every initial state, then every
`pretrain`), `fold_on_target`, `fold_critic_loss`, `fold_actor_loss`,
`fold_post_update` and `fold_eval_metrics`. Each call gets a random key of
its own, on which the agent draws nothing else (`init_state` and
`eval_metrics` split it, one sub-key per extension). On an empty stack a
fold returns its input untouched and draws nothing. The `action` and
`transform_action` phases are the exceptions: SAC's action pipeline calls
them directly (see [below](#which-agent-runs-which-phase)).

## Which agent runs which phase

Every agent runs `pretrain`, `post_update` and `eval_metrics`, which belong to
the shared training loop. The others depend on what the agent's update has: a
target, a critic, an actor, a collection step it lets an extension change.

| Agent | `on_target` | `critic_loss` | `actor_loss` | `action`, `transform_action` |
| --- | --- | --- | --- | --- |
| [SAC](../algorithms/sac.md) | yes | yes | yes | yes, with an `expert_policy` |
| [TD3](../algorithms/td3.md), [REDQ](../algorithms/redq.md), [ASAC](../algorithms/asac.md), [AVG](../algorithms/avg.md), [IQL](../algorithms/iql.md) | yes | yes | yes | no |
| [PPO](../algorithms/ppo.md), [APO](../algorithms/apo.md) | yes (value targets) | yes | yes | no |
| [DQN](../algorithms/dqn.md), [PQN](../algorithms/pqn.md) | yes | yes (the Q-network's loss) | no | no |
| [UDRL](../algorithms/udrl.md), [DAgger](../algorithms/dagger.md), [APG](../algorithms/apg.md) | no | no | yes | no |
| [DreamerV3](../algorithms/dreamerv3.md), [TDMPC2, TDMPC2MultiTask](../algorithms/tdmpc2.md) | no | no | no | no |

## What an agent has, and what an extension needs

Each agent says what its algorithm has, in `capabilities`:

| Capability | Meaning | Agents |
| --- | --- | --- |
| `actor` | a policy network trained by gradient | all but DQN, PQN |
| `q_critic` | a learned action value Q(s, a) | SAC, TD3, REDQ, ASAC, AVG, DQN, PQN, TDMPC2, IQL |
| `v_critic` | a learned state value V(s) | PPO, APO, DreamerV3, IQL |
| `discounted` | it maximises a discounted return | all but ASAC, APO, UDRL, DAgger, APG |
| `replay` | its updates sample stored past transitions | SAC, TD3, REDQ, ASAC, DQN, UDRL, DAgger, DreamerV3, TDMPC2, IQL |
| `off_policy` | its update stays correct on actions its policy did not choose | all but PPO, APO, APG |
| `critic_ensemble` | two critics or more | SAC, TD3, REDQ, ASAC, TDMPC2, IQL; AVG with `num_critics` of 2 or more |
| `obs_networks` | its policy and value networks read the observation, not a world model's latent state (memory needs it) | all but DreamerV3, TDMPC2 |

A run adds what its configuration gives: `critic` (a Q or V critic),
`continuous_actions` or `discrete_actions` (the environment), `agent_expert`
(the agent has an `expert_policy`), and `phi_star` when an extension of the
stack fits it (`MCPretrain`).

An extension needs what its phases imply (`on_target` and `critic_loss` a
critic, `actor_loss` an actor, `action` an off-policy update, since the
executed action is then not the policy's, `transform_action` continuous
actions) and what it declares in `requires`. An extension needing nothing runs on every
agent. When the agent is built, an extension needing something its run lacks
is refused with `MissingCapability`, which names it:

```python
import jax.numpy as jnp

from ajax import DQN
from ajax.extensions.expert import ImitationLoss


def expert(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


try:
    DQN("CartPole-v1", extensions=(ImitationLoss(expert),))
except ValueError as error:
    print(error)  # DQN cannot run ImitationLoss ('imitation_loss'), which needs actor (...), continuous_actions (...).
```

When the run has everything but the agent's code does not pass it on yet (it
does not fold the phase, bind the extension's context, put a batch key in
the phase's batch or store a replay field), the refusal is `NotWired`, which
says what is missing.

[Compatibility](compatibility.md) shows, for every agent and module, which
runs, which the algorithm cannot run, and which is not wired yet.

Two more limits:

- **`action` and `transform_action` are SAC's alone**, and only with an
  `expert_policy` and without `use_pid_policy`. SAC's collection step offers two places for an
  extension to act, before and after its warm-up choice, and an extension
  says which one in its `action_slot` (`"pre_warmup"` or `"post_warmup"`).
  [Expert guidance](expert-guidance.md) explains them.
- **Memory and extensions do not mix yet.** With `memory=`, PPO, SAC, TD3,
  REDQ and ASAC raise when training starts if you also pass extensions. APG
  takes both.

## What a phase hands an extension

`on_target`, `critic_loss` and `actor_loss` hand the extension a `batch`
dictionary of the quantities at that point of the update. The keys an agent
always passes are listed, per phase, in
`ajax.extensions.base.PHASE_BATCH_KEYS`:

| Phase | Every agent folding it | Some agents add |
| --- | --- | --- |
| `on_target` | `observations`, `next_observations`, `rewards`, `gamma` | `actions`, `dones`, `reward_scale` (SAC, TD3, REDQ, ASAC, AVG, DQN, PQN); `raw_next_observations`, `next_a_expert` (SAC, TD3, REDQ, ASAC, AVG; None where nothing stores them); `terminated`, `truncated` (PQN, PPO, APO); `gae`, `values` (PPO, APO) |
| `critic_loss` | `observations`, `targets` | `critic_params`, `critic_state` (SAC, TD3, REDQ, ASAC, AVG, PPO, APO); `actions` (SAC, TD3, REDQ, ASAC, AVG, DQN, PQN); `q_state` (DQN, PQN) |
| `actor_loss` | `actor_params`, `actor_state`, `observations` | `raw_observations`, `pi_mean` (SAC, TD3, REDQ, ASAC, AVG, PPO, APO, APG, UDRL, DAgger; `pi_mean` None for a discrete policy; UDRL only with extensions); `a_expert` (SAC, TD3, REDQ, ASAC, AVG, APO, DAgger; None without an `expert_policy`); `critic_state` (SAC, TD3, REDQ, ASAC, AVG); `actions` (UDRL); `rollout`, `returns` (APG) |

`gamma` is `None` for an average-reward agent. `dones` marks where the
target cut the bootstrap. `targets` is the target, held constant. The critic
(`critic_params`, or DQN's and PQN's `q_state`) and the actor
(`actor_params`) are the parameters being differentiated. `pi_mean` is the
actor's mean action. `raw_observations` are the environment's own
observations, before any normalisation; a replay agent keeps them only in a
run with an expert, and passes `None` otherwise, except UDRL, whose segments
hold them anyway. An agent may add keys of
its own: read its fold call.

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
| `expert_fields` | the fields holding an expert. The agent converts each one to [agent units](expert-guidance.md#the-expert), and a run whose extensions hold an expert keeps the replay fields only experts read. |
| `experts()` | the experts the extension calls itself, without a state, on the observations it is given. |
| `reads_agent_expert_actions` | the extension reads the expert's actions the agent stored at collection when its expert is the agent's own `expert_policy` (SAC, TD3, REDQ, ASAC, AVG and APO store them). |
| `requires` | the capabilities it needs beyond those its phases imply (`needs()` returns them all; an extension whose needs depend on its options overrides it). |
| `provides` | the capabilities it gives the rest of its stack: `MCPretrain` provides `phi_star`. |
| `context` | the agent-context keys it cannot work without (`bind_to_agent`); an agent's `extension_context` lists those it binds. |
| `pretrain_context` | the `ExtensionContext` fields its `pretrain` reads (`update`, `buffer`: `PriorData`); an agent's `pretrain_context` lists those its loop fills. |
| `batch_keys` | per phase, the batch keys it reads beyond those every agent folding the phase passes (`reads()` returns them). |
| `reads_step_in_episode` | the extension reads each copy's step within its episode, which the agent then counts. |
| `action_slot` | where in SAC's action pipeline its `action` runs. |

`φ*` is a separate critic fitted to an expert's returns before training.
`MCPretrain` fits it (it provides `phi_star`); `OnlineBC`, `CriticBlend`,
`MCVarianceCorrection`, `ValueBox` and `PhiRefresh` read it (they require
`phi_star`). Without `MCPretrain` in the stack they would add nothing for the
whole run, or fail when the program is built, so the agent refuses them. `EDGEExploration` scores with `φ*` when it exists and with the
agent's own critic otherwise, so it needs no `MCPretrain`.

## What is checked, and when

When the agent is built, its stack is refused (`ValueError`, naming each
offending extension) if an extension:

- needs a capability the run lacks (`MissingCapability`): `φ*` without an
  `MCPretrain` in the stack, the agent's `expert_policy` on an agent without
  one, a Q critic on PPO;
- implements a phase the agent does not run, needs context the agent does
  not bind, reads a batch key the phase's batch lacks or a replay field the
  agent does not store (`buffer_fields`) (`NotWired`);
- calls a stateful expert without its state. An extension calls its expert
  on the observations it is given, where no expert state exists. A stateful
  expert is allowed only when it is the agent's own `expert_policy`, on an
  agent that stores its actions from the expert's tracked state, and the
  extension reads those actions;
- (SAC) acts in collection without an `action_slot`, or acts in collection
  or makes the policy residual in a run without an `expert_policy` or with
  `use_pid_policy`.

When training starts, `CloningPretrain` raises when placed after another
pre-training extension, and `PriorData` on a replay whose rows it cannot
fill.

## Catalogue

The checks above are necessary but not always sufficient: the last column
says where each extension does its job, and
[Compatibility](compatibility.md), generated from the code, where the agent
accepts it. "SAC with an expert" means `SAC(..., expert_policy=...)`.

**Learning from an expert** (`ajax.extensions.expert`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OnlineBC` | Pulls the actor's mean action towards the expert's, weighted by `φ*`'s value at each state, fading to zero over the first `critic_warmup_frac` of training. Logs `online_bc/term` and its weight `online_bc/weight`. | `actor_loss` | SAC with an expert and `MCPretrain` |
| `ImitationLoss` | Adds `coef · mean((μ(s) − a_E(s))²) / 4`, the behaviour-cloning term of TD3+BC, towards a fixed expert it holds. When that expert is the agent's `expert_policy`, it reads the actions the agent stored, so the expert may be stateful. With `anneal_frac`, its weight falls linearly to 0 at that fraction of training (kickstarting). With `q_filter=True` a row counts only where the critic rates the expert's action above the policy's (Q-filtered BC), and evaluations log the share of a replay batch it keeps (`imitation/q_filter_frac`; not on AVG, which keeps no replay). With `target="replay"` it clones the replayed rows' own actions, so logged data needs no expert, and with `alpha` it takes TD3+BC's weight, `mean|Q_1(s, μ(s))| / alpha` with no `/ 4` (see [TD3+BC](../algorithms/td3.md#td3bc-offline)). Logs `imitation/term` and the squared distance before the filter and the schedule, `imitation/mse`. | `actor_loss`, `eval_metrics` | SAC, TD3, REDQ, ASAC, AVG, APO, DAgger, and PPO, APG and UDRL with a stateless expert (they take none); a discrete policy has no mean action and raises (no cross-entropy form yet). `q_filter=True`: SAC, TD3, REDQ, ASAC and AVG, without `ResidualPolicy`; refused where the critic rates states, not actions (PPO, APO), or there is none (APG, UDRL) |
| `ResidualPolicy` | Executes `clip(a_E + scale · a_π, −1, 1)`: the policy learns a correction on top of the expert. | `transform_action` | SAC with an expert, without `use_pid_policy` |
| `JSRLCurriculum` | The expert acts for the first `H_t` steps of each episode, the policy afterwards; `H_t` shrinks from `episode_length` to 0 over `decay_frac` of training. Logs `jsrl/horizon` and `jsrl/expert_frac`. | `action`, `eval_metrics` | SAC with an expert |
| `ExpertValueMetrics` | Logs the critic's value at the expert's actions (`value/q_expert_mean`), its gap to the value of the actions taken (`value/q_gap`) and, with `MCPretrain`, its gap to `φ*` (`value/phi_star_q_gap_ood`), on one replay batch per evaluation. | `eval_metrics` | SAC, TD3, REDQ, ASAC with an expert |

**Exploration** (`ajax.extensions.exploration`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EDGEExploration` | Lets the expert act in place of the policy, step by step, as a gate (`gate=`) decides. `"fixed"`: with probability `fixed_prob`; `"argmax"`: whenever the critic rates the expert's action higher; `"boltzmann"`: with a probability that grows with that gap. These three fade out over `decay_frac` of training. `"lcb"`, `"argmax_lcb"` and `"thompson"` also weigh the critics' disagreement. Logs `edge/live_expert_frac` and its gate's schedule (`edge/decay` or `edge/beta`). | `action`, `eval_metrics` | SAC with an expert |

**Changing the critic's target** (`ajax.extensions.target_mods`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `IBRL` | Adds `γ(1 − d) · max(Q(s′, a_E) − Q(s′, a′), 0)` to the target, so the values match acting with whichever of expert and policy looks better. Logs `ibrl/bonus` and `ibrl/expert_wins`. | `on_target` | SAC, TD3, REDQ, AVG; refused on an average-reward target (ASAC, APO), which has no discount for the bonus |
| `LCBGatedBootstrap` | Blends the policy's and the expert's next-action values in the target, weighted by a sigmoid of their pessimistic (lower-confidence-bound) scores. Logs its pessimism `lcb_bootstrap/beta` and its mean gate `lcb_bootstrap/p_expert`. | `on_target`, `eval_metrics` | SAC, TD3, REDQ, which have critic ensembles; refused on an average-reward target (ASAC, APO) |
| `CriticBlend` | Blends the target with a one-step target on `φ*`, `(1 − α) y + α y_E`, with `α` falling from 1 to 0 over `critic_warmup_frac`. Logs `α` as `critic_blend/expert_weight`, and `critic_blend/target_gap`. | `on_target`, `eval_metrics` | SAC with `MCPretrain` |
| `MCVarianceCorrection` | Where the critics disagree by more than `threshold` (variance), uses `φ*`'s value as the target. Logs `mc_variance/replaced_frac`. | `on_target` | SAC with `MCPretrain` |
| `ValueBox` | Lets the expert act where its value under `φ*` exceeds a threshold that rises from the lowest to the highest expert value over training. Logs `value_box/threshold`, `value_box/in_box_frac` and `value_box/expert_frac`. | `action`, `eval_metrics` | SAC with an expert and `MCPretrain` |

**Pre-training** (`ajax.extensions.pretrain`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CloningPretrain` | Rolls the expert out for `n_steps` steps per copy and trains the actor to reproduce its actions (`epochs` passes in batches of `batch_size`, at the actor's learning rate); the agent's observation statistics start from the dataset's. Logs the mean loss of its first and last epoch, `cloning/first_loss` and `cloning/last_loss`. | `pretrain`, `eval_metrics` | SAC, TD3, REDQ, ASAC, AVG, APO, DAgger (other agents refuse it when built) |
| `MCPretrain` | Rolls the expert out, fits `φ*` to its Monte-Carlo returns, records its value range, and by default nudges the agent's own critic towards it. Logs the range, `mc_pretrain/v_min` and `mc_pretrain/v_max`, and the fit: its first and last loss, `mc_pretrain/first_loss` and `mc_pretrain/last_loss`, and `φ*`'s mean value at the expert's actions, `mc_pretrain/q_expert_mean`. | `pretrain`, `eval_metrics` | SAC (other agents refuse it when built) |
| `BellmanPretrain` | Runs `n_steps` ordinary critic updates on the replay buffer, which holds the expert's prefill, before training. Its values bootstrap from an untrained critic; kept for ablations. Logs the critic loss of its first and last update, `bellman_pretrain/first_loss` and `bellman_pretrain/last_loss`. | `pretrain`, `eval_metrics` | SAC with an expert |
| `PhiRefresh` | Every `interval` steps, refits `φ*` for `steps` updates on the replay rows where the expert acted. | `post_update` | SAC with an expert and `MCPretrain` |
| `PriorData` | Adds a transition dataset (`ajax.buffers.dataset`) to the replay, where it stays, and to the observation and reward statistics, then runs `offline_updates` of the agent's own update on it before acting. Pair it with `learning_starts=0`. | `pretrain` | SAC, TD3, REDQ, ASAC, DQN (other agents refuse it when built) |

**Offline RL** (`ajax.extensions.offline`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConservativeQ` | CQL(H)'s critic term: `alpha` times, for each critic, the logsumexp of its values at `n_actions` uniform actions and as many of the policy's at `s` and `s'` (each less its log-density), minus its value at the data's action. Logs `cql/term` and `cql/gap`. | `critic_loss` | SAC, REDQ, ASAC, AVG, IQL: a Q critic and a policy with a density, refused on a deterministic one (TD3); see [SAC](../algorithms/sac.md#cql-and-cal-ql-offline) |
| `CalQL` | `ConservativeQ` with the policy actions' values floored at the row's return to go (Cal-QL), from a transition dataset (`return_to_go`; collected rows have none). Also logs `calql/bound_frac`, the share it raised. | `critic_loss` | SAC, REDQ, IQL (the replays that keep the return to go); refused on an average-reward critic (ASAC), without a replay (AVG) or on a deterministic policy (TD3) |
| `TargetBC` | ReBRAC's critic penalty: the target less `γ (1 − d) β Σ (ã′ − a′)²`, `ã′` the next actions the target scored and `a′` the dataset's next action (`next_action`; a row without one keeps its target). Logs `target_bc/penalty` and `target_bc/known_frac`. | `on_target` | TD3, whose on_target batch holds its target's next actions (see [ReBRAC](../algorithms/td3.md#rebrac-offline)) |

**Adversarial imitation** (`ajax.extensions.adversarial`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `AdversarialReward` | GAIL: a discriminator, trained after each rollout to tell its rows from a transition dataset of demonstrations, replaces the rollout's rewards with `-log(1 - D(s, a))`; the environment's own reward is not used. Logs `gail/reward`, `gail/disc_loss`, `gail/expert_acc` and `gail/policy_acc`. | `reward` | PPO with continuous actions (see [PPO](../algorithms/ppo.md#gail-imitation-by-a-discriminator)) |

**Critic ensembles** (`ajax.extensions.ensemble`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `KernelRepulsion` | Adds `coef` times the mean similarity (an RBF kernel) between the critics' predictions, which pushes the critics of an ensemble apart. Logs `kernel_repulsion/term`, `kernel_repulsion/kernel` and `kernel_repulsion/bandwidth`. | `critic_loss` | SAC, TD3, REDQ, ASAC with two or more critics; other agents, AVG's single critic included, refuse it when built |

**Measuring the critic** (`ajax.extensions.instrumentation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConditioningMetrics` | Logs the critic's effective rank (`Cond/critic_srank`), its fraction of dormant units, its feature and weight norms. It finds the critic from the agent's networks: an ensemble critic, or a Q-network. | `eval_metrics` | replay agents given `buffer=` (SAC, TD3, REDQ, ASAC, DQN); PPO, APO and PQN run with `expose_recent_rollout=True` (AVG too, but its rollout is one step per copy) |
| `BiasVoreDecomposition` | Splits the critic's squared one-step error into squared bias and variance (`EVarEst/bias_sq`, `EVarEst/var_resid`, `EVarEst/mse`, `EVarEst/abs_bias`), against a one-step target on its current parameters, or your own `value_target_fn`. | `eval_metrics` | as `ConditioningMetrics` |
| `BiasVorePenalty` | Adds `((1 − 2α)/α) · Var(residual)` to the critic loss; `alpha=0.5` adds nothing. Logs `bias_vore/penalty` and `bias_vore/residual_var`. | `critic_loss` | every agent with a `critic_loss` phase: SAC, TD3, REDQ, ASAC, AVG, PPO, APO, DQN, PQN |

Sources: TD3+BC is Fujimoto and Gu, 2021; CQL is Kumar et al., 2020, Cal-QL Nakamoto et al., 2023, and ReBRAC Tarasov et al., 2023; GAIL is Ho and Ermon, 2016; Q-filtered BC is Nair et al.,
ICRA 2018; residual RL is Johannink et al., 2019; JSRL is Uchendu et al.,
2023; the kernel's bandwidth follows the median heuristic of SVGD (Liu and
Wang, 2016); effective rank is Kumar et al., 2021, and dormant units Sokar et
al., 2023, and IBRL is Hu et al., *Imitation Bootstrapped Reinforcement
Learning*, 2023 (all cited in the code). EDGE, the LCB gates and the
bias-variance (EVarEst) measurements are research features
developed with Ajax, with no external reference in the code.

## Measuring a replay agent's critic

`ConditioningMetrics` and `BiasVoreDecomposition` need states the agent has
visited. An on-policy agent keeps its latest rollout when built with
`expose_recent_rollout=True`, as in the example at the top of this page. A
replay agent keeps them in its replay buffer, and the extension must be given
the buffer object. The extension is built before the agent, so take the
buffer from an agent built with the same settings (`n_envs`, `buffer_size`,
`batch_size`):

```python
from ajax import SAC
from ajax.extensions.instrumentation import BiasVoreDecomposition, ConditioningMetrics
from ajax.logging.wandb_logging import LoggingConfig

buffer = SAC("Pendulum-v1", learning_starts=500).buffer  # same settings as below
agent = SAC(
    "Pendulum-v1",
    learning_starts=500,
    extensions=(ConditioningMetrics(buffer=buffer), BiasVoreDecomposition(buffer=buffer)),
)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Cond/critic_dormant_frac"], evaluations["EVarEst/bias_sq"])
```

When the batch they need is not available yet (an evaluation before the
first rollout, say), these extensions log nothing rather than wrong numbers.

## Extensions that need the agent's context

Some extensions need objects only the agent builds: its environment, network
settings, optimisers, replay buffer or discount. They receive them through
`bind_to_agent`, which an agent calls on its whole stack when it builds its
training program; each extension picks the arguments it needs and returns a
filled-in copy. Each agent lists the keys it binds in its
`extension_context`. SAC binds everything its extensions read, which is why
`MCPretrain`, `BellmanPretrain` and `PhiRefresh` work with SAC only. TD3,
REDQ, ASAC, AVG and APO bind their environment, actor optimiser and
`expert_policy`, which is what `CloningPretrain` needs; TD3, REDQ and ASAC
also bind their replay buffer, which `ExpertValueMetrics` reads. DreamerV3
binds its own context. The other agents bind nothing. You never call
`bind_to_agent` yourself.

## Writing your own

An extension is a frozen dataclass subclassing
[`Extension`](../../src/ajax/extensions/base.py) that overrides the phases it
needs and declares what it reads. [Writing an extension](../developer/writing-an-extension.md)
walks through the contract, the dictionary each phase receives on each
agent, and a complete example. If a feature seems to need a new keyword on
an agent, that is the sign it should be an extension instead.

## See also

- [Expert guidance](expert-guidance.md): how the expert extensions fit
  together with SAC's expert options, with an example.
- [Extensions API](../api/extensions.md): every extension's settings.
- [`ajax/extensions/`](../../src/ajax/extensions): the source, one module per
  family.
