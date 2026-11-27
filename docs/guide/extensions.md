# Extensions

An extension adds a research feature to an agent's training without changing
the agent: an extra loss term, a change to the critic's target, another way to
choose actions while collecting, an expert to learn from, a pre-training step,
or measurements logged at each evaluation. Ajax ships forty-four of them,
listed in the [catalogue](#catalogue). This page shows how to pass them, which
agent takes which, and what each one does.

## Using extensions

Pass a tuple of extension objects as `extensions=`. Each object holds its own
settings:

```python
from ajax import PPO
from ajax.extensions.instrumentation import BiasVorePenalty, CriticDiagnostics
from ajax.logging import LoggingConfig

agent = PPO(
    "CartPole-v1",
    extensions=(
        BiasVorePenalty(alpha=0.3),  # an extra critic-loss term
        CriticDiagnostics(),  # measurements at each evaluation
    ),
)
logging = LoggingConfig(log_frequency=8_192, backends=())
state, evaluations = agent.train(seed=[1, 2], n_timesteps=16_384, logging_config=logging)
print(evaluations["Cond/critic_srank"])  # one row per seed, one column per evaluation
```

- **Extensions are fixed for a run.** They are compiled into the training
  program, so changing a setting means a new agent and a new compilation.
  Two extensions with equal settings compare and hash equal, so a second
  agent built the same way traces the same program, which the
  [compile cache](performance.md#the-compile-cache) can load instead of
  compiling.
- **Their state is the agent's.** What changes during training lives in
  `state.ext_state`, one entry per extension, which checkpoints save and a
  resumed run carries on.
- **One-shot work runs once.** An extension's initial state and its
  `pretrain` step run on a fresh run only, not when a run resumes from
  `initial_state=`.
- **Measurements come with the evaluations.** An extension's metrics join
  the evaluations `train` returns, next to `Eval/episodic mean reward`
  ([Logging](logging.md)). With `logging_config=None` there are no
  evaluations, and no metrics.
- **Order matters only where it can.** Target changes apply one after the
  other, in list order. Loss terms add up. For collection actions, the last
  extension that proposes one wins.

An empty tuple costs nothing: the agent's program is the same as without the
argument.

An agent's constructor takes its algorithm's hyperparameters plus
`extensions=()`. A feature kept apart as an extension works with every agent
that runs the part of training it touches, and each agent's code stays
readable next to its paper. The exceptions are the expert an agent tracks
along each trajectory, `expert_policy`, and `eval_expert_policy`, an expert
whose return the evaluations log and nothing else reads; every agent but
REINFORCE and the world models takes both ([Expert guidance](expert-guidance.md)).

## Which agent runs which phase

An extension acts at one or more *phases* of training: a critic target, a
loss term, the action taken, a metric at evaluation. The
[catalogue](#catalogue) names each extension's phases, and
[Extension phases](../developer/extension-phases.md) describes what each one
receives. There are fifteen: `pretrain`, `reward`, `observe`, `on_target`,
`critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`,
`actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory`,
`post_update` and `eval_metrics`.

The phases an agent runs are those its code declares, in
`supported_extension_phases`; this table and each agent's page list them.
Every agent runs `pretrain`, `post_update` and `eval_metrics`, which belong to
the shared training loop. The others depend on what the agent's update and
collection have:

| Agent | Other phases it runs |
| --- | --- |
| SAC, TD3, REDQ, CrossQ, ASAC, RVISAC, IQL | all but `reward` |
| TQC | all but `reward` and `choose_warmup_action` |
| AVG | all but `reward` and `choose_replay_action` |
| PPO | all but `choose_warmup_action` and `choose_replay_action` |
| APO | all but `reward`, `choose_warmup_action` and `choose_replay_action` |
| DQN | all but `reward`, `actor_loss`, `transform_action` and `actuate` |
| PQN | `observe`, `on_target`, `critic_loss`, `action`, `eval_action`, `track_memory` |
| DAgger | `observe`, `actor_loss`, `action`, `eval_action`, `track_memory` |
| UDRL | `observe`, `actor_loss`, `action`, `transform_action`, `actuate`, `choose_replay_action` |
| APG | `observe`, `actor_loss`, `transform_action`, `actuate` |
| REINFORCE | `observe`, `action`, `eval_action`, `track_memory` |
| DreamerV3, TDMPC2 | `observe` |
| TDMPC2MultiTask | none |

The shared collector runs an `action` extension in one of two places, before
or after the warm-up's choice, which the extension names in its
`action_slot` (`"pre_warmup"` or `"post_warmup"`);
[Expert guidance](expert-guidance.md#collection) describes both.

Running a phase is not enough: the agent must also have what the extension
needs, as the next section explains: PPO, APO and REINFORCE run `action` but
refuse the extensions acting there, which need an off-policy update.

## What an agent has, and what an extension needs

Each agent declares what its algorithm has, in `capabilities`:

| Capability | Meaning | Agents |
| --- | --- | --- |
| `actor` | a policy network trained by gradient | all but DQN, PQN |
| `q_critic` | a learned action value Q(s, a) | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, AVG, IQL, DQN, PQN, TDMPC2, TDMPC2MultiTask |
| `v_critic` | a learned state value V(s) | PPO, APO, REINFORCE, IQL, DreamerV3 |
| `discounted` | it maximises a discounted return, not the average reward | all but ASAC, RVISAC, APO, UDRL, DAgger, APG |
| `entropy_temperature` | a maximum-entropy objective, whose temperature the agent's state holds | SAC, REDQ, CrossQ, TQC, ASAC, RVISAC, AVG |
| `policy_density` | a stochastic policy, with a density over its actions | all with an actor but TD3, DAgger, APG |
| `replay` | its updates sample stored past transitions | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL, DQN, DAgger, UDRL, DreamerV3, TDMPC2, TDMPC2MultiTask |
| `off_policy` | its update stays correct on actions its policy did not choose | all but PPO, APO, REINFORCE, APG |
| `one_step_target` | its target bootstraps one step ahead, `r + γ V(s′)`, not on a multi-step or lambda-return | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, AVG, IQL, DQN |
| `next_action_target` | its target scores the critic at a next action, not a state value | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, AVG, DQN, PQN, TDMPC2, TDMPC2MultiTask |
| `continuing_target` | its target goes on through a termination into the reset observation | RVISAC |
| `critic_ensemble` | two critics or more | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL, TDMPC2, TDMPC2MultiTask; AVG with `num_critics` of 2 or more |
| `target_critic` | a slowly updated copy of its critic, which its bootstrap reads | SAC, TD3, REDQ, TQC, ASAC, RVISAC, IQL, DQN, TDMPC2, TDMPC2MultiTask |
| `distributional_critic` | its critic predicts quantiles of the return | TQC |
| `warmup` | it acts uniformly at random for its first steps (`learning_starts`) | SAC, TD3, REDQ, CrossQ, ASAC, RVISAC, AVG, IQL, DQN, TDMPC2 |
| `obs_networks` | its networks read the observation, not a world model's latent state | all but DreamerV3, TDMPC2, TDMPC2MultiTask |
| `reward_learning` | it learns from the reward, not from an expert's labels alone | all but DAgger and REINFORCE |

A run adds what its configuration and its stack give:

| Capability | Given by |
| --- | --- |
| `critic` | a `q_critic` or a `v_critic` |
| `scalar_target` | a critic that regresses on one target per row (every critic but TQC's) |
| `continuous_actions`, `discrete_actions` | the environment's actions |
| `agent_expert` | an `expert_policy` |
| `warmup_steps` | `learning_starts` above 0, on an agent with a `warmup` |
| `stored_steps` | a replay, or the latest rollout an on-policy or incremental agent keeps |
| `phi_star` | `MCPretrain` in the stack, which fits the expert critic `φ*` |
| `seeded_replay` | `ExpertPrefill` or `PriorData` in the stack, which fill the replay before training |

An extension needs what it declares in `requires` and what its phases imply:
a critic for `on_target` and `critic_loss`, an actor for `actor_loss`, an
off-policy update for `action` (the executed action is then not the
policy's), continuous actions for `transform_action` and `actuate`, a warm-up
with steps for `choose_warmup_action`, a replay for `choose_replay_action`.
An extension needing nothing runs on every agent. When the agent is built,
an extension needing something its run lacks is refused with
`MissingCapability`, which names it:

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

When the run has everything but the agent's code does not pass it on yet,
the refusal is `NotWired`, which says what is missing.
[Compatibility](compatibility.md), generated from the code, shows for every
agent and module which runs, which the algorithm cannot run, and which is not
wired yet.

## What is checked, and when

Mistakes in a stack fail early, with a `ValueError` that names each offending
extension (`MissingCapability` and `NotWired` are `ValueError`s):

| Refused | When |
| --- | --- |
| an extension needing a capability the run lacks (`MissingCapability`): `φ*` without `MCPretrain`, an expert on an agent without one, a Q critic on PPO | the agent is built |
| an extension the agent's code does not serve yet (`NotWired`): a phase it does not run, context it does not hand over, a batch key or a replay field it does not provide | the agent is built |
| a stateful expert called without its state: an extension calls its expert on the observations it is given, where no expert state exists, so a stateful one is allowed only when it is the agent's own `expert_policy`, on an agent that stores its actions ([Expert guidance](expert-guidance.md#the-expert)) | the agent is built |
| an `action` extension without an `action_slot`; more than one of `ResidualPolicy`, `ExpertGains` and `OuterLoopReference`; under gain mode, an extension using the expert's actions as the policy's | the agent is built |
| `CloningPretrain` after another pre-training extension; `PriorData` on a replay whose rows it cannot fill | training starts |
| with `memory=`, an `expert_policy` or an extension other than the observation features and `StorePolicyAction`, on every agent but APG (`NotImplementedError`) | training starts |

## Catalogue

Each table holds one module's extensions. "Works with" names the agents that
accept the extension; [Compatibility](compatibility.md) has the full matrix,
generated from the code. "The off-policy actor-critics" are SAC, TD3, REDQ,
CrossQ, TQC, ASAC, RVISAC, AVG and IQL, and "with an expert" means the agent
was given an `expert_policy`.

**Learning from an expert** (`ajax.extensions.expert`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OnlineBC` | Pulls the actor's mean action towards the expert's, weighted by `φ*`'s value at each state and fading to zero over the first `critic_warmup_frac` of training. Logs `online_bc/term` and its weight, `online_bc/weight`. | `actor_loss` | SAC with an expert and `MCPretrain` |
| `ImitationLoss` | Adds TD3+BC's cloning term towards a fixed expert it holds, `coef · mean((μ(s) − a_E(s))²) / 4`. With the agent's own `expert_policy`, it reads the actions the agent stored, so the expert may be stateful. `anneal_frac` lowers its weight linearly to 0 at that fraction of training (kickstarting). `q_filter=True` counts only the rows where the critic rates the expert's action above the policy's (Q-filtered BC) and logs the share kept, `imitation/q_filter_frac`. `target="replay"` clones the replayed rows' own actions, so logged data needs no expert; `alpha` then makes the actor loss TD3+BC's, `λ` times the agent's own plus `coef · mean((μ(s) − a)²)` in the environment's units, `λ = alpha / mean|Q_1(s, μ(s))|` ([TD3+BC](../algorithms/td3.md#td3bc-offline)). Logs `imitation/term` and the squared distance, `imitation/mse`. | `actor_loss`, `eval_metrics` | continuous actions: the off-policy actor-critics, APO and DAgger; with a stateless expert on IQL, PPO, APG and UDRL, which store no expert actions. `q_filter=True`: the off-policy actor-critics, without `ResidualPolicy`. `target="replay"`: the off-policy actor-critics but AVG, which keeps no replay |
| `ResidualPolicy` | Executes `clip(a_E + scale · a_π, −1, 1)`: the policy learns a correction to the expert. The replay stores the executed action and the critics score it; `critic_reads_residual=True` has them store and score the residual instead, and `zero_start=True` starts the policy as the expert. | `pretrain`, `transform_action` | the off-policy actor-critics, PPO, APO, APG and UDRL, with an expert and continuous actions |
| `ExpertGains` | Gain mode: the policy's action `a` sets each of the expert's learnable gains to `anchor · max_factor^a` (`max_factor` 10 by default), and the expert acts with them; the replay and the critics hold the gains. Needs a stateful expert with `learnable_fields` (`register_learnable_gains`, or a `GainScheduledLaw`). | `actuate` | as `ResidualPolicy`, without `ExpertActionInput` |
| `ExpertWarmup` | During the warm-up, the expert takes a share `fraction` of the steps (one draw per step for every copy) and uniform random actions take the others; the replay flags the rows the expert took. | `choose_warmup_action` | the agents with a warm-up (<!-- agents: warmup-action -->SAC, TD3, REDQ, CrossQ, ASAC, RVI-SAC, AVG, IQL, DQN<!-- /agents -->), with an expert and `learning_starts` above 0 |
| `JSRLCurriculum` | The expert acts for the first `h` steps of each episode, the policy afterwards. `schedule="performance"` (the paper's, the default) lowers `h` from the task's horizon a stage at a time when an evaluation of the combined policy passes; `"random"` draws it each episode; `"time"` (Ajax's own) lowers it to 0 over `decay_frac` of training. Logs `jsrl/horizon` and `jsrl/expert_frac`, and with `"performance"` `jsrl/stage` and `jsrl/best`. | `action`, `post_update`, `eval_metrics` | the off-policy actor-critics, DQN, PQN, DAgger and UDRL, with an expert |
| `StorePolicyAction` | An ablation of off-policy correctness: the replay stores the action the policy chose, not the one executed, so the rows an expert, a warm-up or a residual acted in no longer hold the action behind their reward and next observation. | `choose_replay_action` | <!-- agents: replay-action -->SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVI-SAC, IQL, DQN and UDRL<!-- /agents --> |
| `ExpertValueMetrics` | Logs, on one replay batch per evaluation, the critic's value at the expert's actions (`value/q_expert_mean`), its gap to the value of the actions taken (`value/q_gap`) and, with `MCPretrain`, its gap to `φ*` (`value/phi_star_q_gap_ood`). | `eval_metrics` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL and DQN, with an expert |

**Acting around a controller** (`ajax.extensions.structure`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OuterLoopReference` | The policy's action moves the targets the expert reads, each by at most its `max_change`, and the unchanged expert acts; the replay and the critics hold the changes. | `actuate` | as `ResidualPolicy` |
| `BumplessTracking` | The expert's memory follows the action applied, by the conditioning technique, from the columns its errors read. | `track_memory` | every agent with a stateful expert but APG and UDRL, whose collectors step the environment themselves |

**Observation features** (`ajax.extensions.observe`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ExpertActionInput` | Appends the expert's action at each observation, raw (it is an action), from the state the agent tracks the expert in. | `observe` | every agent with an expert but DAgger, which would read the label it learns; not under gain mode |
| `ExpertStateInput` | Appends a stateful expert's internal state before its call at the observation (`ExpertStateInput.for_expert(expert)` sizes it). | `observe` | every agent with a stateful expert |
| `PIDTermsInput` | Appends a PID controller's terms for each tracking error `e = obs[target_idx] − obs[current_idx]`: the error, its integral (clamped at `integral_limit`) and its difference, with sampling period `dt`; raw, and started afresh each episode. A [`PIDPolicy`](networks.md#the-pid-policy) actor reads them. | `observe` | every agent but `TDMPC2MultiTask`, whose dataset holds no feature columns |
| `PreviousAction` | Appends the action executed just before each observation, raw, in agent units: zero at an episode's first observation, a discrete action one-hot. A recurrent actor then reads its own past actions, as Ni et al. (2022)'s does ([Memory](memory.md)). | `observe` | every agent but `TDMPC2MultiTask` |
| `PreviousReward` | Appends the reward paid for the action executed just before each observation, raw: zero at an episode's first observation. Ni et al. (2022)'s recurrent actor and critic read it with the previous action ([Memory](memory.md)). `agent.policy`'s `act` then needs each reward. | `observe` | every agent but `TDMPC2MultiTask` and APG, whose rollout passes no reward yet |
| `TrainingProgressInput` | Appends the fraction of training done (`ajax.schedule.progress`); an evaluation holds the collector's. | `observe` | every agent but `TDMPC2MultiTask` |

**Exploration** (`ajax.extensions.exploration`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EDGEExploration` | Lets the expert act in place of the policy, step by step, as its `gate` decides. `"fixed"` (the default): with probability `fixed_prob`; `"argmax"`: whenever the critic rates the expert's action higher; `"boltzmann"`: with a probability that grows with that gap. These three fade out over `decay_frac` of training. `"lcb"`, `"argmax_lcb"` and `"thompson"` also weigh the critics' disagreement. Logs `edge/live_expert_frac` and its gate's schedule (`edge/decay` or `edge/beta`). | `action`, `eval_metrics` | the off-policy actor-critics, DAgger and UDRL, with an expert and continuous actions |

**Changing the critic's target** (`ajax.extensions.target_mods`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `IBRL` | Adds `γ(1 − d) · max(Q(s′, a_E) − Q(s′, a′), 0)` to the target, so the values match acting with whichever of expert and policy looks better: `a′` is the next action the target scored, `Q` the minimum over the critics the target took its minimum over. With `hybrid_acting` (the default) it also acts with whichever of the two the target critics rate higher, in collection and evaluation. `soft_choice=True` draws that choice, with probability `sigmoid(soft_choice_beta · (Q(a_E) − Q(a)))` for the expert (`soft_choice_beta` 10 by default). On REDQ, and with `soft_choice`, the bonus can be negative ([Expert guidance](expert-guidance.md#critic)). Logs `ibrl/bonus` and `ibrl/expert_wins`, and with `hybrid_acting` `ibrl/expert_frac`. | `on_target`; `action`, `eval_action`, `eval_metrics` with `hybrid_acting` | SAC, TD3, REDQ, TQC and IQL; refused on an average-reward target (ASAC, RVISAC), which has no discount for the bonus, and without target critics (AVG, CrossQ) |
| `LCBGatedBootstrap` | Blends the policy's and the expert's next-action values in the target, weighted by a sigmoid of their pessimistic (lower-confidence-bound) scores. Logs its pessimism `lcb_bootstrap/beta` and its mean gate `lcb_bootstrap/p_expert`. | `on_target`, `eval_metrics` | SAC, TD3, REDQ, TQC and IQL |
| `CriticBlend` | Blends the target with a one-step target on `φ*`, `(1 − α) y + α y_E`, with `α` falling from 1 to 0 over `critic_warmup_frac`. Logs `α` as `critic_blend/expert_weight`, and `critic_blend/target_gap`. | `on_target`, `eval_metrics` | SAC with `MCPretrain` |
| `MCVarianceCorrection` | Where the critics' variance exceeds `threshold`, uses `φ*`'s value as the target. Logs `mc_variance/replaced_frac`. | `on_target` | SAC with `MCPretrain` |
| `ValueBox` | Lets the expert act where its value under `φ*` exceeds a threshold that rises from the lowest to the highest expert value over training. Logs `value_box/threshold`, `value_box/in_box_frac` and `value_box/expert_frac`. | `action`, `eval_metrics` | SAC with an expert and `MCPretrain` |

**Pre-training** (`ajax.extensions.pretrain`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CloningPretrain` | Rolls the expert out for `n_steps` steps per copy and trains the actor to reproduce its actions (`epochs` passes in batches of `batch_size`, at the actor's learning rate); the agent's observation statistics start from the dataset's. Logs the mean loss of its first and last epoch, `cloning/first_loss` and `cloning/last_loss`. | `pretrain`, `eval_metrics` | the off-policy actor-critics, APO and DAgger, with continuous actions |
| `MCPretrain` | Rolls the expert out, fits the expert critic `φ*` to its Monte-Carlo returns, records its value range, and by default nudges the agent's own critic towards it. Logs the range (`mc_pretrain/v_min`, `mc_pretrain/v_max`), the fit's first and last loss (`mc_pretrain/first_loss`, `mc_pretrain/last_loss`) and `φ*`'s mean value at the expert's actions, `mc_pretrain/q_expert_mean`. | `pretrain`, `eval_metrics` | SAC |
| `BellmanPretrain` | Runs `n_steps` ordinary critic updates on the replay before training, once `ExpertPrefill` or `PriorData` has filled it. Its values bootstrap from an untrained critic; kept for ablations. Logs the critic loss of its first and last update, `bellman_pretrain/first_loss` and `bellman_pretrain/last_loss`. | `pretrain`, `eval_metrics` | SAC, TD3, REDQ and TQC, with `ExpertPrefill` or `PriorData` |
| `PhiRefresh` | Every `interval` steps, refits `φ*` for `steps` updates on the replay rows where the expert acted. | `post_update` | SAC with an expert and `MCPretrain` |
| `PriorData` | Adds a transition dataset (`ajax.buffers.dataset`) to the replay, where it stays, and to the observation and reward statistics, then runs `offline_updates` of the agent's own update on it before acting. Pair it with `learning_starts=0`. | `pretrain` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL and DQN |
| `ExpertPrefill` | The expert acts `n_steps` steps in every copy before training, and its transitions go to the replay as collection writes its own, flagged as expert rows. They count once in the observation and reward statistics, as `PriorData`'s do, and in no timestep. `offline_updates` of the agent's own update then run on them. It runs before the other pre-training, which may read its rows. | `pretrain` | as `PriorData`, with an expert or an `expert` of its own |

**Rows kept for the whole run** (`ajax.extensions.retained`)

These put their rows in a ring of their own beside the one collection writes,
so nothing evicts them. The update ratio, the critics and their architecture
stay the agent's.

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `RetainedDemonstrations` | Keeps `n_steps` steps of the agent's expert per copy, or a transition `dataset`, for the whole run (DDPGfD). `share` is the part of each batch they fill; `None` draws each row over all the rows held. After `CloningPretrain`, with `share=0.5`, it is Rainbow-DemoRL's BC start plus prefill. | `pretrain` | as `PriorData`; the expert's rows need an expert |
| `RLPD` | The same rows at `share=0.5`, exact in every batch, on critics with LayerNorm: pass `critic_architecture=("256", "layernorm", "relu", "256", "layernorm", "relu")`, or `critic_layer_norm=False` for the ablation. `critic_subset` has each target bootstrap on that many target critics drawn at random. The original configuration is `REDQ` with those critics. | `pretrain`, `on_target` | as `RetainedDemonstrations`; `critic_subset` needs a Q critic with targets and next actions, on an agent without a subset of its own |

**Fine-tuning a clone** (`ajax.extensions.porl`)

`porl(cloning, presample_steps, epsilon, features)` returns the agent's
keyword arguments for PORL (Xiao et al., 2025): `cloning` clones the expert,
then only a fresh critic trains for `presample_steps` environment steps, then
the agent trains as usual. A stateful expert needs its memory in `features`:
`ExpertStateInput`, or `PIDTermsInput` for a PI whose anti-windup holds its
integrator.

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EpsilonGreedyPresample` | For the first `presample_steps` environment steps, each copy takes a uniform action with probability `epsilon` and the actor's mean otherwise, then the agent's own action. The replay stores the action taken, and the observation statistics stay where pre-training left them, so the clone's actions hold too. | `action`, `pretrain`, `post_update` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL, UDRL and DAgger, with continuous actions and no action transform |

**Blending with the expert** (`ajax.extensions.cheq`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CHEQ` | Executes `(1 − λ) a_E + λ a_RL`, with `λ` falling from `lambda_max` to `lambda_min` as the critics' disagreement, a share of the spread of the values their targets bootstrap, rises from `u_low` to `u_high`. `λ` joins every observation, the replay stores `a_RL`, and a stateful expert needs `PIDTermsInput` or `ExpertStateInput` in the stack. Logs `cheq/weight`, `cheq/uncertainty` and `cheq/value_spread`. | `observe`, `action`, `eval_action`, `on_target`, `critic_loss`, `post_update`, `eval_metrics` | SAC, TD3, REDQ, CrossQ, ASAC, RVISAC and IQL, with an expert and two critics or more |

**Offline RL** (`ajax.extensions.offline`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConservativeQ` | CQL(H)'s critic term: `alpha` times, for each critic, the logsumexp of its values at `n_actions` uniform actions and as many of the policy's at `s` and `s′` (each less its log-density), minus its value at the data's action. Logs `cql/term` and `cql/gap`. | `critic_loss` | the off-policy actor-critics but TD3, whose deterministic policy has no density ([SAC](../algorithms/sac.md#cql-and-cal-ql-offline)) |
| `CalQL` | `ConservativeQ` with the policy actions' values floored at the row's return to go (Cal-QL), from a transition dataset's `return_to_go`; collected rows have none. Also logs `calql/bound_frac`, the share it raised. | `critic_loss` | SAC, REDQ, CrossQ, TQC and IQL |
| `TargetBC` | ReBRAC's critic penalty: the target less `γ (1 − d) β Σ (ã′ − a′)²`, `ã′` the next actions the target scored and `a′` the dataset's next action (`next_action`; a row without one keeps its target). Logs `target_bc/penalty` and `target_bc/known_frac`. | `on_target` | SAC, TD3, REDQ, CrossQ and TQC ([ReBRAC](../algorithms/td3.md#rebrac-offline)) |

**Adversarial imitation** (`ajax.extensions.adversarial`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `AdversarialReward` | GAIL: a discriminator, trained after each rollout to tell its rows from a transition dataset of demonstrations, replaces the rollout's rewards with `−log(1 − D(s, a))`; the environment's own reward is not used. It reads the action the environment runs, clipped to `[−1, 1]`. Logs `gail/reward`, `gail/disc_loss`, `gail/expert_acc` and `gail/policy_acc`. | `reward` | PPO with continuous actions ([PPO](../algorithms/ppo.md#gail-imitation-by-a-discriminator)) |

**Critic ensembles** (`ajax.extensions.ensemble`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `KernelRepulsion` | Adds `coef` times the mean similarity (an RBF kernel) between the critics' predictions, which pushes the critics of an ensemble apart. Logs `kernel_repulsion/term`, `kernel_repulsion/kernel` and `kernel_repulsion/bandwidth`. | `critic_loss` | the off-policy actor-critics with two critics or more (AVG has one by default) |

**Continuing tasks** (`ajax.extensions.centering`, `ajax.extensions.reset_cost`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ValueCentering` | Subtracts from the target an offset learned as the reward rate (reward centering), so the values lose the constant `r(π) / (1 − γ)`. Its `rule` is one of `ajax.agents.centering`'s: `RVIBatchMean(rate)` (the default, 0.005), the batch mean of the next values the target read, followed slowly (RVI-SAC's); `TDErrorCentering(step_size)`, the mean TD error of the centred target; `ReferencePoint(observation, action)`, the target critics' value there (ASAC's `Q(0, 0)`). It comes last of the extensions shaping the target. On a task whose episodes terminate, centering also prices an episode's length, unless `ResetCost`, before it, has the target go on through terminations. Logs `centering/offset`. | `on_target` | SAC, TD3, REDQ, CrossQ, TQC, AVG, IQL and DQN; refused on a lambda-return (PPO, PQN) and on the average-reward agents, which centre themselves (ASAC, RVISAC, APO) |
| `ResetCost` | Treats a termination as a step into the reset state: the target bootstraps on the reset observation the replay stores, and a terminated row is charged `c`. `c` is a Lagrange multiplier: a reset critic learns the policy's termination rate, and `c` rises while that rate exceeds `target_rate` (1e-3) and falls while it is below, never below 0. Its settings are RVI-SAC's (`ajax.agents.reset_cost.ResetCostSettings`). It refuses a stack reading the episode a termination ended (`IBRL`, `CriticBlend`, `LCBGatedBootstrap`, `CalQL`). Logs `reset_cost/cost`, `reset_cost/rate` and `reset_cost/critic_loss`. | `on_target` | SAC, TD3, REDQ, CrossQ, TQC and DQN; refused on IQL (no next action), on AVG and the on-policy agents (their rollouts keep the final observation), and on ASAC and RVISAC, which price terminations themselves |

**Temperature annealing** (`ajax.extensions.annealing`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `MEW` | Minimum Excess Work (Adamczyk et al., 2026): sets the temperature in place of the agent's own rule, lowering it at each update by `beta <- beta + speed / sqrt(beta * sum_k Cov(r_t, r_t+k))`, `alpha = 1 / beta`, the covariance that of the last `history` rewards per environment (from the replay; MEW keeps them itself on AVG). It waits for more than `history` steps per environment. `speed` 1e-6 and `history` 50,000 are the authors' code's. Logs `mew/friction`. | `post_update`, `eval_metrics` | the maximum-entropy agents: SAC, REDQ, CrossQ, TQC, ASAC, RVISAC and AVG |

**Measuring the critic** (`ajax.extensions.instrumentation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CriticDiagnostics` | Logs the critic's effective rank (`Cond/critic_srank`), its fraction of dormant units, its feature and weight norms. It finds the critic from the agent's networks: an ensemble critic, or a Q-network. | `eval_metrics` | every agent with a critic (all but APG, UDRL and DAgger): from its replay, or from the latest rollout an on-policy or incremental agent keeps (AVG's is one step per copy) |
| `BiasVoreDecomposition` | Splits the critic's squared one-step error into squared bias and variance (`EVarEst/bias_sq`, `EVarEst/var_resid`, `EVarEst/mse`, `EVarEst/abs_bias`), against a one-step target on its current parameters, or your own `value_target_fn`. | `eval_metrics` | as `CriticDiagnostics` |
| `BiasVorePenalty` | Adds `((1 − 2α)/α) · Var(residual)` to the critic loss; `alpha=0.5` adds nothing. Logs `bias_vore/penalty` and `bias_vore/residual_var`. | `critic_loss` | the off-policy actor-critics but TQC, whose critic regresses on quantiles, and PPO, APO, DQN and PQN |

**Recording representations** (`ajax.extensions.representation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `RepresentationProbe` | Reads the networks' activations and parameters on a fixed set of states every `probe_interval` steps, writes them to a directory and logs each layer's effective rank, alignment with the previous probe, activation norm and dead units ([guide](representations.md)). | `pretrain`, `post_update`, `eval_metrics` | every agent whose networks read the observation but APG and UDRL; not the world models |

Sources: reward centering is Naik, Wan, Tomar and Sutton, RLC 2024, and the
reset cost Hisaki and Ono, ICML 2024; TD3+BC is Fujimoto and Gu, 2021; CQL is
Kumar et al., 2020, Cal-QL Nakamoto et al., 2023, and ReBRAC Tarasov et al.,
2023; GAIL is Ho and Ermon, 2016; Q-filtered BC is Nair et al., ICRA 2018;
residual RL is Johannink et al., 2019; JSRL is Uchendu et al., 2023; the
kernel's bandwidth follows the median heuristic of SVGD (Liu and Wang, 2016);
effective rank is Kumar et al., 2021, and dormant units Sokar et al., 2023;
IBRL is Hu et al., *Imitation Bootstrapped Reinforcement Learning*, 2023 (all
cited in the code). EDGE, the LCB gates and the bias-variance (EVarEst)
measurements were developed with Ajax and have no external reference.

## Measuring a replay agent's critic

`CriticDiagnostics` and `BiasVoreDecomposition` need states the agent has
visited. An on-policy or incremental agent keeps its latest rollout for them,
as in the example at the top of this page; a replay agent's measurements
sample its replay:

```python
from ajax import SAC
from ajax.extensions.instrumentation import BiasVoreDecomposition, CriticDiagnostics
from ajax.logging import LoggingConfig

agent = SAC(
    "Pendulum-v1",
    learning_starts=500,
    extensions=(CriticDiagnostics(), BiasVoreDecomposition()),
)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Cond/critic_dormant_frac"], evaluations["EVarEst/bias_sq"])
```

When the batch they need is not available yet (an evaluation before the
first rollout, say), these extensions log nothing rather than wrong numbers.

## Extensions that need the agent's context

Some extensions need objects only the agent builds: its environment, network
settings, optimisers, replay buffer or discount. The agent hands them over
when it builds its training program, through each extension's
`bind_to_agent`; you never call it yourself. Each agent lists what it hands
over in its `extension_context`, and an extension needing more is refused
with `NotWired`. SAC hands over the most, which is why `MCPretrain`, and the
extensions reading its `φ*`, run on SAC only.

## Writing your own

An extension is a frozen dataclass subclassing
[`Extension`](../../src/ajax/extensions/base.py) that overrides the phases it
needs and declares what it reads.
[Writing an extension](../developer/writing-an-extension.md) walks through
the contract, the dictionary each phase receives on each agent, and a
complete example. If a feature seems to need a new keyword on an agent, that
is the sign it should be an extension instead.

## See also

- [Expert guidance](expert-guidance.md): how the expert extensions fit
  together, with an example.
- [Extensions API](../api/extensions.md): every extension's settings.
- [`ajax/extensions/`](../../src/ajax/extensions): the source, one module per
  family.
