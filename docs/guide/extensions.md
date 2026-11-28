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
`supported_extension_phases`, which this table and each agent's page list.
All run the shared loop's `pretrain`, `post_update` and `eval_metrics`; the
others depend on what the agent's update and collection have:

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

An `action` extension runs before or after the warm-up's choice, as its
`action_slot` says (`"pre_warmup"` or `"post_warmup"`;
[Expert guidance](expert-guidance.md#collection) describes both).

Running a phase is not enough: the agent must also have what the extension
needs, as the next section explains: PPO, APO and REINFORCE run `action` but
refuse the extensions acting there, which need an off-policy update; the
refusal names the phase.

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
| `reward_learning` | it learns from the reward, not from an expert's labels alone | all but DAgger |

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
`MissingCapability`, which names it, and the phase needing it where one does:

```python
import jax.numpy as jnp

from ajax import DQN
from ajax.extensions.expert import ImitationLoss


def expert(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


try:
    DQN("CartPole-v1", extensions=(ImitationLoss(expert),))
except ValueError as error:
    print(error)  # DQN cannot run ImitationLoss ('imitation_loss'), which needs actor (...) for its actor_loss phase, continuous_actions (...).
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

One table per module. "Works with" names the agents that accept the
extension, and "details" links its explanation or its settings;
[Compatibility](compatibility.md), generated from the code, has the full
matrix. "The off-policy actor-critics" are
<!-- agents: off-policy-actor-critics -->SAC, TD3, REDQ, CrossQ, TQC, ASAC,
RVISAC, AVG and IQL<!-- /agents -->, and "with an expert" means the agent was
given an `expert_policy`.

**Learning from an expert** (`ajax.extensions.expert`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OnlineBC` | Pulls the actor's mean action towards the expert's, weighted by `φ*`'s value and fading out over `critic_warmup_frac` of training ([details](../api/extensions.md#ajax.extensions.expert.OnlineBC)). | `actor_loss` | SAC with an expert and `MCPretrain` |
| `ImitationLoss` | Adds TD3+BC's cloning term towards the expert's actions, weighted by `coef`. `q_filter=True` keeps the rows where the critic rates the expert higher; `target="replay"` clones the logged actions instead ([details](expert-guidance.md#imitation-alongside-the-reward)). | `actor_loss`, `eval_metrics` | continuous actions: the off-policy actor-critics, APO, DAgger, PPO, APG and UDRL (on PPO, APG and UDRL a stateless expert); `q_filter` needs a Q critic, `target="replay"` a replay |
| `ResidualPolicy` | Executes `clip(a_E + scale · a_π, −1, 1)`, so the policy learns a correction to the expert; `zero_start=True` starts it as the expert ([details](expert-guidance.md#collection)). | `pretrain`, `transform_action` | the off-policy actor-critics, PPO, APO, APG and UDRL, with an expert and continuous actions |
| `ExpertGains` | Gain mode: the policy's action sets the expert's learnable gains, between `1 / max_factor` and `max_factor` times their anchors, and the expert acts with them ([details](expert-guidance.md#learning-a-pids-gains-gain-mode)). | `actuate` | as `ResidualPolicy`, with a stateful expert exposing `learnable_fields` |
| `ExpertWarmup` | During the warm-up, the expert takes a share `fraction` of the steps and uniform random actions the others ([details](expert-guidance.md#warm-up)). | `choose_warmup_action` | the agents with a warm-up (<!-- agents: warmup-action -->SAC, TD3, REDQ, CrossQ, ASAC, RVI-SAC, AVG, IQL, DQN<!-- /agents -->), with an expert and `learning_starts` above 0 |
| `JSRLCurriculum` | The expert acts for the first `h` steps of each episode and the policy afterwards; `schedule` lowers `h` to 0, by default as evaluations pass (the paper's) ([details](expert-guidance.md#collection)). | `action`, `post_update`, `eval_metrics` | the off-policy actor-critics, DQN, PQN, DAgger and UDRL, with an expert |
| `StorePolicyAction` | An ablation: the replay stores the action the policy chose, not the one executed ([details](../api/extensions.md#ajax.extensions.expert.StorePolicyAction)). | `choose_replay_action` | <!-- agents: replay-action -->SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVI-SAC, IQL, DQN and UDRL<!-- /agents --> |
| `ExpertValueMetrics` | Logs the critic's value at the expert's actions and its gap to the actions taken, on a replay batch at each evaluation ([details](expert-guidance.md#evaluation)). | `eval_metrics` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL and DQN, with an expert |

**Acting around a controller** (`ajax.extensions.structure`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OuterLoopReference` | The policy's action moves the targets the expert reads, each by at most `max_change`, and the expert acts unchanged ([details](expert-guidance.md#acting-around-a-controller)). | `actuate` | as `ResidualPolicy` |
| `BumplessTracking` | Keeps the expert's memory true to the action applied, from the readings in `reading_channels` ([details](expert-guidance.md#acting-around-a-controller)). | `track_memory` | every agent with a stateful expert but APG and UDRL |

**Observation features** (`ajax.extensions.observe`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ExpertActionInput` | Appends the expert's action to each observation ([details](expert-guidance.md#observation)). | `observe` | every agent with an expert but DAgger; not under gain mode |
| `ExpertStateInput` | Appends a stateful expert's internal state, sized by `ExpertStateInput.for_expert(expert)` ([details](expert-guidance.md#observation)). | `observe` | every agent with a stateful expert |
| `PIDTermsInput` | Appends a PID's terms for the error `obs[target_idx] − obs[current_idx]`, which a [`PIDPolicy`](networks.md#the-pid-policy) actor reads ([details](../api/extensions.md#ajax.extensions.observe.PIDTermsInput)). | `observe` | every agent but `TDMPC2MultiTask` |
| `PreviousAction` | Appends the action executed before each observation, which recurrent networks read ([details](memory.md)). | `observe` | every agent but `TDMPC2MultiTask` |
| `PreviousReward` | Appends the reward paid before each observation, which recurrent networks read ([details](memory.md)). | `observe` | every agent but `TDMPC2MultiTask` and APG |
| `TrainingProgressInput` | Appends the fraction of training done ([details](../api/extensions.md#ajax.extensions.observe.TrainingProgressInput)). | `observe` | every agent but `TDMPC2MultiTask` |

**Exploration** (`ajax.extensions.exploration`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EDGEExploration` | Lets the expert act in place of the policy, step by step, as its `gate` decides: by default with probability `fixed_prob`, fading out over `decay_frac` of training ([details](expert-guidance.md#collection)). | `action`, `eval_metrics` | the off-policy actor-critics, DAgger and UDRL, with an expert and continuous actions |

**Changing the critic's target** (`ajax.extensions.target_mods`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `IBRL` | Bootstraps the target on the better of the expert's and the policy's next actions, and by default (`hybrid_acting`) acts with the better of the two ([details](expert-guidance.md#critic)). | `on_target`; `action`, `eval_action`, `eval_metrics` with `hybrid_acting` | SAC, TD3, REDQ, TQC and IQL |
| `LCBGatedBootstrap` | Blends the policy's and the expert's next-action values in the target, weighted by their pessimistic (lower-confidence-bound) scores ([details](../api/extensions.md#ajax.extensions.target_mods.LCBGatedBootstrap)). | `on_target`, `eval_metrics` | SAC, TD3, REDQ, TQC and IQL |
| `CriticBlend` | Blends the target with a one-step target on `φ*`, whose share falls from 1 to 0 over `critic_warmup_frac` ([details](../api/extensions.md#ajax.extensions.target_mods.CriticBlend)). | `on_target`, `eval_metrics` | SAC with `MCPretrain` |
| `MCVarianceCorrection` | Uses `φ*`'s value as the target where the critics' variance exceeds `threshold` ([details](../api/extensions.md#ajax.extensions.target_mods.MCVarianceCorrection)). | `on_target` | SAC with `MCPretrain` |
| `ValueBox` | Lets the expert act where its value under `φ*` exceeds a threshold that rises over training ([details](expert-guidance.md#collection)). | `action`, `eval_metrics` | SAC with an expert and `MCPretrain` |

**Pre-training** (`ajax.extensions.pretrain`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CloningPretrain` | Before training, trains the actor to reproduce the expert's actions over `n_steps` steps of its rollouts per copy ([details](expert-guidance.md#before-training)). | `pretrain`, `eval_metrics` | the off-policy actor-critics, APO and DAgger, with continuous actions |
| `MCPretrain` | Before training, fits the expert critic `φ*` to the expert's Monte-Carlo returns, and by default nudges the agent's critic towards it ([details](expert-guidance.md#before-training)). | `pretrain`, `eval_metrics` | SAC |
| `BellmanPretrain` | Before training, runs `n_steps` ordinary critic updates on the replay `ExpertPrefill` or `PriorData` filled; kept for ablations ([details](expert-guidance.md#before-training)). | `pretrain`, `eval_metrics` | SAC, TD3, REDQ and TQC, with `ExpertPrefill` or `PriorData` |
| `PhiRefresh` | Refits `φ*` every `interval` steps on the replay rows where the expert acted ([details](../api/extensions.md#ajax.extensions.pretrain.PhiRefresh)). | `post_update` | SAC with an expert and `MCPretrain` |
| `PriorData` | Adds a transition `dataset` to the replay, then runs `offline_updates` of the agent's own update on it; pair it with `learning_starts=0` ([details](../api/extensions.md#ajax.extensions.pretrain.PriorData)). | `pretrain` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL and DQN |
| `ExpertPrefill` | The expert acts `n_steps` steps in every copy before training, and its transitions go to the replay as expert rows ([details](expert-guidance.md#before-training)). | `pretrain` | as `PriorData`, with an expert or an `expert` of its own |

**Rows kept for the whole run** (`ajax.extensions.retained`)

These keep their rows in a ring of their own, which nothing evicts.

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `RetainedDemonstrations` | Keeps `n_steps` steps of the expert per copy, or a transition `dataset`, for the whole run, as a `share` of each batch (DDPGfD) ([details](../api/extensions.md#ajax.extensions.retained.RetainedDemonstrations)). | `pretrain` | as `PriorData` |
| `RLPD` | The same rows as half of every batch, on critics with LayerNorm, which it checks unless `critic_layer_norm=False` ([details](../api/extensions.md#ajax.extensions.retained.RLPD)). | `pretrain`, `on_target` | as `PriorData` |

**Fine-tuning a clone** (`ajax.extensions.porl`)

`porl(cloning, presample_steps, epsilon, features)` returns the agent's
keyword arguments for PORL (Xiao et al., 2025): clone the expert, train only
a fresh critic for `presample_steps` steps, then train as usual. A stateful
expert needs its memory in `features` (`ExpertStateInput`, or
`PIDTermsInput`).

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EpsilonGreedyPresample` | For the first `presample_steps` steps, acts with the actor's mean action, or a uniform one with probability `epsilon` ([details](../api/extensions.md#ajax.extensions.porl.EpsilonGreedyPresample)). | `action`, `pretrain`, `post_update` | SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVISAC, IQL, UDRL and DAgger, with continuous actions and no action transform |

**Blending with the expert** (`ajax.extensions.cheq`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CHEQ` | Executes a blend of the expert's and the policy's actions, the policy's weight falling from `lambda_max` to `lambda_min` as the critics disagree more ([details](../api/extensions.md#ajax.extensions.cheq.CHEQ)). | `observe`, `action`, `eval_action`, `on_target`, `critic_loss`, `post_update`, `eval_metrics` | SAC, TD3, REDQ, CrossQ, ASAC, RVISAC and IQL, with an expert |

**Offline RL** (`ajax.extensions.offline`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConservativeQ` | CQL's critic term, weighted by `alpha`: it pushes the critic's values down on actions away from the data's ([details](../algorithms/sac.md#cql-and-cal-ql-offline)). | `critic_loss` | the off-policy actor-critics but TD3 |
| `CalQL` | `ConservativeQ` with the values floored at the dataset's return to go ([details](../algorithms/sac.md#cql-and-cal-ql-offline)). | `critic_loss` | SAC, REDQ, CrossQ, TQC and IQL |
| `TargetBC` | ReBRAC's critic penalty: the target less `beta` times the distance from its next actions to the dataset's ([details](../algorithms/td3.md#rebrac-offline)). | `on_target` | SAC, TD3, REDQ, CrossQ and TQC |

**Adversarial imitation** (`ajax.extensions.adversarial`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `AdversarialReward` | GAIL: a discriminator trained against the `demonstrations` dataset replaces the rollout's rewards ([details](../algorithms/ppo.md#gail-imitation-by-a-discriminator)). | `reward` | PPO with continuous actions |

**Critic ensembles** (`ajax.extensions.ensemble`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `KernelRepulsion` | Adds `coef` times the similarity of the critics' predictions to the critic loss, pushing an ensemble's critics apart ([details](../api/extensions.md#ajax.extensions.ensemble.KernelRepulsion)). | `critic_loss` | the off-policy actor-critics with two critics or more |

**Continuing tasks** (`ajax.extensions.centering`, `ajax.extensions.reset_cost`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ValueCentering` | Subtracts from the target an offset learned as the reward rate (reward centering), as its `rule` says ([details](../api/extensions.md#ajax.extensions.centering.ValueCentering)). | `on_target` | SAC, TD3, REDQ, CrossQ, TQC, AVG, IQL and DQN |
| `ResetCost` | Treats a termination as a step into the reset state, charging terminated rows a cost that rises while the termination rate exceeds `target_rate` ([details](../api/extensions.md#ajax.extensions.reset_cost.ResetCost)). | `on_target` | SAC, TD3, REDQ, CrossQ, TQC and DQN |

**Temperature annealing** (`ajax.extensions.annealing`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `MEW` | Minimum Excess Work (Adamczyk et al., 2026): sets the entropy temperature from the rewards' covariance in place of the agent's own rule, at the pace `speed` sets ([details](../api/extensions.md#ajax.extensions.annealing.MEW)). | `post_update`, `eval_metrics` | the maximum-entropy agents: SAC, REDQ, CrossQ, TQC, ASAC, RVISAC and AVG |

**Measuring the critic** (`ajax.extensions.instrumentation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CriticDiagnostics` | Logs whether the critic's features are degrading: their effective rank, dormant units and norms ([details](#measuring-a-replay-agents-critic)). | `eval_metrics` | every agent with a critic |
| `BiasVoreDecomposition` | Splits the critic's squared one-step error into squared bias and variance ([details](#measuring-a-replay-agents-critic)). | `eval_metrics` | as `CriticDiagnostics` |
| `BiasVorePenalty` | Adds a penalty on the variance of the critic's residual, set by `alpha` (0.5 adds nothing) ([details](../api/extensions.md#ajax.extensions.instrumentation.BiasVorePenalty)). | `critic_loss` | PPO, APO, DQN, PQN and the off-policy actor-critics but TQC |

**Recording representations** (`ajax.extensions.representation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `RepresentationProbe` | Every `probe_interval` steps, records the networks' activations and parameters on fixed states to `run_directory`, and logs each layer's rank and dead units ([details](representations.md)). | `pretrain`, `post_update`, `eval_metrics` | every agent whose networks read the observation, but APG and UDRL |

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
