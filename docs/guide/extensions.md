# Extensions

An extension adds a research feature to an agent's training without changing
the agent: an extra term in a loss, a change to the TD target, another way to
pick actions while collecting, an expert to learn from, a pre-training step,
or measurements logged at each evaluation. Ajax ships nineteen of them,
listed in the [catalogue](#catalogue) below.

Why extensions instead of agent options? An agent's constructor takes its
paper's hyperparameters plus `extensions=()`. The exceptions are
`expert_policy` (SAC, TD3, REDQ, AVG, APO), whose return each evaluation
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
| `on_target` | at each critic update | change the TD or value target | in list order, each sees the previous result |
| `critic_loss` | at each critic update | add a term to the critic loss | summed |
| `actor_loss` | at each actor update | add a term to the actor loss | summed |
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
| [TD3](../algorithms/td3.md), [REDQ](../algorithms/redq.md), [ASAC](../algorithms/asac.md), [AVG](../algorithms/avg.md) | yes | yes | yes | no |
| [PPO](../algorithms/ppo.md), [APO](../algorithms/apo.md) | yes (value targets) | yes | yes | no |
| [DQN](../algorithms/dqn.md), [PQN](../algorithms/pqn.md) | yes | yes (the Q-network's loss) | no | no |
| [UDRL](../algorithms/udrl.md), [APG](../algorithms/apg.md) | no | no | yes | no |
| [DreamerV3](../algorithms/dreamerv3.md), [TDMPC2, TDMPC2MultiTask](../algorithms/tdmpc2.md) | no | no | no | no |

Each agent declares these in its `supported_extension_phases`. An extension
that implements a phase the agent does not run is refused when the agent is
built, instead of being silently ignored:

```python
import jax.numpy as jnp

from ajax import DQN
from ajax.extensions.expert import ImitationLoss


def expert(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


try:
    DQN("CartPole-v1", extensions=(ImitationLoss(expert),))
except ValueError as error:
    print(error)  # DQN does not support these extension phases: ... ['actor_loss'] ...
```

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
| `on_target` | `observations`, `next_observations`, `rewards`, `gamma` | `actions`, `dones`, `reward_scale` (SAC, TD3, REDQ, ASAC, AVG, DQN, PQN); `terminated`, `truncated` (PQN, PPO, APO); `gae`, `values` (PPO, APO) |
| `critic_loss` | `observations`, `targets` | `critic_params`, `critic_state` (SAC, TD3, REDQ, ASAC, AVG, PPO, APO); `actions` (SAC, TD3, REDQ, ASAC, AVG, DQN, PQN); `q_state` (DQN, PQN) |
| `actor_loss` | `actor_params` | `actor_state` (all but APG); `observations` (all but PPO and APG); `raw_observations`, `pi_mean` (SAC, TD3, REDQ, ASAC, APO); `a_expert`, `critic_state` (SAC); `actions` (UDRL); `rollout`, `returns` (APG) |

`gamma` is `None` for an average-reward agent. `dones` marks where the
target cut the bootstrap. `targets` is the target, held constant. The critic
(`critic_params`, or DQN's and PQN's `q_state`) and the actor
(`actor_params`) are the parameters being differentiated. `pi_mean` is the
actor's mean action. `raw_observations` are the environment's own
observations, before any normalisation; a replay agent keeps them only in a
run with an expert, and passes `None` otherwise. An agent may add keys of
its own: read its fold call.

An extension that needs a key the batch lacks raises an error. It never
falls back to adding nothing, so a misplaced extension cannot look like a
result. `ImitationLoss` on PPO, for example, raises because PPO's actor
batch has no `pi_mean`; `BiasVorePenalty` raises on a critic batch without
the critic.

## What an extension declares

Besides its phases, an extension states what it needs as class attributes,
which the agent reads when it is built:

| Attribute | What it says |
| --- | --- |
| `buffer_fields` | replay fields beyond the agent's own that the extension reads. Today only `raw_next_obs`, the environment's own next observation; SAC stores it when an extension declares it, and the other agents refuse such an extension. |
| `expert_fields` | the fields holding an expert. The agent converts each one to [agent units](expert-guidance.md#the-expert), and a run whose extensions hold an expert keeps the replay fields only experts read. |
| `experts()` | the experts the extension calls itself, without a state, on the observations it is given. |
| `reads_agent_expert_actions` | the extension reads the expert's actions SAC stored at collection when its expert is SAC's own. |
| `needs_agent_expert` | the extension reads the actions of the agent's own `expert_policy` (`OnlineBC`), so an agent without one refuses it. |
| `fits_phi_star`, `reads_phi_star()` | the extension fits, or reads, the frozen expert critic `φ*` (below). |
| `reads_step_in_episode` | the extension reads each copy's step within its episode, which the agent then counts. |
| `action_slot` | where in SAC's action pipeline its `action` runs. |

`φ*` is a separate critic fitted to an expert's returns before training.
`MCPretrain` fits it (`fits_phi_star`); `OnlineBC`, `CriticBlend`,
`MCVarianceCorrection`, `ValueBox` and `PhiRefresh` read it
(`reads_phi_star()`). Without `MCPretrain` in the stack they would add
nothing for the whole run, or fail when the program is built, so the agent
refuses them. `EDGEExploration` scores with `φ*` when it exists and with the
agent's own critic otherwise, so it needs no `MCPretrain`.

## What is checked, and when

When the agent is built, its stack is refused (`ValueError`, naming each
offending extension) if an extension:

- implements a phase the agent does not run;
- reads a replay field the agent does not store (`buffer_fields`);
- reads `φ*` without an `MCPretrain` in the stack;
- reads the actions of the agent's `expert_policy` on an agent without
  one (`needs_agent_expert`);
- calls a stateful expert without its state. An extension calls its expert
  on the observations it is given, where no expert state exists. A stateful
  expert is allowed only when it is SAC's own `expert_policy` and the
  extension reads the actions SAC stored from the expert's tracked state;
- (SAC) acts in collection without an `action_slot`, or acts in collection
  or makes the policy residual in a run without an `expert_policy` or with
  `use_pid_policy`.

When training starts, an extension that takes context from the agent
(below) raises if the agent does not give it what it needs: `MCPretrain`
or `BellmanPretrain` outside SAC, `BellmanPretrain` or `ExpertValueMetrics`
without an `expert_policy`, `CloningPretrain` on an agent that does not bind
it, or placed after another pre-training extension, `PriorData` on an agent
without a replay or on a replay whose rows it cannot fill. While the program is
built, an extension raises on a batch without the keys it needs (above).

## Catalogue

The checks above are necessary but not always sufficient: the last column
says where each extension does its job. "SAC with an expert" means
`SAC(..., expert_policy=...)`.

**Learning from an expert** (`ajax.extensions.expert`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OnlineBC` | Pulls the actor's mean action towards the expert's, weighted by `φ*`'s value at each state, fading to zero over the first `critic_warmup_frac` of training. | `actor_loss` | SAC with an expert and `MCPretrain` |
| `ImitationLoss` | Adds `coef · mean((μ(s) − a_E(s))²) / 4`, the behaviour-cloning term of TD3+BC, towards an expert it holds itself: stateless, or SAC's own, whose stored actions it reads. With `q_filter=True` a row counts only where the critic rates the expert's action above the policy's (Q-filtered BC), and evaluations log the share of a replay batch it keeps (`imitation/q_filter_frac`). | `actor_loss`, `eval_metrics` | SAC, TD3, REDQ, ASAC, APO; the other actor batches lack `pi_mean` and raise. `q_filter=True`: SAC without `ResidualPolicy`; the other actor batches lack `critic_state` and raise |
| `ResidualPolicy` | Executes `clip(a_E + scale · a_π, −1, 1)`: the policy learns a correction on top of the expert. | `transform_action` | SAC with an expert, without `use_pid_policy` |
| `JSRLCurriculum` | The expert acts for the first `H_t` steps of each episode, the policy afterwards; `H_t` shrinks from `episode_length` to 0 over `decay_frac` of training. | `action` | SAC with an expert |
| `ExpertValueMetrics` | Logs the critic's value at the expert's actions (`value/q_expert_mean`), its gap to the value of the actions taken (`value/q_gap`) and, with `MCPretrain`, its gap to `φ*` (`value/phi_star_q_gap_ood`), on one replay batch per evaluation. | `eval_metrics` | SAC with an expert |

**Exploration** (`ajax.extensions.exploration`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EDGEExploration` | Lets the expert act in place of the policy, step by step, as a gate (`gate=`) decides. `"fixed"`: with probability `fixed_prob`; `"argmax"`: whenever the critic rates the expert's action higher; `"boltzmann"`: with a probability that grows with that gap. These three fade out over `decay_frac` of training. `"lcb"`, `"argmax_lcb"` and `"thompson"` also weigh the critics' disagreement. Logs `edge/live_expert_frac`. | `action`, `eval_metrics` | SAC with an expert |

**Changing the critic's target** (`ajax.extensions.target_mods`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `IBRL` | Adds `γ(1 − d) · max(Q(s′, a_E) − Q(s′, a′), 0)` to the target, so the values match acting with whichever of expert and policy looks better. | `on_target` | SAC |
| `LCBGatedBootstrap` | Blends the policy's and the expert's next-action values in the target, weighted by a sigmoid of their pessimistic (lower-confidence-bound) scores. | `on_target` | SAC |
| `CriticBlend` | Blends the target with a one-step target on `φ*`, `(1 − α) y + α y_E`, with `α` falling from 1 to 0 over `critic_warmup_frac`. | `on_target` | SAC with `MCPretrain` |
| `MCVarianceCorrection` | Where the critics disagree by more than `threshold` (variance), uses `φ*`'s value as the target. | `on_target` | SAC with `MCPretrain` |
| `ValueBox` | Lets the expert act where its value under `φ*` exceeds a threshold that rises from the lowest to the highest expert value over training. | `action` | SAC with an expert and `MCPretrain` |

**Pre-training** (`ajax.extensions.pretrain`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `CloningPretrain` | Rolls the expert out for `n_steps` steps per copy and trains the actor to reproduce its actions (`epochs` passes in batches of `batch_size`, at the actor's learning rate); the agent's observation statistics start from the dataset's. | `pretrain` | SAC, TD3, REDQ, APO (other agents raise when training starts) |
| `MCPretrain` | Rolls the expert out, fits `φ*` to its Monte-Carlo returns, records its value range, and by default nudges the agent's own critic towards it. | `pretrain` | SAC (other agents raise when training starts) |
| `BellmanPretrain` | Runs `n_steps` ordinary critic updates on the replay buffer, which holds the expert's prefill, before training. Its values bootstrap from an untrained critic; kept for ablations. | `pretrain` | SAC with an expert |
| `PhiRefresh` | Every `interval` steps, refits `φ*` for `steps` updates on the replay rows where the expert acted. | `post_update` | SAC with an expert and `MCPretrain` |
| `PriorData` | Adds a transition dataset (`ajax.buffers.dataset`) to the replay, where it stays, and to the observation and reward statistics, then runs `offline_updates` of the agent's own update on it before acting. Pair it with `learning_starts=0`. | `pretrain` | SAC, TD3, REDQ, ASAC, DQN (other agents raise when training starts) |

**Critic ensembles** (`ajax.extensions.ensemble`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `KernelRepulsion` | Adds `coef` times the mean similarity (an RBF kernel) between the critics' predictions, which pushes the critics of an ensemble apart. | `critic_loss` | SAC, TD3, REDQ, ASAC, AVG with two or more critics; the other critic batches lack its inputs and raise |

**Measuring the critic** (`ajax.extensions.instrumentation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConditioningMetrics` | Logs the critic's effective rank (`Cond/critic_srank`), its fraction of dormant units, its feature and weight norms. It finds the critic from the agent's networks: an ensemble critic, or a Q-network. | `eval_metrics` | replay agents given `buffer=` (SAC, TD3, REDQ, ASAC, DQN); PPO, APO and PQN run with `expose_recent_rollout=True` (AVG too, but its rollout is one step per copy) |
| `BiasVoreDecomposition` | Splits the critic's squared one-step error into squared bias and variance (`EVarEst/bias_sq`, `EVarEst/var_resid`, `EVarEst/mse`, `EVarEst/abs_bias`), against a one-step target on its current parameters, or your own `value_target_fn`. | `eval_metrics` | as `ConditioningMetrics` |
| `BiasVorePenalty` | Adds `((1 − 2α)/α) · Var(residual)` to the critic loss; `alpha=0.5` adds nothing. | `critic_loss` | every agent with a `critic_loss` phase: SAC, TD3, REDQ, ASAC, AVG, PPO, APO, DQN, PQN |

Sources: TD3+BC is Fujimoto and Gu, 2021; Q-filtered BC is Nair et al.,
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
filled-in copy. SAC binds everything its extensions read, which is why
`MCPretrain`, `BellmanPretrain`, `PhiRefresh` and `ExpertValueMetrics` work
with SAC only. TD3, REDQ and APO bind their environment, actor optimiser and
`expert_policy`, which is what `CloningPretrain` needs. DreamerV3 binds its
own context. The other agents bind nothing. You never call `bind_to_agent`
yourself.

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
