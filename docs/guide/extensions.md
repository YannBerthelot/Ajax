# Extensions

An extension adds a research feature to an agent's training without changing
the agent: an extra term in a loss, a change to the TD target, another way to
pick actions while collecting, an expert to learn from, or measurements
logged at each evaluation. Ajax ships nineteen of them, listed in the
[catalogue](#catalogue) below.

Why extensions instead of agent options? An agent's constructor takes its
paper's hyperparameters plus `extensions=()`. The exceptions are a few expert
options (`expert_policy`, cloning, and SAC's
[expert options](../algorithms/sac.md#expert-options)), kept because they
change the network's input, the collection or the action pipeline, which
extensions do not reach yet. A feature
written into an agent grows that agent's file and ties it to one experiment;
SAC once had about a hundred constructor keywords that way. Kept apart, a
feature works with every agent that runs the parts of training it touches,
and each agent's code stays readable next to its paper.

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
  Two extensions with equal settings compare equal, so a second agent built
  the same way reuses the compiled program.
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
| `pretrain` | once, before training, on a fresh run | change the agent's state (for example, pre-train a critic) | in list order |
| `on_target` | at each critic update | change the TD or value target | in list order, each sees the previous result |
| `critic_loss` | at each critic update | add a term to the critic loss | summed |
| `actor_loss` | at each actor update | add a term to the actor loss | summed |
| `action` | at each collection step | replace the action the agent would take | the last proposal wins |
| `eval_action` | at evaluation | replace the evaluation action | no agent runs it yet |
| `post_update` | after each update | change the agent's state (schedules, refreshes, snapshots) | in list order |
| `eval_metrics` | at each evaluation | add metrics to the log | merged |

## Which agent runs which phase

Every agent runs `pretrain`, `post_update` and `eval_metrics`, which belong to
the shared training loop. The others depend on what the agent's update has: a
target, a critic, an actor, a collection step it lets an extension change.

| Agent | `on_target` | `critic_loss` | `actor_loss` | `action` |
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

- **`action` is SAC's alone**, and only with an `expert_policy`: SAC's
  collection step offers two places for an extension to act, before and after
  its warm-up choice, and an extension must say which one (its
  `action_slot`). [Expert guidance](expert-guidance.md) explains them.
- **Memory and extensions do not mix yet.** With `memory=`, PPO, SAC, TD3,
  REDQ and ASAC raise when training starts if you also pass extensions. APG
  takes both.

## Catalogue

The phase check above is necessary but not always sufficient. Each phase
hands the extension a dictionary of the quantities at that point, and the
keys differ by agent. Most expert extensions read quantities only SAC
provides. The last column says where each extension does its job; "SAC with
an expert" means `SAC(..., expert_policy=...)`.

**Learning from an expert** (`ajax.extensions.expert`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `OnlineBC` | Pulls the actor's mean action towards the expert's, weighted by the expert's value at each state, fading to zero over the first `critic_warmup_frac` of training. | `actor_loss` | SAC with an expert and `MCPretrain`; without `MCPretrain` it adds nothing |
| `ImitationLoss` | Adds `coef · mean((μ(s) − a_E(s))²) / 4`, the behaviour-cloning term of TD3+BC, towards a fixed expert it holds itself. | `actor_loss` | TD3, REDQ, ASAC, APO; the other agents' actor batches lack the mean action and raise |
| `ResidualPolicy` | Executes `clip(a_E + scale · a_π, −1, 1)`: the policy learns a correction on top of the expert. | none (SAC reads it) | SAC with an expert, without `use_pid_policy`; other agents ignore it |
| `JSRLCurriculum` | The expert acts for the first `H_t` steps of each episode, the policy afterwards; `H_t` shrinks from `episode_length` to 0 over `decay_frac` of training. | `action` | SAC with an expert and `jsrl_curriculum=True` |
| `ExpertValueMetrics` | Logs the critic's value at the expert's actions (`value/q_expert_mean`) and its gap to the value of the actions taken (`value/q_gap`). | `eval_metrics` | SAC with an expert |

**Exploration** (`ajax.extensions.exploration`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `EDGEExploration` | Lets the expert act in place of the policy, step by step, as a gate (`gate=`) decides. `"fixed"`: with probability `fixed_prob`; `"argmax"`: whenever the critic rates the expert's action higher; `"boltzmann"`: with a probability that grows with that gap. These three fade out over `decay_frac` of training. `"lcb"`, `"argmax_lcb"` and `"thompson"` also weigh the critics' disagreement. Logs `edge/live_expert_frac`. | `action`, `eval_metrics` | SAC with an expert |

**Changing the critic's target** (`ajax.extensions.target_mods`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `IBRL` | Adds `γ(1 − d) · max(Q(s′, a_E) − Q(s′, a′), 0)` to the target, so the values match acting with whichever of expert and policy looks better. | `on_target` | SAC |
| `LCBGatedBootstrap` | Blends the policy's and the expert's next-action values in the target, weighted by a sigmoid of their pessimistic (lower-confidence-bound) scores. | `on_target` | SAC |
| `CriticBlend` | Blends the target with a one-step target on the frozen expert critic, `(1 − α) y + α y_E`, with `α` falling from 1 to 0 over `critic_warmup_frac`. | `on_target` | SAC with `MCPretrain` |
| `MCVarianceCorrection` | Where the critics disagree by more than `threshold` (variance), uses the frozen expert critic's value as the target. | `on_target` | SAC with `MCPretrain` |
| `ValueBox` | Lets the expert act where its value exceeds a threshold that rises from the lowest to the highest expert value over training. | `action` | SAC with an expert and `MCPretrain` |

**Pre-training a critic** (`ajax.extensions.pretrain`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `MCPretrain` | Rolls the expert out, fits a separate critic (the frozen expert critic, `φ*`) to its Monte-Carlo returns, records its value range, and by default nudges the agent's own critic towards it. | `pretrain` | SAC (other agents raise: they do not supply what it needs) |
| `BellmanPretrain` | Runs `n_steps` ordinary critic updates on the replay buffer, which holds the expert's prefill, before training. Its values bootstrap from an untrained critic; kept for ablations. | `pretrain` | SAC with an expert |
| `PhiRefresh` | Every `interval` steps, refits the frozen expert critic for `steps` updates on the replay rows where the expert acted. | `post_update` | SAC with an expert and `MCPretrain` |

**Critic ensembles** (`ajax.extensions.ensemble`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `KernelRepulsion` | Adds `coef` times the mean similarity (an RBF kernel) between the critics' predictions, which pushes the critics of an ensemble apart. | `critic_loss` | SAC, TD3, REDQ, ASAC, AVG with two or more critics; PPO, APO, DQN and PQN accept it but their critic batches lack its inputs and fail |

**Measuring the critic** (`ajax.extensions.instrumentation`)

| Extension | What it does | Phases | Works with |
| --- | --- | --- | --- |
| `ConditioningMetrics` | Logs the critic's effective rank (`Cond/critic_srank`), its fraction of dormant units, its feature and weight norms. | `eval_metrics` | PPO and APO run with `expose_recent_rollout=True`; AVG too with `concat_action=True`, PQN with `critic_loc="actor"`; replay agents given `buffer=` |
| `BiasVoreDecomposition` | Splits the critic's squared one-step error into squared bias and variance (`EVarEst/bias_sq`, `EVarEst/var_resid`, `EVarEst/mse`, `EVarEst/abs_bias`). | `eval_metrics` | SAC given `buffer=`; PPO and APO run with `expose_recent_rollout=True`; other agents through your own `value_target_fn` |
| `DiagnosticSnapshots` | Every `every_n_iters` updates, pickles a sample of transitions and the actor's parameters to `directory`, for analysis after the run. | `post_update` | agents with an actor, given `buffer=` or run with `expose_recent_rollout=True` (tested on PPO) |
| `BiasVorePenalty` | Adds `((1 − 2α)/α) · Var(residual)` to the critic loss; `alpha=0.5` adds nothing. | `critic_loss` | SAC, PPO, APO, DQN, PQN; TD3, REDQ, ASAC and AVG pass no target to it, so it adds nothing there |

Sources: TD3+BC is Fujimoto and Gu, 2021; residual RL is Johannink et al.,
2019; JSRL is Uchendu et al., 2023; the kernel's bandwidth follows the median
heuristic of SVGD (Liu and Wang, 2016); effective rank is Kumar et al., 2021,
and dormant units Sokar et al., 2023 (all cited in the code). IBRL is Hu et
al., *Imitation Bootstrapped Reinforcement Learning*, 2023. EDGE, the LCB
gates and the bias-variance (EVarEst) measurements are research features
developed with Ajax, with no external reference in the code.

## Measuring a replay agent's critic

`ConditioningMetrics`, `BiasVoreDecomposition` and `DiagnosticSnapshots`
need states the agent has visited. An on-policy agent keeps its latest rollout
when built with `expose_recent_rollout=True`, as in the example at the top of
this page. A replay agent keeps them in its replay buffer, and the extension
must be given the buffer object. The extension is built before the agent, so
take the buffer from an agent built with the same settings (`n_envs`,
`buffer_size`, `batch_size`):

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
settings, replay buffer or discount. They receive them through
`bind_to_agent`, which an agent calls on its whole stack when it builds its
training program; each extension picks the arguments it needs and returns a
filled-in copy. SAC binds `MCPretrain`, `BellmanPretrain`, `PhiRefresh` and
`ExpertValueMetrics`; DreamerV3 also calls `bind_to_agent`, with its own
context. The other agents do not, which is why those four extensions work
with SAC only. You never call `bind_to_agent` yourself.

## Writing your own

An extension is a frozen dataclass subclassing
[`Extension`](../../src/ajax/extensions/base.py) that overrides the phases it
needs. [Writing an extension](../developer/writing-an-extension.md) walks
through the contract, the dictionary each phase receives on each agent, and
a complete example. If a feature seems to need a new keyword on an agent,
that is the sign it should be an extension instead.

## See also

- [Expert guidance](expert-guidance.md): how the expert extensions fit
  together with SAC's expert options, with an example.
- [Extensions API](../api/extensions.md): every extension's settings.
- [`ajax/extensions/`](../../src/ajax/extensions): the source, one module per
  family.
