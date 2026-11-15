# Expert guidance

Often a controller already exists for the task: a tuned PID, a model
predictive controller, a hand-written rule, an earlier policy. Starting
reinforcement learning from scratch throws that knowledge away. Expert
guidance lets an agent learn from such an *expert*: copy it at the start,
let it act while the agent is still poor, use its values in the critic's
target, and then improve on it.

In Ajax, an expert is any function you pass to an agent or to an
[extension](extensions.md). SAC uses it the most: it fills the replay buffer
with the expert's experience and lets the expert extensions shape
collection, the critic and the actor. SAC, TD3, REDQ, ASAC, AVG and APO can
clone it before training, and those and PPO, APG and UDRL can imitate it
through an extra actor-loss term. [DAgger](../algorithms/dagger.md)
imitates it with no reward at all. Recorded transitions, from an expert or
any earlier run, can start a replay agent's buffer (`PriorData`).
[Choosing the pieces](#choosing-the-pieces) names each method, its paper and
the agents that run it.

## The expert

An expert maps observations to actions. Ajax accepts two forms:

- **Stateless:** `expert(obs) -> action`.
- **Stateful**, for a controller with memory such as a PID's integral: an
  object with `init_state(n_envs)`, which returns the starting state for
  `n_envs` environments, and `expert(state, obs) -> (action, new_state)`.
  TargetGym's PID experts (`target_gym.experts.pid.FunctionalExpertPolicy`)
  follow this form: build one with
  `target_gym.registry.REGISTRY[name].make_pid_controller().batched()`.
  `target_gym.make_expert(name)` returns TargetGym's own form, with `reset`
  and `step`, which Ajax does not call.

Four conventions matter:

1. **The expert lives in the environment's units, both ways.** It reads the
   environment's own observation, before any observation normalisation and
   without the columns an agent appends (the training fraction, the
   expert's own state). It returns actions in the environment's bounds: on
   Pendulum, a torque between -2 and 2. Ajax reads that action in the
   agent's units, `[-1, 1]` (see [Actions](environments.md#actions)),
   without clipping it, so it can be stored and compared with the agent's
   own actions; when it is executed, the usual map sends it back to the
   environment, clipped at its bounds.
2. **Its state is tracked.** Wherever Ajax runs a stateful expert itself,
   it keeps one state per environment copy and starts every episode from
   `init_state`: SAC's replay prefill, warm-up and collection, the cloning
   and Monte-Carlo rollouts, and every agent's evaluation.
3. **It works on batches.** Ajax calls it on one observation per
   environment, and on replay batches. Index the last axis (`obs[..., 0]`),
   not the first, and keep the action's last axis (`[..., None]` for one
   action; a discrete action's index is an int `(n, 1)` column).
4. **It is part of the compiled program**: a plain function or a frozen
   dataclass works.

A stateful expert, a PID on Pendulum's angle:

```python
from dataclasses import dataclass

import jax.numpy as jnp

from ajax import SAC
from ajax.logging import LoggingConfig


@dataclass(frozen=True)
class PIDExpert:
    """A PID on the pole's angle; its state is the error's integral."""

    kp: float = 2.0
    ki: float = 0.1
    kd: float = 0.5
    dt: float = 0.05  # Pendulum's time step

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, integral, obs):
        theta = jnp.arctan2(obs[..., 1:2], obs[..., 0:1])  # raw [cos, sin, velocity]
        integral = integral + self.dt * theta
        torque = -(self.kp * theta + self.ki * integral + self.kd * obs[..., 2:3])
        return jnp.clip(torque, -2.0, 2.0), integral  # newton-metres, as the env takes them


agent = SAC("Pendulum-v1", expert_policy=PIDExpert(), learning_starts=500)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=0, n_timesteps=1_000, logging_config=logging)
print(evaluations["Eval/episodic mean expert reward"])  # the PID's own return
```

Extensions that call an expert themselves, on the observations of a replay
batch (`ImitationLoss`, `IBRL`, `LCBGatedBootstrap`, `CriticBlend`,
`PhiRefresh`), call it as `expert(obs)`, on raw observations, where no
expert state exists. A stateful expert is allowed there in one case only:
when it is the agent's own `expert_policy` (the same object), on an agent
that stores that expert's actions (SAC, TD3, REDQ, TQC, CrossQ, ASAC, AVG,
APO, DAgger, DQN). The extensions
then read the actions the agent stored at collection, computed from the
expert's tracked state, instead of calling it. Any other stateful expert is
refused when the agent is built.

## Which agents take an expert

Every agent but the world models takes an `expert_policy=`: collection
tracks it along each trajectory, the extensions read it, and evaluations
log its return. DAgger requires one: the expert acts and labels every
visited state. Which expert extensions each agent runs is in
[Compatibility](compatibility.md).

`ImitationLoss` needs a policy's mean action: PPO, APG and UDRL take it with
continuous actions and a stateless expert. A discrete policy has no mean
action, DQN and PQN have no actor, and the world models fold no actor phase.

With an expert, evaluations also log `Eval/episodic mean expert reward`, the
expert's return on the same episodes as the agent's, and `Eval/expert
bias`, the agent's return minus the expert's (positive once the agent does
better). SAC takes a separate `eval_expert_policy` when the expert to
compare against is not the one to learn from.

None of this works with `memory=` yet: recurrent SAC, TD3 and REDQ refuse an
expert and extensions.

## Example

SAC on Pendulum with a PD controller that is good near the top and poor
elsewhere. Three extensions build on it: `MCPretrain` fits a critic to the
expert's returns, `OnlineBC` pulls the actor towards the expert early on, and
`EDGEExploration` lets the expert act while the critic rates it higher than
the policy.

```python
import jax.numpy as jnp

from ajax import SAC
from ajax.extensions.expert import OnlineBC
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import ExpertPrefill, MCPretrain
from ajax.logging import LoggingConfig


def pd_expert(obs):
    """A PD controller on Pendulum's raw observation [cos θ, sin θ, θ̇]."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    torque = -(2.0 * theta + 0.5 * obs[..., 2])
    return jnp.clip(torque, -2.0, 2.0)[..., None]  # the env's torque range


agent = SAC(
    "Pendulum-v1",
    expert_policy=pd_expert,
    learning_starts=1_000,
    extensions=(
        # 1,000 expert steps written to the replay before training.
        ExpertPrefill(n_steps=1_000),
        # The frozen expert critic, fitted to ~1,000 steps of the expert's returns.
        MCPretrain(pd_expert, n_mc_steps=100, n_mc_episodes=10, n_steps=200),
        OnlineBC(bc_coef=1.0, critic_warmup_frac=0.2),
        EDGEExploration(gate="argmax", decay_frac=0.3),
    ),
)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
print(evaluations["Eval/expert bias"])       # agent minus expert
print(evaluations["edge/live_expert_frac"])  # share of envs the expert drove
```

At this budget the agent has barely started; the example shows the wiring.
Give `MCPretrain` a few hundred thousand expert steps and train for tens of
thousands of steps for a real run (its defaults collect about a million).

## What SAC does with an expert, step by step

### Before training

1. **Replay prefill**, by extension. `ExpertPrefill(n_steps)` runs the
   expert for `n_steps` steps in each environment and writes them to the
   replay buffer, flagged as expert rows. Every row also stores the expert's
   action at `s` and `s′`, which several extensions read later. Under gain
   mode a row stores the anchor gains (action 0) as the action, as
   collection does when the expert acts as itself. `offline_updates` runs the
   agent's own update on them before it acts. By default the expert is the
   agent's, which collection also calls on every step; `expert=` gives the
   prefill one of its own, and an arm with no expert online queries none.
2. **Cloning**, by extension. `CloningPretrain(expert, n_steps=...)` rolls
   the expert out for `n_steps` steps per environment, in the observation
   layout the agent sees, and trains the actor to reproduce its actions for
   `epochs` passes in batches of `batch_size`, at the actor's learning rate.
   The actor learns on observations standardised with the dataset's
   statistics, so the agent's running observation statistics start from
   them (an agent that kept none gets them), and the actor keeps seeing
   inputs as it did when cloned. `reset_log_std=True` restores the cloned
   actor's exploration noise afterwards, `reset_actor_head=True` keeps only
   its hidden layers, and `skip_actor=True` leaves the actor as it was. Put
   it first among the pre-training extensions: the critic pre-trained by the
   others reads the statistics it sets, and SAC refuses it after them. TD3,
   REDQ and APO take it too.
3. **Critic pre-training**, by extension. `MCPretrain` runs the expert, fits a
   separate critic, the *frozen expert critic* `φ*`, to its Monte-Carlo
   returns, and records the range of values it predicts. Unlike a Bellman
   target, a Monte-Carlo return does not lean on an untrained critic, so it
   is unbiased. `BellmanPretrain` instead runs ordinary critic updates on the
   prefilled buffer (it needs `ExpertPrefill` or `PriorData` in the stack);
   it is kept for ablations.

### Warm-up

Until `learning_starts`, SAC, TD3, REDQ, ASAC, CrossQ and IQL do not use
their policy. With `ExpertWarmup(fraction)` (`ajax.extensions.expert`), at
each step, with probability `fraction`, every environment takes the
expert's action; otherwise, a uniform random one. The replay flags the rows
the expert took, where it keeps the expert flags. Under gain mode, or with
`ResidualPolicy`, the warm-up draws the policy's actions, and the
expert's rows store the one that executes the expert's own action (0).
Without it the warm-up is uniform.

### Collection

Collection extensions decide, at each step, whether the expert or the policy
acts. The shared collector runs them around the agent's own action choice,
on every off-policy agent with an expert (SAC, TD3, REDQ, TQC, CrossQ, ASAC,
AVG, IQL, DAgger, DQN, PQN), and UDRL's collector too, in two places, which
each extension names in its `action_slot`:

- **Before the warm-up choice** (`"pre_warmup"`): `EDGEExploration` (a
  value-based gate, comparing the expert's action with the one the step
  would execute, which the critics are trained on) and `JSRLCurriculum`
  (the expert starts each episode, then the learner takes over). Its
  `schedule` sets the expert's steps `h`, from the task's horizon down to 0
  in `n_stages` stages. `"performance"` is the paper's JSRL-Curriculum: `h`
  moves down a stage when an evaluation of the combined policy passes, every
  `eval_every` steps, on `eval_episodes` episodes of its own (1 by default, as
  they are plant steps, which `ajax.ledger` charges). It passes when
  the moving average of the last `moving_average` evaluations clears
  `threshold` or, without one, comes within `tolerance` of the best previous
  moving average. `"random"` is the paper's JSRL-Random: `h` drawn uniformly
  from those stages each episode. `"time"` is Ajax's own and not the paper's:
  `h` falls linearly to 0 at `decay_frac` of training, whatever the
  performance.
- **After it** (`"post_warmup"`): `ValueBox`, which lets the expert act
  wherever its value exceeds a threshold that rises over training. The
  step that enters this region ends the agent's episode in the stored
  transition and pays the expert's value as a reward, in place of the
  bootstrap, whatever its sign.

Where an extension hands the expert a step, UDRL's segment keeps the
choice that executes the expert's action (the expert's own, or a residual
of 0), which its actor then fits under the command the step realised:
Ajax's composition of JSRL and EDGE with UDRL, not a published pairing.

`ResidualPolicy` changes what the policy's action means: the agent executes
`clip(a_E + scale · a_π, −1, 1)`, in agent units, in collection (the warm-up
too, which draws uniform residuals around the expert), in the critics'
target, in the actor's loss and in evaluation; the replay stores the
executed action. The policy then learns a correction to the expert
(residual RL). `zero_start=True` puts the actor's output head at 0, so that
the policy starts as the expert (Silver et al., 2018).
`critic_reads_residual=True` stores the residual and has the critics read it,
as both Johannink et al. and Silver et al. do; it needs a Q critic and refuses
what puts an expert's action where a residual goes. SAC, TD3, REDQ, TQC, CrossQ, ASAC and AVG
run it, their critics scoring the executed action; PPO and APO too, their
ratio reading the policy's own sample; and IQL, whose replay also keeps the
residual the policy chose, which its actor fits. Where a
collection extension handed the agent's expert the step, IQL's replay keeps
the residual that executes the expert's action, 0, which is Ajax's own
composition of JSRL, EDGE and IBRL with residual IQL. UDRL
acts through it in its own collector, keeping the policy's action, and APG
in its differentiable rollout, whose gradient runs through the residual
and the expert.

Steps where the expert acted are flagged in the buffer;
`replay/expert_share` logs, at each evaluation, their share of the rows the
replay holds, for every replay agent that keeps the flags. The
replay keeps the raw observations and these flags only in a run with an
expert (an agent's `expert_policy`, or one an extension holds); without one
nothing reads them.

### Critic

Target extensions change what the critic learns towards. `IBRL` and
`LCBGatedBootstrap` compare the expert's and the policy's next actions.

`IBRL` computes its target as its authors' code does (hengyuan-hu/ibrl):
the policy's next action is the one the agent's target scored (TD3's
smoothed target action, the sample SAC's, REDQ's and TQC's targets valued).
REDQ's target takes the minimum over a random subset of its critics; there
`IBRL` selects between the two actions on a second random subset of the
same size and values the selected one on the target's, as the authors'
code selects on one random pair and values on another, so the bonus can be
negative. The rest has no original form and is Ajax's own: the entropy
term of SAC's, REDQ's and TQC's targets stays at the policy's action when
the expert is selected, TQC's bonus reads its quantiles' means, and IQL,
whose target scores no next action, scores a sample of its actor.

`IBRL` also acts as the authors' code acts (`hybrid_acting=False` leaves
the target alone): past the warm-up it executes the better of the agent's
action (TD3's noisy one, a sample) and the expert's, each scored by the
minimum over the target critics (on REDQ, over a fresh random subset of the
target's size), and the replay flags the rows the expert's action took.
Evaluation makes the same choice from the agent's evaluation action, its
mean. It uses its own expert, which need not be the agent's.
`ibrl/expert_frac` and `Eval/expert share` log how often the expert's action
was taken. IQL under a residual policy takes it with the agent's expert
only: a residual reaches clip(a_E + residual), and another expert's action
need not lie in that reach. `tests/conformance/test_ibrl_original.py` checks the target and the
acting against the authors' code on fixed inputs.

The choice between the two actions is hard by default (the authors'
`ibrl`). `IBRL(expert_policy, soft_choice=True)` makes it their
`ibrl_soft`, in acting and in the target alike: the expert's action is
taken with probability `sigmoid(β (Q(a_E) − Q(a)))`, `β` being
`soft_choice_beta` (10 by default, as in their configurations). The target
can then draw an expert's action worth less than the policy's, so its bonus
can be negative. The draw happens in evaluation too, as in the authors'
code, so evaluation can switch between the two actions.

`LCBGatedBootstrap` scores the same next actions and blends minima over
the same critics.
`CriticBlend` and `MCVarianceCorrection` lean on `φ*`, so they need
`MCPretrain` in the same stack. `PhiRefresh` refits `φ*` during training on
the rows where the expert acted.

### Actor

`OnlineBC` adds a cloning term to the actor's loss, weighted by `φ*`'s value
at each state and fading out over `critic_warmup_frac` of training. It
needs `MCPretrain`.

`ImitationLoss(expert, q_filter=True)` adds the unweighted cloning term
(see [below](#imitation-for-td3-redq-asac-and-apo)) only on the rows where
the agent's critics rate the expert's action above the policy's mean action,
`min_k Q_k(s, a_E) > min_k Q_k(s, μ(s))`: the Q-filter of Nair et al.
(ICRA 2018, Sec. IV-C). Where the agent already does better than the
expert, it stops copying it, with no schedule to tune: a PID that is good
near the setpoint and poor in transients keeps guiding only where it still
helps. Unlike the paper, the rows are replay rows labelled with the
expert's action, not a separate demonstration buffer, and `coef` weights
the cloning term in place of the paper's two weights. The filter needs a
critic that ranks actions, a Q critic: SAC, TD3, REDQ, ASAC and AVG have
one, PPO and APO a state-value critic, which cannot rank actions. The
minimum runs over all the agent's critics, though TD3's actor follows its
first and REDQ's the mean of its ensemble. Pair it with the expert
prefill, as the paper
pairs it with demonstrations. With `ResidualPolicy`, whose critic reads
the executed action rather than the policy's, it is refused.

### Observation

Observation features append columns to every observation, which the actor,
the critics and the replay all read (`ajax.extensions.observe`):

- `ExpertActionInput()` appends the expert's action, so the policy can
  condition on the expert's suggestion. It stays raw: it is an action. A
  discrete expert's index enters one-hot.
- `ExpertStateInput.for_expert(expert)` appends a stateful expert's internal
  state (a PID's integral, say).
- `TrainingProgressInput()` appends the fraction of training done; an
  evaluation holds the collector's.

They join an observation when it arrives, in collection and evaluation, and
are stored with it.

### Evaluation

Besides the expert's return and the bias above, `ExpertValueMetrics` logs
the critic's value at the expert's actions (`value/q_expert_mean`) and its
gap to the value of the actions taken (`value/q_gap`). `EDGEExploration` logs
`edge/live_expert_frac`, and `ImitationLoss(q_filter=True)` the share of a
replay batch its filter keeps (`imitation/q_filter_frac`; not on AVG, which
keeps no replay).

## Choosing the pieces

The agents in brackets are those that accept each piece, as
[Compatibility](compatibility.md) lists them from the code. "Developed with
Ajax" marks a feature with no external reference.

| You want to | Use | Method |
| --- | --- | --- |
| Start the replay with good experience | `ExpertPrefill(n_steps)` (SAC, REDQ, ASAC, TQC, TD3, CrossQ, IQL, DQN); `PriorData` with a [transition dataset](../api/buffers.md) (SAC, TD3, REDQ, ASAC, TQC, CrossQ, IQL, DQN) | expert prefill; prior data in the replay as RLPD (Ball et al., 2023), without its symmetric sampling, and optional offline updates |
| Keep good experience for the whole run | `RetainedDemonstrations(n_steps)`, with `CloningPretrain` for a BC start (SAC, TD3, REDQ, ASAC, RVISAC, TQC, CrossQ, IQL, DQN) | DDPGfD (Vecerik et al., 2017); BC start plus prefill (Bhatt et al., 2026) |
| Learn online with half of each batch from recorded data | `RLPD` on LayerNorm critics (SAC, TD3, REDQ, ASAC, RVISAC, TQC, CrossQ, IQL, DQN) | RLPD (Ball et al., 2023) |
| Clone the expert, warm up a fresh critic, then fine-tune | `porl(CloningPretrain(expert))`, the keyword arguments of SAC or REDQ | PORL (Xiao et al., 2025) |
| Train on recorded transitions alone, then fine-tune online | `agent.train_offline(dataset, n_updates)` (SAC, TD3, REDQ, TQC, CrossQ, ASAC, RVI-SAC, DQN, IQL, DAgger) | offline training on the agent's own update; IQL (Kostrikov et al., 2022) |
| Keep the actor near the dataset's actions | `ImitationLoss(None, target="replay", alpha=2.5)` (SAC, TD3, REDQ, ASAC) | TD3+BC (Fujimoto and Gu, 2021) on TD3 |
| Keep the critic's target near the dataset's actions too | `TargetBC` with TD3+BC's term (TD3) | ReBRAC (Tarasov et al., 2023) |
| Keep the critic low on actions the dataset lacks | `ConservativeQ` (SAC, REDQ, ASAC, AVG, IQL), `CalQL` (SAC, REDQ, IQL) | CQL (Kumar et al., 2020), Cal-QL (Nakamoto et al., 2023) |
| Imitate demonstrations, with no reward and no expert | `AdversarialReward` with a transition dataset (PPO, continuous actions) | GAIL (Ho and Ermon, 2016) |
| Start the actor as a copy of the expert | `CloningPretrain` (SAC, TD3, REDQ, ASAC, AVG, APO, DAgger) | behaviour cloning |
| Imitate the expert, with no reward | the `DAgger` agent | DAgger (Ross, Gordon and Bagnell, 2011); `ImitationLoss` with a replay agent's own expert is its aggregation plus the RL loss |
| Keep the actor near the expert early on | `ImitationLoss` (SAC, TD3, REDQ, ASAC, AVG, APO, DAgger; PPO, APG, UDRL with a stateless expert), `OnlineBC` (SAC) | TD3+BC's actor term (Fujimoto and Gu, 2021), kickstarting (Schmitt et al., 2018) with `anneal_frac`; `OnlineBC` developed with Ajax |
| Copy the expert only where the critic rates it higher | `ImitationLoss(q_filter=True)` (SAC, TD3, REDQ, ASAC, AVG) | Q-filtered behaviour cloning (Nair et al., 2018) |
| Let the expert act while the agent is poor | `JSRLCurriculum` (SAC, TD3, REDQ, TQC, CrossQ, ASAC, AVG, IQL, DAgger, DQN, PQN, UDRL), `EDGEExploration` (the same with continuous actions; on UDRL its fixed gate alone, UDRL having no Q critic), `ValueBox` (SAC) | jump-start RL (Uchendu et al., 2023); EDGE and the value box developed with Ajax |
| Blend with the expert where the critics disagree | `CHEQ` (SAC, TD3, REDQ, ASAC, RVISAC, CrossQ, IQL) | CHEQ (Cramer et al., 2024) |
| Learn a correction to the expert | `ResidualPolicy` (SAC, TD3, REDQ, TQC, CrossQ, ASAC, AVG, PPO, APO, IQL, APG, UDRL) | residual RL (Johannink et al., 2019; Silver et al., 2018) |
| Move the target the expert reads | `OuterLoopReference` | outer-loop control (Arroyo and Giraldo, 2020) |
| Schedule the expert's gains each step | `ExpertGains` on a `GainScheduledLaw` | control-informed RL (Bloor et al., 2024) |
| Keep the expert's memory true to what ran | `BumplessTracking` | the conditioning technique (Hanus et al., 1987) |
| Give the critic a head start | `MCPretrain`, then `CriticBlend`, `MCVarianceCorrection`, `PhiRefresh` (SAC) | developed with Ajax |
| Make the target account for the expert | `IBRL` (SAC, TD3, REDQ, AVG), `LCBGatedBootstrap` (SAC, TD3, REDQ) | IBRL (Hu et al., 2023); the LCB gate developed with Ajax |
| Measure how the critic rates the expert | `ExpertValueMetrics` (SAC, TD3, REDQ, TQC, CrossQ, ASAC, IQL, DQN) | |

Some pieces need others. The agent refuses wrong combinations when it is
built:

| Extension | Needs | Refused |
| --- | --- | --- |
| `EDGEExploration`, `JSRLCurriculum`, `ValueBox` | an off-policy agent with an `expert_policy`, without gain mode (`ExpertGains`); `EDGEExploration` continuous actions; on IQL with a residual policy, `IBRL` acting with the agent's expert only | when built |
| `ResidualPolicy` | SAC (without gain mode), TD3, REDQ, TQC, CrossQ, ASAC, AVG, PPO, APO, IQL, APG, UDRL, with an `expert_policy` | when built |
| `OnlineBC`, `CriticBlend`, `MCVarianceCorrection`, `ValueBox`, `PhiRefresh` | `MCPretrain` in the stack | when built |
| `OnlineBC` | SAC with an `expert_policy`, whose actions it clones | when built |
| `IBRL`, `LCBGatedBootstrap` | a discounted agent that keeps the raw next observations they read; `LCBGatedBootstrap` also a critic ensemble | when built |
| `BellmanPretrain` | the agent's replay, filled by `ExpertPrefill` or `PriorData` | when built |
| `ExpertValueMetrics` | an `expert_policy` and the agent's replay | when built |
| `ImitationLoss(q_filter=True)` | a Q critic (SAC, TD3, REDQ, ASAC, AVG); without `ResidualPolicy` | when built; with `ResidualPolicy`, when training starts |
| `CloningPretrain` | SAC, TD3, REDQ, ASAC, AVG or APO; first among the pre-training extensions | the agent when built, the order when training starts |

The extensions that hold an expert themselves (`CloningPretrain`,
`MCPretrain`, `PhiRefresh`, `IBRL`, `LCBGatedBootstrap`, `CriticBlend`,
`ImitationLoss`) take it as their first argument. Pass the same object you
give the agent. The target extensions, `PhiRefresh` and `ImitationLoss`
then read the actions the agent stored at collection rather than calling
the expert again.
`CloningPretrain` must have it when SAC's observations carry the expert's
action or state, and `MCPretrain` when they carry its state; they are
refused otherwise.

## Imitation for TD3, REDQ, ASAC and APO

These agents take `ImitationLoss`, the behaviour-cloning term of TD3+BC
(Fujimoto and Gu, 2021) with a fixed expert as the target:
`coef · mean((μ(s) − a_E(s))²) / 4`, where `μ(s)` is the actor's mean
action and `a_E(s)` the expert's, both in agent units. Dividing by 4 maps
the squared distance of two actions in `[-1, 1]` to `[0, 1]`. Unlike
`OnlineBC`, it is unweighted and needs no expert critic. Give the agent the
same expert as its `expert_policy` to use a stateful one: the term then
reads the actions the agent stored along each trajectory. With
`anneal_frac=f`, the term's weight falls linearly from 1 to 0 at the
fraction `f` of training, as in kickstarting (Schmitt et al., 2018), so the
agent ends on its own objective; kickstarting uses a cross-entropy to a
stochastic teacher and tunes the schedule, where here the expert is
deterministic and the schedule linear. Its Q-filter (`q_filter=True`, see
[Actor](#actor)) runs on TD3, REDQ and ASAC too; APO's critic rates
states, not actions, so it refuses the filter. These agents can also start from a
clone of the expert:

```python
import jax.numpy as jnp

from ajax import TD3
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.pretrain import CloningPretrain


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = TD3(
    "Pendulum-v1",
    expert_policy=pd_expert,  # logs the expert's own return at evaluations
    learning_starts=500,
    extensions=(
        CloningPretrain(pd_expert, n_steps=1_000, epochs=2),
        ImitationLoss(pd_expert, coef=1.0),
    ),
)
state, _ = agent.train(seed=0, n_timesteps=2_000)
```

## Learning a PID's gains: gain mode

With a stateful PID expert that exposes `learnable_fields`, `anchor_gains`
and `step_with_gains` (TargetGym's `FunctionalExpertPolicy` after
`register_learnable_gains`), the `ExpertGains` extension
(`ajax.extensions.expert`) makes an agent schedule the expert's gains
instead of acting directly: SAC, TD3, REDQ, TQC, CrossQ, ASAC, AVG, PPO, APO, IQL, APG and UDRL run it with the expert as their
`expert_policy`. The
actor outputs one number `a` in `[-1, 1]` per learnable gain; the gain
applied is `anchor · max_factor^a`, by default between a tenth and ten times
the expert's own.
The PID computes the action. The replay, or the rollout, holds the gains
(the agent's actions, as many as the learnable gains), so the critics score
them and the warm-up draws uniform gains; the expert's own action is stored
as `a_expert`, as wide as the env's action. An agent refuses this mode, when
it is built, with `ResidualPolicy` (one transform a run), `ExpertActionInput`
(the expert's action does not exist before the actor has chosen the gains),
the extensions using an expert's actions where the policy's go
(`ImitationLoss` cloning its expert, `OnlineBC`, `IBRL`, `CloningPretrain`,
`MCPretrain` and the like: env actions against gains, not wired yet), and
SAC with its collection extensions (their `action` phase never runs in it).

It is different from the [PID policy](networks.md#the-pid-policy), in which
the actor itself is a PID whose gains are learned, for any
continuous-action agent.

## Acting around a controller

`ajax.extensions.structure` holds three arms that keep the expert and change what
is around it. Each asks only for what it uses: the expert's call
`expert(memory, obs)`, which columns of the observation hold its targets and its
readings, and its gains. TargetGym's batched controllers have that form, and its
envs name the columns in `obs_target_index` and `obs_value_index`. Read them from
the env rather than writing positions, since observation layouts change between
versions.

- **Moving the target.** `OuterLoopReference(target_channels, max_change)` with the
  expert as `expert_policy`: the policy outputs one change per target, bounded by
  `max_change` in the target's units, and the expert acts on the moved target. No
  change is the expert.
- **Scheduling gains.** `GainScheduledLaw(expert, obs_width)` as `expert_policy`,
  with `ExpertGains(max_factor=2.0)`: the policy sets a multiplier per gain, and the
  multipliers move at most `max_log_rate` a step. The action changes as the law does
  between two steps at the same gains, the velocity form, so a new gain alone moves
  nothing, and unit multipliers are the law. Add
  `ExpertStateInput.for_expert(layer)`, so that the policy reads the multipliers it
  moves from.
- **Tracking what ran.** Add `BumplessTracking(reading_channels)` to a switching or
  blending arm (`JSRLCurriculum`, `CHEQ`, `IBRL`, `ResidualPolicy`): the expert's
  memory takes the error at which it would have output the action applied, which an
  env reports in `info["applied_action"]`, and otherwise the command. Without it, the
  memory advances on the error whoever acts. A step the expert chose itself stays its
  own, and a law sitting at its own limit keeps its own anti-windup. Handing back
  still steps by the proportional term.

`GainScheduledLaw` reads the expert's `step_with_gains`, `learnable_fields` and
`anchor_gains`. TargetGym's batched form has them for the laws it registers. For the
others, `ExpertLaw` builds them from the law:

```python
# doc: skip
from target_gym.registry import REGISTRY

from ajax.extensions.structure import ExpertLaw

law = ExpertLaw.of(
    REGISTRY["four_tank"].make_pid_controller(),
    gain_names=("Kp_1", "Ti_1", "Kp_2", "Ti_2"),
)
```

## Limits and caveats

- **No memory.** Recurrent SAC, TD3, REDQ and ASAC refuse an expert and
  extensions.
- **SAC is the hub.** Most expert extensions read quantities only SAC's
  update provides; the [catalogue](extensions.md#catalogue) gives each one's
  scope.
- **Stateless queries.** An extension that calls its expert on replayed
  observations needs a stateless expert, unless it is the agent's own on an
  agent that stores its actions (see [The expert](#the-expert)).
- **Budgets add up.** The prefill, cloning and `MCPretrain` collections run
  before the first training step and are not counted in `n_timesteps`.

## See also

- [Extensions](extensions.md): the phases and the full catalogue.
- [SAC](../algorithms/sac.md): every SAC option.
- [Networks and the PID policy](networks.md).
- [Expert guidance on Pendulum](../reproduction/expert-guidance.md): four of
  these extensions against SAC alone, with a swing-up controller as the expert.
