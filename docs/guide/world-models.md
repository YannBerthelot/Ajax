# World models

Most agents in Ajax are *model-free*: they learn values and a policy directly
from the experience they collect. A *world model* agent first learns a model
of the environment, a network that predicts what happens next and what
reward follows, and then uses that model to decide how to act. Learning the
model reuses every transition many times, so these agents tend to need fewer
environment steps; the price is more computation per step.

Ajax has two world-model agents, each reproducing its paper's official code:

- **[DreamerV3](../algorithms/dreamerv3.md)** (Hafner, Pasukonis, Ba and
  Lillicrap, *Mastering Diverse Domains through World Models*, 2023; Nature,
  2025) learns its policy *inside* the model, by imagination.
- **[TD-MPC2](../algorithms/tdmpc2.md)** (Hansen, Su and Wang, *TD-MPC2:
  Scalable, Robust World Models for Continuous Control*, ICLR 2024) *plans*
  with the model at every step.

## DreamerV3 in plain words

DreamerV3's world model is recurrent: it compresses everything seen so far in
the episode into a latent state, part deterministic memory, part a set of
discrete random variables. From that state it learns to reconstruct the
observation, predict the reward and predict whether the episode continues.
It trains on sequences replayed from a buffer.

The actor learns only from imagined experience. Starting from states the
model has encoded, the model imagines 15 steps ahead under the current
actor; the critic learns the value of those imagined trajectories (plus a
smaller term on the replayed ones), and the actor learns to make them
better. The model, the actor
and the critic take one joint gradient step per update.

A few choices make one set of hyperparameters work across very different
tasks: rewards and values are predicted on a log-like (`symlog`) scale over
fixed bins, returns are normalised by their spread, and the optimiser clips
gradients relative to the size of the weights. Because the model carries
memory, DreamerV3 handles partially observed tasks without a `memory=`
option.

## TD-MPC2 in plain words

TD-MPC2's world model has no decoder: it never reconstructs observations. An
encoder maps the observation to a latent state, and the model learns to
predict the next latent state, the reward and the value (an ensemble of
Q-functions) in that latent space. The model is trained to be consistent
over a short horizon of replayed steps (3 by default).

To act, it plans. At every step, a sampling planner (MPPI) tries hundreds of
short action sequences in the latent model, scores each by its predicted
rewards plus the value at the end, and refines the best ones over a few
iterations; a learned policy proposes some of the candidates. The first
action of the best plans is executed. Planning makes each step slower, but
the agent corrects its course at every step.

`TDMPC2MultiTask` trains one such model offline on the data of several
tasks, with a learned task embedding; it does not plan while training.

## Which one to use

| | DreamerV3 | TD-MPC2 |
| --- | --- | --- |
| Actions | discrete and continuous | continuous only |
| Episodes that end early (terminations) | yes, a model head predicts them | refused by default; `episodic=True` accepts them |
| Partial observability | built in: the model is recurrent | no memory |
| Who acts | the actor, learned in imagination: one network pass per step | the planner: 512 candidate sequences, 6 refinements per step by default, some proposed by a learned policy |
| Several tasks in one model | no | `TDMPC2MultiTask`, offline |
| Model sizes | `model_size="1m"` to `"400m"` (default `"12m"`) | `model_size=1, 5, 19, 48, 317` (millions of parameters; default 5) |
| `n_timesteps` counts | rows: one per environment per step, reset rows included | agent steps |
| Default `n_envs` | 16 | 1 |

In short:

- **Discrete actions or hidden state:** DreamerV3.
- **Continuous control with fixed-length episodes** (the DeepMind Control
  tasks, for example): TD-MPC2, whose paper-era code is written for exactly
  that. With terminations, it needs `episodic=True`, grafted from its later
  code.
- **One model for many tasks:** `TDMPC2MultiTask`.
- **A cheap baseline or a quick experiment:** neither. Both are slow on a CPU
  (see below); start with a model-free agent such as [SAC](../algorithms/sac.md)
  or [PPO](../algorithms/ppo.md).

Both take flat observation vectors only, and want the environment's raw
observations and rewards: do not normalise them, the agents do their own
scaling as their papers do. Both run gymnax, Brax and MuJoCo Playground
environments. Both take extensions in the phases every agent runs
(`pretrain`, `post_update`, `eval_metrics`) and no others.

## Examples

The smallest presets on Pendulum, as a first look. On a laptop CPU, the
DreamerV3 run below took about five minutes and the TD-MPC2 run about two
(compilation included):

```python
# doc: skip (about five minutes on a CPU)
from ajax import DreamerV3
from ajax.logging.wandb_logging import LoggingConfig

agent = DreamerV3("Pendulum-v1", model_size="1m", n_envs=4)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Train/episodic mean reward"])  # the score the paper reports
```

```python
# doc: skip (about two minutes on a CPU)
from ajax import TDMPC2
from ajax.logging.wandb_logging import LoggingConfig

agent = TDMPC2("Pendulum-v1", model_size=1)  # the first 1,000 steps are random (seed_steps)
logging = LoggingConfig(config={}, log_frequency=600, use_wandb=False)
state, evaluations = agent.train(seed=0, n_timesteps=1_200, num_episode_test=2,
                                 logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
```

The algorithm pages give faster configurations, the paper protocols, every
hyperparameter, and the multi-task pipeline (single-task runs, then
`export_episodes` and `pool_tasks`, then `TDMPC2MultiTask`).

## Cost

Both agents are much heavier than the model-free ones. In the CPU learning
checks recorded in [PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md), on a
busy shared machine, TD-MPC2 at `model_size=1` took 45 minutes for 6,000
Pendulum steps over two seeds, and DreamerV3 at `"1m"` 68 minutes for 20,000
Pendulum rows. Plan on a GPU for anything beyond a check.

Memory grows with the replay. DreamerV3 stores about 8 KB per row at `"12m"`
on a 24-dimensional observation: the paper's DeepMind Control protocol
(250,000 rows) needs about 2 GB of replay per seed, and about 3 GB with the
training step's working set. TD-MPC2's whole-episode replay is smaller: about
124 MB per seed for a million steps of DMC Walker. Seeds trained together
multiply these figures. Each agent reports its own figure for a configuration
as `replay_bytes_per_seed`.

## Fidelity

Each agent follows the code that produced its paper's numbers, not the
latest version of the official repository: DreamerV3 `danijar/dreamerv3`
at `2411f7d` plus one later bug fix (`29eb964`), TD-MPC2
`nicklashansen/tdmpc2` at `b67b21c`. Where the paper and that code disagree,
Ajax follows the code; paper typos are treated as typos. Asynchronous or
unseeded parts of the reference infrastructure are made synchronous and
seeded, so a run is reproducible from its seed. Every place where Ajax
departs from the reference is listed, with its impact, in the deviations
register.

How far this has been checked:

- **Component by component.** Tests compare Ajax's pieces with literal
  transcriptions of the reference functions, and with fixtures recorded by
  running the pinned reference code itself.
- **End to end, DreamerV3.** The reference code and Ajax were run on the same
  CartPole protocol in three pre-registered rounds; the training statistics
  of the two agree within their seed ranges in most measurement windows
  ([reference comparison](../world_models/reference_comparison/README.md)).
- **Against the published curves.** The scripts, reference data and
  acceptance criteria for the GPU validation are written
  ([Validation](../world_models/VALIDATION.md)); the GPU runs have not been
  made yet. Ajax runs the MuJoCo Playground ports of the DeepMind Control
  tasks, whose physics differ from the originals, so any comparison is
  approximate.

## Design notes

The design notes hold the full specifications, written from the papers and
the pinned official code and checked line by line by an independent reader.

| Note | What it holds |
| --- | --- |
| [Overview](../world_models/README.md) | the papers, the pinned commits, how the specifications were produced |
| [Design](../world_models/DESIGN.md) | the architecture of both agents in Ajax: units, shared blocks, collection, replay, schedules, testing |
| [Deviations](../world_models/deviations.md) | which code version Ajax follows where versions differ, and every departure from the reference |
| [DreamerV3 specification](../world_models/dreamerv3_spec.md) | every component, with exact values and paper and code citations |
| [TD-MPC2 specification](../world_models/tdmpc2_spec.md) | the same for TD-MPC2, single-task and multi-task |
| [Shared building blocks](../world_models/shared_blocks.md) | `symlog`, two-hot encodings, return normalisers, normalised MLPs, and their tests |
| [Validation](../world_models/VALIDATION.md) | the validation against the published curves: data, runner, criteria |
| [Reference comparison](../world_models/reference_comparison/README.md) | DreamerV3 against the reference code on CartPole |

## See also

- [DreamerV3](../algorithms/dreamerv3.md) and [TD-MPC2](../algorithms/tdmpc2.md):
  hyperparameters, examples, notes and results.
- [Memory](memory.md): memory blocks for the model-free agents.
- [Performance](performance.md): running seeds side by side, compilation
  cache.
