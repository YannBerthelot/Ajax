# Memory

When the current observation is not enough to act well (a velocity the
sensors do not show, a cue seen earlier in the episode), the agent must
remember. The `memory` argument adds a memory block to the agent's networks,
between the hidden layers and the output heads. Ajax carries the memory from
step to step, resets it when an episode ends, and replays it correctly in
training.

```python
from ajax import PPO
from ajax.networks.memory import MemoryConfig

agent = PPO(
    "CartPole-v1",
    n_envs=8,
    n_steps=128,
    memory=MemoryConfig(kind="gru", hidden_size=32),
    bptt_length=32,  # learn on 32-step pieces of each copy's rollout
)
state, _ = agent.train(seed=0, n_timesteps=4_096)
```

## Which agents

| Agent | Memory |
| --- | --- |
| PPO | `memory=`, trained on rollouts |
| SAC, ASAC, REDQ, TD3, TQC | `memory=`, trained on replayed sequences |
| APG | `memory=`, trained through the simulator |
| DreamerV3 | built in: its world model is recurrent; `memory=` raises an error |
| AVG, CrossQ, DQN, PQN, APO, IQL, UDRL, DAgger, TDMPC2, TDMPC2MultiTask | none; `memory=` raises an error |

## The kinds

`MemoryConfig` (or the same settings as a dictionary) chooses the block:

| `kind` | What it is | Cost |
| --- | --- | --- |
| `"gru"` | a gated recurrent unit | steps through a sequence one at a time; keeps one vector per copy |
| `"lstm"` | a long short-term memory cell | as the GRU |
| `"transformer"` | causal self-attention over the episode's last `window` steps | trains on a whole sequence in parallel; keeps the last `window - 1` inputs per copy |
| `"mamba"` | a selective state-space block, as in Mamba | trains in parallel with an associative scan; keeps a small state per copy |

```python
from ajax.networks.memory import MemoryConfig

gru = MemoryConfig(kind="gru", hidden_size=64, num_layers=1)
attention = MemoryConfig(kind="transformer", hidden_size=64, window=32, num_heads=4)
ssm = MemoryConfig(kind="mamba", hidden_size=64, d_state=16, d_conv=4, expand=2)
lstm = {"kind": "lstm", "hidden_size": 64}  # a dictionary works too
```

| Setting | Default | Used by |
| --- | --- | --- |
| `hidden_size` | 64 | every kind; a multiple of `num_heads` for the transformer |
| `num_layers` | 1 | every kind |
| `gradient_checkpoint` | `False` | every kind: recompute activations in the backward pass, trading time for memory on long sequences |
| `window`, `num_heads` | 32, 4 | transformer: a step attends to itself and the `window - 1` steps before it in the same episode |
| `d_state`, `d_conv`, `expand` | 16, 4, 2 | mamba: state size, causal-convolution width, inner-width multiplier |
| `final_norm` | `True` | transformer, mamba: end with a LayerNorm, as GPT-2 and Mamba do; `False` only restores a checkpoint saved before it, and goes once none remains (an APG controller's parameters also map exactly onto the new network with `controller_params_with_final_norm`, which maps the parameters only, so a resumed training state must restart its optimizer) |
| `init` | `None` | every kind: the block's initialisation. `None` and `"flax"` are flax's defaults; `"stacked_orthogonal"` (GRU, LSTM) starts each gate as torch's orthogonal init of the stacked gates leaves it, as Ni et al. do; `"gpt2"` (transformer) draws every kernel from N(0, 0.02²), and `"nanogpt"` also scales the residual projections by 1/√(2 `num_layers`); `"mamba"` draws the step sizes log-uniformly in [0.001, 0.1], as Mamba does |

Acting and training run the same code, a sequence of one step being the
acting case, so what the agent learns is what it executes. Each kind is
tested for that equivalence.

Memory has a price, on the CPU especially. One measurement in
[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md) (recurrent SAC on
Pendulum, 4 copies, 2,048 steps, CPU) took about 16 s with a GRU or a
transformer and about 95 s with Mamba.

## On-policy: PPO

PPO collects a rollout of `n_steps` per copy (`n_envs` copies) with the
memory running, cuts each copy's rollout into pieces of `bptt_length` steps
and shuffles whole pieces, never single steps, into its minibatches. A piece
starts from the memory the policy had at its first step while collecting
(recomputed once before the update), never from zero in the middle of an
episode. Gradients flow through a piece but not back past its start:
`bptt_length` is how far back backpropagation through time reaches. Where a
time limit cut an episode, the value it bootstraps on is the critic's on the
episode's final observation, read with the memory that episode built, as
sb3-contrib's `RecurrentPPO` does.

### The libraries disagree on the length

No PPO paper gives one length for every task, and the reference codes fall
into two camps:

| Code or paper | Piece length | A minibatch holds |
| --- | --- | --- |
| OpenAI Baselines' ppo2 at `ea25b9e` (`baselines/ppo2/ppo2.py:167-180`) | each copy's whole rollout | the whole rollouts of a subset of the copies; it asserts `nenvs % nminibatches == 0` |
| CleanRL's `ppo_atari_lstm.py` at `fe8d8a0` (`:296-313`) | the whole rollout (128 steps by default) | as Baselines, with the same assertion |
| PureJaxRL's `ppo_rnn.py` at `31756b1` (`:258-291`) | the whole rollout (128 steps by default) | as Baselines; it reshapes the copies into minibatches, which fails when they do not divide |
| RLlib (ray at `00a3adb`, `rllib/core/rl_module/default_model_config.py:188`) | `max_seq_len=20`: episodes cut into 20-step chunks, zero-padded and masked | whole chunks |
| Sample Factory at `3d48f55` (`sample_factory/algo/learning/learner.py:498-516`) | `recurrence`, by default the rollout length (32 steps) | whole chunks, shuffled |
| Pleines et al., 2022 ([arXiv:2205.11104](https://arxiv.org/abs/2205.11104)) | tuned per task by grid search; their code (`MarcoMeter/recurrent-ppo-truncated-bptt` at `a1e87a8`) uses 8 steps or whole episodes | whole chunks, zero-padded and masked |

The first three share the copies between minibatches, as Huang et al.'s
*37 Implementation Details of PPO* (2022) describes; the last three cut
fixed-length chunks. sb3-contrib's RecurrentPPO (at `075bd5b`) has no length
setting: it cuts at episode starts, copy changes and minibatch edges. Morad
et al., 2024 ([arXiv:2402.09900](https://arxiv.org/abs/2402.09900)), argue
that fixed-length chunks estimate the full gradient poorly, so a dependency
longer than a chunk is unlikely to be learned, and found whole episodes more
sample-efficient (with Q-learning, not PPO). All of them, and Ajax, agree on
the rest: a piece stays in one minibatch, the memory resets when an episode
ends, and a piece starts from the memory the policy had while collecting.

### What Ajax does by default

With `bptt_length=None` (the default) the number of minibatches chooses the
length: the longest pieces that cut every rollout into equal parts and split
exactly into the minibatches.

- When `n_envs` is a multiple of the number of minibatches, a piece is a
  copy's whole rollout and a minibatch holds the rollouts of
  `n_envs // num_minibatches` copies: the scheme of Baselines, CleanRL and
  PureJaxRL.
- Otherwise each rollout is cut into `num_minibatches // gcd(n_envs,
  num_minibatches)` equal pieces. When that count does not divide `n_steps`
  no length splits, and PPO refuses the settings at construction.

No published length fits every task, so Ajax keeps the reference codes'
whole rollouts whenever the settings allow them, cuts the rollouts only as
much as the requested minibatches need, and always trains on the number of
minibatches it was asked for. That number is `num_minibatches` when
positive, otherwise `max(n_steps, batch_size) // min(n_steps, batch_size)`
(see [PPO](../algorithms/ppo.md)).

| Settings | Minibatches | Default `bptt_length` | A minibatch holds |
| --- | --- | --- | --- |
| PPO's defaults: `n_envs=4`, `n_steps=2048`, `batch_size=64` | 2048 // 64 = 32 | 256: 32 // gcd(4, 32) = 8 pieces per rollout | one 256-step piece |
| PureJaxRL's defaults: `n_envs=4`, `n_steps=128`, `num_minibatches=4` | 4 | 128, the whole rollout | one copy's rollout |
| CleanRL's defaults: `n_envs=8`, `n_steps=128`, `num_minibatches=4` | 4 | 128, the whole rollout | two copies' rollouts |
| `n_envs=1`, `n_steps=32`, `num_minibatches=2` | 2 | 16, half the rollout | one half |
| `n_envs=4`, `n_steps=100`, `num_minibatches=32` | 32 | refused: 8 pieces do not divide 100 | |

### Setting it

- **To reproduce Baselines, CleanRL or PureJaxRL**, leave `bptt_length=None`
  and set `num_minibatches` to a divisor of `n_envs`, as in the second and
  third rows. PPO's defaults do not (4 copies, 32 minibatches): they learn on
  256-step pieces, with 32 × 10 epochs = 320 gradient steps per update, as
  the feedforward PPO takes.
- **An explicit `bptt_length` can only shorten the pieces.** It must divide
  the default length (256 with PPO's defaults); PPO refuses any other value.
  For longer pieces, ask for fewer minibatches.
- **The pieces must cover the memory horizon the task needs.** No gradient
  crosses the start of a piece, so the memory learns to keep a cue only from
  pieces holding both the step that shows it and the step that needs it. For
  a cue needed k steps later, 2k-step pieces hold both for only about half
  the places the cue can fall: make them several times longer than k. The
  memory still crosses pieces: this limits what the policy learns to keep,
  not what it uses once learned.
- **Shorter pieces trade that reach for cost.** An update takes `n_epochs`
  × the number of minibatches gradient steps, whatever `bptt_length`.
  Shorter pieces give a GRU or an LSTM, which steps through a piece one step
  at a time, shorter loops, and with few copies they are what allows more
  minibatches (one copy needs k pieces for k minibatches). Pieces shorter
  than the rollout cost one more actor pass per update, for their start
  memories.

## Off-policy: sequence replay

The replay agents (SAC, ASAC, REDQ, TD3) store whole trajectories and learn
from contiguous sequences, as R2D2 does (Kapturowski et al., 2019):

```python
from ajax import SAC

agent = SAC(
    "Pendulum-v1",
    memory={"kind": "lstm", "hidden_size": 32},
    burn_in=4,  # steps that warm the memory up, without gradients
    sequence_length=8,  # steps learned from after the burn-in
    stored_state=True,  # start from the memory the actor had when it acted
    learning_starts=200,
    batch_size=32,
)
state, _ = agent.train(seed=0, n_timesteps=400)
```

Each sampled sequence has `burn_in + sequence_length + 1` steps. The memory
starts from zero, runs through the first `burn_in` steps with the current
networks and no gradient, and the losses then train on the next
`sequence_length` steps. The burn-in lets the memory reach a realistic state
before the steps that are learned from. With `stored_state=True` the buffer
also stores the actor's memory at every step, and the actor starts the
learned steps from the memory it had when it acted; the critics, which do
not run while acting, are still burned in from zero. Kapturowski et al.
found stored state better than a zero start with burn-in at keeping the
replayed memory close to the one the policy had. The defaults are
`burn_in=8`, `sequence_length=16`, `stored_state=False`. The agent refuses
at construction a `learning_starts` that would not leave each copy one
whole sequence before the first update (`learning_starts // n_envs` must
exceed `burn_in + sequence_length + 1`), and `stored_state=True` without
`memory=`.

A recurrent critic's memory reads each step's observation and the action
taken before it. The action the loss asks about joins after the memory, at
the critic's head, so asking about another action does not change what the
critic remembers (the recurrent off-policy critic of Ni et al., 2022).

With memory, SAC, TD3 and REDQ accept no expert policy and no extensions
(behaviour cloning included, which is the `CloningPretrain` extension), and
ASAC and PPO no extensions, yet: they raise an error when training starts
rather than run untested combinations. APG takes memory and extensions
together.

## Evaluation and deployment

Evaluation runs its own copies of the environment, so each evaluation
episode starts the memory from zero, as a deployed policy would. A PID
policy takes no memory: its integral and difference already carry the past
(see [Networks](networks.md)).
