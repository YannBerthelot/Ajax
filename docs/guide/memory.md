# Memory

When the current observation is not enough to act well (a velocity the
sensors do not show, a cue seen earlier in the episode), the agent must
remember. The `memory` argument adds a memory block to the agent's networks:
between the hidden layers and the output heads for PPO and APG, before the
hidden layers for the replay agents (see
[Off-policy](#off-policy-sequence-replay)). Ajax carries the memory from step
to step, resets it when an episode ends, and replays it correctly in
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
| SAC, ASAC, REDQ, TD3 | `memory=`, trained on replayed sequences |
| APG | `memory=`, trained through the simulator |
| DreamerV3 | built in: its world model is recurrent |
| AVG, DQN, PQN, APO, UDRL, TDMPC2 | none; `memory=` raises an error |

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

gru = MemoryConfig(kind="gru", hidden_size=128, num_layers=1)
attention = MemoryConfig(kind="transformer", hidden_size=128, window=32, num_heads=4)
ssm = MemoryConfig(kind="mamba", hidden_size=128, d_state=16, d_conv=4, expand=2)
lstm = {"kind": "lstm", "hidden_size": 128}  # a dictionary works too
```

| Setting | Default | Used by |
| --- | --- | --- |
| `hidden_size` | 128 | every kind; a multiple of `num_heads` for the transformer |
| `num_layers` | 1 | every kind |
| `gradient_checkpoint` | `False` | every kind: recompute activations in the backward pass, trading time for memory on long sequences |
| `window`, `num_heads` | 32, 4 | transformer: a step attends to itself and the `window - 1` steps before it in the same episode |
| `d_state`, `d_conv`, `expand` | 16, 4, 2 | mamba: state size, causal-convolution width, inner-width multiplier |
| `final_norm` | `True` | transformer, mamba: end with a LayerNorm, as GPT-2 and Mamba do; `False` only restores a checkpoint saved before it, and goes once none remains (an APG controller's parameters also map exactly onto the new network with `controller_params_with_final_norm`, which maps the parameters only, so a resumed training state must restart its optimizer) |

The default width, 128, is the one the references of Ajax's memory agents
use, for every agent and every kind: the recurrent PPO of OpenAI Baselines
(`lstm(nlstm=128)`), CleanRL (`ppo_atari_lstm.py`) and PureJaxRL
(`ppo_rnn.py`); the recurrent SAC and TD3 of Ni et al. (2022) (128 in 26 of
their 28 configurations; 512 on Atari's pixels, 64 on one meta-learning
task), of their Memory-RL (2023) and of LSTM-TD3; for the transformer, Memory-RL's GPT and the Decision
Transformer; for mamba, Decision Mamba, whose state size, convolution width
and expansion are Ajax's defaults too. sb3-contrib's RecurrentPPO and the
POPGym benchmark use 256. APG's `contextual_controller` sets its own width
(see [APG](../algorithms/apg.md)). A checkpoint saved at another width is
refused; one saved with the default before it became 128 restores into
`hidden_size=64`.

Acting and training run the same code, a sequence of one step being the
acting case, so what the agent learns is what it executes. Each kind is
tested for that equivalence.

Memory has a price, on the CPU especially. One measurement in
[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md) (recurrent SAC on
Pendulum, 4 copies, 2,048 steps, CPU) took about 16 s with a GRU or a
transformer and about 95 s with Mamba.

## On-policy: PPO

PPO collects a rollout of `n_steps` per copy with the memory running, then
learns from it starting from the memory each copy had when the rollout
began. By default it backpropagates through each copy's whole rollout.
`bptt_length` cuts the rollouts into pieces of that many steps (it must
divide `n_steps`); each piece starts from the memory recomputed up to its
first step, never from zero in the middle of an episode. PPO shuffles whole
pieces, not single steps, into its minibatches (see [PPO](../algorithms/ppo.md)
for how many). When the pieces cannot be split into that many minibatches,
PPO warns and trains on one.

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

The replay agents' networks take the layout of Ni et al. (2022)'s recurrent
off-policy actor and critic. The memory reads each step's observation and
the action taken before it (zero at an episode's start), each through one
dense layer and a ReLU of fixed width, 32 for the observation and 16 for the
action. The hidden layers (`actor_architecture`, `critic_architecture`)
come after the memory and read its output beside a second embedding of the
current observation: 32 wide in the actor, and in the critic 48 wide with
the action the loss asks about, so asking about another action does not
change what the critic remembers. The critic's Q-heads (two in SAC, ASAC and
TD3, `num_critics` in REDQ) share its memory, the embeddings before it and
the second embedding, as Ni et al.'s two heads do: each head is only its own
hidden layers and output. Checkpoints saved before this layout cannot be
restored.

| Network | The memory reads | The hidden layers read |
| --- | --- | --- |
| Actor (TD3's deterministic one too) | the observation (32) and the previous action (16) | the memory's output and the observation (32) |
| Critic (one memory for all its Q-heads) | the observation (32) and the previous action (16) | the memory's output and the observation with the queried action (48) |

Each width in brackets is one dense layer and a ReLU. The reference is Ni et
al.'s code (twni2016/pomdp-baselines @e7c19c3: `recurrent_actor.py`,
`recurrent_critic.py` and `policy_rnn.py` in `policies/models`). Ajax
differs from it in these ways:

- **No reward input.** Their memory can also read the last reward, embedded
  8 or 16 wide in their POMDP configurations (`configs/pomdp`). Ajax's reads
  the observation and the previous action only, their best inputs on the
  standard POMDPs.
- **The memory.** Theirs is a GRU or an LSTM, of 128 units on their POMDP
  tasks. Ajax's is any `MemoryConfig` kind, the transformer and mamba
  included, 128 units wide by default.
- **Initialisation and heads.** Ajax initialises the embedders as it does
  the hidden layers (orthogonal, zero bias), not with torch's default, and a
  GRU's or an LSTM's input weights with flax's default (LeCun normal), where
  Ni et al. make every recurrent weight orthogonal. The layers after the
  memory are each agent's own, as without memory: its hidden layers, output
  heads and their initialisers, where Ni et al.'s are rlkit's (fan-in
  uniform with a bias of 0.1, the output layer uniform within 0.003 for a
  Q-head and 0.001 for the policy).
- **The critic's loss.** SAC, ASAC and TD3 average their two heads' squared
  errors where Ni et al. add them: half the gradient, which Adam's step
  divides out but for its epsilon. REDQ, which Ni et al. do not have, adds
  its heads' errors.
- **Replay.** Ni et al. replay sequences that start inside one episode
  (whole episodes on Pendulum, CartPole and LunarLander, 64 steps on their
  PyBullet tasks), each from a zero memory, with no burn-in and the steps
  after the episode's end masked out. Ajax's windows run across episode
  ends, restarting the memory at each, and burn in first (above).

PPO and APG keep the memory after their hidden layers: Ni et al.'s layout is
off-policy and needs a critic over actions.

With memory, SAC, TD3 and REDQ accept no expert policy and no extensions
(behaviour cloning included, which is the `CloningPretrain` extension), and
ASAC and PPO no extensions, yet: they raise an error when training starts
rather than run untested combinations. APG takes memory and extensions
together.

## Evaluation and deployment

Evaluation runs its own copies of the environment, so each evaluation
episode starts the memory from zero, as a deployed policy would. A replay
agent's actor also reads the action it executed at the step before (zero
after a reset), after the observation (`ajax.networks.networks.policy_input`):
a deployed SAC, ASAC, REDQ or TD3 actor with memory needs that action fed
back each step, as evaluation does. A PID policy takes no memory: its
integral and difference already carry the past (see
[Networks](networks.md)).
