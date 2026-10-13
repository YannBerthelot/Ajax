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
| `final_norm` | `True` | transformer, mamba: end with a LayerNorm, as GPT-2 and Mamba do; `False` only restores a checkpoint saved before it, and goes once none remains (an APG controller's parameters also map exactly onto the new network with `controller_params_with_final_norm`) |

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
