# Representations during training

Neuroscience studies a population of neurons by embedding its activity into a
low-dimensional manifold. The same question applies to an agent: how do the
activations of its networks change while it learns? `RepresentationProbe`
reads the networks on a fixed set of states during training and writes what it
reads to a directory. It runs inside the compiled training loop; embedding the
result happens offline, and is not part of training.

```python
from ajax import SAC
from ajax.extensions.representation import RepresentationProbe
from ajax.logging import LoggingConfig

probe = RepresentationProbe("runs/pendulum", probe_interval=1_000)
agent = SAC("Pendulum-v1", extensions=(probe,))
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=10_000, logging_config=logging)
```

It adds nothing to a run that does not list it. A run that lists it trains as
it would without it: the probe draws on no key of the agent's. Recurrent agents
and the world models are refused, and so is a network without an activation
layer the probe can read, such as a CNN encoder's.

## The probe set

The first run in a directory makes `probe_set.npz`: `n_states` states (2,048 by
default) drawn from rollouts of a random policy, on `probe_seed`. Every
checkpoint of every seed reads these states, so what changes between two probes
is the network and not its input. A later run pointed at the directory reuses
the file, and a run with another `n_states` or `probe_seed` is refused. To
compare runs of two directories, pass the same `probe_set=` to both.

The file holds `observations`, as the env emits them, and `actions`, those of
the random policy, in the agent's units: an action-value critic reads them with
the observations, so that its input stays fixed too. `truth/<field>` holds
what the env knows about each state: the reward and, on a gymnax env, the
fields of its state (`truth/theta` on Pendulum).

## What is written

The probes come at step 0 and then at the first update on or past each multiple
of `probe_interval` env steps. Each writes one file per seed,
`seed000_probe00012.npz`:

| Key | Holds |
| --- | --- |
| `step` | the env step of the probe |
| `layers/<layer>` | activations after the nonlinearity, `(n_states, n_units)`, with a leading member axis for an ensemble critic |
| `values` | the critic's mean value at each state, when the agent has an ensemble critic |
| `parameters/<network>/...` | the network's parameters |
| `metrics/Representation/<layer>/<metric>` | the metrics below |

`layers=("actor/encoder/network/layers_3",)` keeps the layers whose name holds
one of the strings. `float16=True` halves the activations' size, and turns
values above 65504 infinite; parameters stay float32, since rounding them would
swamp the small steps between two probes. A probe of SAC's defaults, 2,048 states
and six layers of 256 units, takes 13 MB per seed.

## Metrics

Computed in the compiled program at each probe, per layer, averaged over an
ensemble's members:

| Metric | Meaning |
| --- | --- |
| `effective_rank` | `exp(H(p))`, with `p` the singular values normalised to sum to 1: how many directions the layer spans (Roy and Vetterli, 2007) |
| `cka_previous` | linear centred kernel alignment with the previous probe's activations, 1 when they differ by a rotation and scale alone (Kornblith et al., 2019); not logged for the first probe |
| `activation_norm` | the mean L2 norm of a state's activations |
| `dead_fraction` | the fraction of units no state changes: a ReLU stuck at zero, a tanh pinned at ±1 |

Each evaluation logs the latest probe's, as
`Representation/<layer>/<metric>`, to the run's backends and to the returned
evaluations. Every probe's are in its file.

## Resuming

A resumed run goes on from its last probe. The state it resumes from has each
seed's own probe counters, which turns the compiled gate into a select: the probe
runs at every update, though it writes only when due. That run is correct and
much slower.
