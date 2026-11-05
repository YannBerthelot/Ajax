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
it would without it: the probe draws on no key of the agent's. It reads
networks that take the observation alone: agents with memory, the world models,
APG and UDRL are refused, and so is a network without an activation layer the
probe can read, such as a CNN encoder's.

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

`replay_states=256` also keeps that many observations of a training batch in
each probe after the first (`replay/observations`, on an agent with a replay),
to set what the agent trains on against the probe set later;
`run_data.load_replay_observations` reads them back.

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

## Embedding a run

The embeddings run offline, on a CPU, in minutes, and need the `manifolds`
extra (umap-learn, M-PHATE, matplotlib):

```bash
pip install "ajax-rl[manifolds]"
python -m ajax.representation runs/pendulum figures/pendulum --truth theta_dot
```

To set the metrics against the returns, keep the evaluations with the probes
first:

```python
from ajax.representation.run_data import save_evaluations

save_evaluations("runs/pendulum", evaluations)
```

The command writes, for the actor and the critic (the critic's first member):

| Figure | Shows |
| --- | --- |
| `manifold_<network>.png` | the last hidden layer's states on an Aligned UMAP: one embedding per probe, tied to the next by the states they share, coloured by env step, the critic's value and the observation columns, then the value at six probes |
| `units_<network>.png` | the network's hidden units on M-PHATE (Gigante et al., 2019): each point is one unit at one probe, placed by what it responds to, by env step and by layer |
| `weights_<network>.png` | the weights on their two principal components, beside a random walk of the same step lengths |
| `metrics.png` | every layer's metrics against env steps, each seed and their mean, under the return |

Read the weights figure with its random walk beside it: PCA draws any
high-dimensional random walk as a smooth curve explaining most of its variance
(Antognini and Sohl-Dickstein, 2018). `--max-probes` and `--max-states` bound
Aligned UMAP's work, and `--max-nodes` M-PHATE's, whose kernel holds a pair of
nodes: its default of 20,000 nodes takes some gigabytes. `ajax.representation`'s
`run_data` reads a run back as NumPy arrays, and `embeddings` holds the embeddings
without the figures.

## What it costs

A run that does not list the probe compiles what it always did. A run that lists
it pays for each probe and for nothing between probes, most of that in the SVDs.
So the overhead falls with the steps between probes: about 10% measured at a
probe every 300 to 500 steps of a small task on a CPU, so about 1% at one every
5,000 steps, and 0.5% for 100 probes over a million steps. `layers=` and a longer
`probe_interval` reduce it. A probe takes 13 MB per seed on SAC's defaults.
`PERFORMANCE_REPORT.md` has the measurements.

## Resuming

A resumed run goes on from its last probe. The state it resumes from has each
seed's own probe counters, which turns the compiled gate into a select: the probe
runs at every update, though it writes only when due. That run is correct and
much slower.
