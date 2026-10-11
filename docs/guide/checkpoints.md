# Checkpoints and resuming

A run can stop and continue: in the same process by passing its state back
to `train`, or in another process through a checkpoint file. Either way the
continued run is the run that never stopped. The tests check it state by
state for most agents and by the counters for the rest
([below](#what-the-tests-guarantee)).

## In the same process

`train` returns the whole agent state. Passing it back as `initial_state`
continues the run from its timestep:

```python
from ajax import SAC

agent = SAC("Pendulum-v1", learning_starts=500)
state, _ = agent.train(seed=[0, 1], n_timesteps=1_000)
state, _ = agent.train(seed=[0, 1], n_timesteps=1_000, initial_state=state)
print(state.collector_state.timestep)  # [2000 2000]
```

Pass the same seeds in every call. The call may reuse the memory of the
state you pass in, so read or save what you need from it first.

## Across processes

`ajax.checkpoint` has three functions:

| Function | What it does |
| --- | --- |
| `save_checkpoint(state, path)` | writes the state's arrays and counters to `path`, atomically (a crash mid-write leaves the previous file intact) |
| `restore_into(skeleton, path)` | loads `path` into `skeleton`, a fresh state of the same agent |
| `checkpoint_exists(path)` | whether `path` holds a checkpoint |

A checkpoint stores data only. The optimisers' update rules and the
networks' functions are code, not data: they come from the skeleton, which
an identically configured agent builds with a 0-step `train` call. A run
that saves a checkpoint after each chunk of training, and picks up from the
last one when restarted:

```python
import os
import tempfile

from ajax import SAC
from ajax.checkpoint import checkpoint_exists, restore_into, save_checkpoint

path = os.path.join(tempfile.mkdtemp(), "sac_pendulum.pkl")
seeds, chunk, total = [0, 1], 1_000, 2_000

agent = SAC("Pendulum-v1", learning_starts=500)
state, _ = agent.train(seed=seeds, n_timesteps=0)  # the skeleton: init only
if checkpoint_exists(path):
    state = restore_into(state, path)

while int(state.collector_state.timestep[0]) < total:
    state, _ = agent.train(seed=seeds, n_timesteps=chunk, initial_state=state)
    save_checkpoint(state, path)
print(state.collector_state.timestep)  # [2000 2000]
```

The file is a pickle. Load only checkpoints you trust, since unpickling can
run code.

## What a resume keeps

Everything in the state: the network parameters, the optimisers' moments and
step counts, the replay buffer, each environment copy where it stood (in the
middle of an episode, if it was), the observation statistics, the memory
carries, the random keys, the extensions' states, and the counters. Because
the counters come back, the evaluations continue on their cadence: a run
split in two logs and evaluates at the same steps as the run left whole.

Schedules continue rather than restart. Each call trains from the timestep
it was given to that timestep plus `n_timesteps`, and measures the training
fraction, the extensions' schedules and the logging against that horizon.
A schedule set as a fraction of training is therefore measured against each
call's own horizon: plan a split run's schedules with that in mind.

## What the tests guarantee

The split-run tests save a run halfway, restore it into a new agent's
skeleton, finish it, and compare it with the uninterrupted run:

| Agents | Check |
| --- | --- |
| SAC, DQN, PQN, PPO, UDRL, APG | every array of the final state equal, to a relative 1e-5 |
| DreamerV3, TDMPC2, TDMPC2MultiTask | the same, in their own test files |
| TD3, REDQ, ASAC, APO, AVG | the resumed run ends at the same timestep as one call |

A test fails if an agent is added without being listed in one of these.
Logging is checked too: for SAC, DQN, PPO and PQN, the two halves log and
evaluate at exactly the steps the whole run does.

## Things to know

- **The seeds must stand at the same timestep.** A state whose seeds
  disagree on the timestep cannot be resumed and raises an error.
- **On-policy agents add a rollout per call.** PPO, APO and PQN train in
  whole rollouts plus one, so each call ends a rollout past its budget. A run
  split into calls ends later than one call with the same total.
- **APG restarts its optimiser on a resume** by default, which its staged
  curriculum relies on. Pass `reset_optimizer_on_resume=False` to `train` to
  keep it.
- **Checkpoints follow the code.** A checkpoint restores into the state
  layout of the Ajax version that loads it. When a layout changes, Ajax keeps
  a conversion for older checkpoints for a while (one exists today for
  replay buffers saved before they stored the next observation) and then
  removes it.
