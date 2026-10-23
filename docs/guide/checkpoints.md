# Checkpoints and resuming

A run can stop and continue: in the same process by passing its state back
to `train`, or in another process through a checkpoint file. Either way the
continued run is the run that never stopped. The tests check it state by
state for every agent ([below](#what-the-tests-guarantee)).

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
an identically configured agent builds with a 0-step `train` call given the
run's `train` options (an update rule may depend on them, as APG's
learning-rate schedule does, [below](#things-to-know)). The agent may be a
new instance in a new process. A run that saves a checkpoint after each
chunk of training, and picks up from the last one when restarted:

```python
import os
import tempfile

from ajax import SAC
from ajax.checkpoint import checkpoint_exists, restore_into, save_checkpoint

path = os.path.join(tempfile.mkdtemp(), "sac_pendulum.pkl")
seeds, chunk, total = [0, 1], 1_000, 2_000

agent = SAC("Pendulum-v1", learning_starts=500)
# The skeleton: init only, with the run's options.
state, _ = agent.train(seed=seeds, n_timesteps=0, total_timesteps=total)
if checkpoint_exists(path):
    state = restore_into(state, path)

while (done := int(state.collector_state.timestep[0])) < total:
    state, _ = agent.train(
        seed=seeds,
        n_timesteps=min(chunk, total - done),
        initial_state=state,
        total_timesteps=total,  # the schedules run over the whole run
    )
    save_checkpoint(state, path)
print(state.collector_state.timestep)  # [2000 2000]
```

The file is a pickle. Load only checkpoints you trust, since unpickling can
run code.

## What a resume keeps

Everything in the state: the network parameters, the optimisers' moments and
step counts, the replay buffer, each environment copy where it stood (in the
middle of an episode, if it was), the observation and reward statistics,
the memory carries, a stateful expert's state, the random keys, the
extensions' states, and the counters. Because the counters come back, the
evaluations continue on their cadence, each from the key of its own count:
a run split in two logs and evaluates at the same steps, on the same
episodes, as the run left whole.

Schedules continue rather than restart. Each call trains from the timestep
it was given to that timestep plus `n_timesteps`, and measures the training
fraction, the exploration, the extensions' schedules and the logging
against the horizon `total_timesteps`. Pass the whole run's horizon to every
call, as the loop above does, and a split run follows the schedules of one
call; without it, each call's horizon is its own end.

## What the tests guarantee

The resume tests train a run in one call and again in two. Between the
two calls they save the state, check that it restores leaf for leaf into a
new agent's skeleton, and continue from the restored state on that new
agent; then they compare the two end states:

| Agents | Check |
| --- | --- |
| SAC, ASAC, REDQ, TD3, AVG, APO, PPO, DQN, PQN, UDRL | every array of the final state equal, to a relative 1e-5, with every normalisation the agent takes switched on and every call told the run's horizon (DQN's and PQN's epsilon decays across the split); once more with a GRU memory for SAC, ASAC, REDQ, TD3 and PPO |
| APG | the same, with and without memory; with memory, the optimiser kept and a warm-up cosine learning rate |
| DreamerV3, TDMPC2, TDMPC2MultiTask | the same, in their own test files |

The tolerance is there because two programs compiled apart may round
differently. A test fails if an agent is added without a row, or one that
takes normalisation or memory without the rows for them. Logging is checked
too: for SAC, DQN, PPO and PQN, the two halves log and evaluate at exactly
the steps the whole run does.

## Things to know

- **The seeds must stand at the same timestep.** A state whose seeds
  disagree on the timestep cannot be resumed and raises an error.
- **On-policy agents add a rollout per call.** PPO, APO and PQN train in
  whole rollouts plus one, so each call ends up to a rollout past its budget.
  A run split into calls ends where one call does when its last call trains
  the rest, `total_timesteps` minus the timestep reached
  ([Continuing a run](training.md#continuing-a-run)); otherwise it may end
  elsewhere.
- **APG restarts its optimiser on a resume** by default, which its staged
  curriculum relies on: its learning-rate schedule then spans each call.
  Pass `reset_optimizer_on_resume=False` to `train` to keep it. The kept
  optimiser runs the schedule it was built with, by the run's first call or
  the skeleton: give that call `reset_optimizer_on_resume=False` and the
  run's `total_timesteps` too, and the schedule spans the run.
- **Checkpoints follow the code.** A checkpoint restores into the state
  layout of the Ajax version that loads it. When a layout changes, Ajax keeps
  a conversion for older checkpoints for a while, then removes it. Today
  these restore:
    - a replay buffer saved before it stored each row's next observation
      (`next_obs`, or the environment's own `raw_next_obs`): the field is
      rebuilt from the following row, which is what that replay read then;
    - a replay buffer saved with the raw observations and expert flags in a
      run without an expert, from before the buffer kept them only for an
      expert: it keeps them, and the resumed run keeps writing them;
    - a state saved before a field that the new layout leaves empty existed:
      the field is restored empty. A missing field that holds data is a
      mismatch and still raises;
    - an on-policy agent's state saved while its observation statistics
      lived in the environment's normaliser (before 2026-10-23): they move
      to the collector, a gymnax run's statistics, one set per copy, merged
      into one, and a Brax state gets its raw observation back;
    - UDRL's segment buffer saved with `cum_rewards`, a running sum of its
      rewards that nothing read: the field is dropped. A saved field that
      the new layout lacks and does not list as removed is a mismatch and
      still raises;
    - a UDRL segment buffer saved with a second copy of the environment's
      own observations (`raw_obs`, kept for a run with an expert on
      2026-10-17): the copy is dropped, since `obs` holds them. A UDRL buffer
      of observations the environment normalised is refused: UDRL now keeps
      them raw. Until 2026-11-17;
    - a PPO's or an APO's optimiser state saved when that optimiser
      clipped its network's gradient (`max_grad_norm`), as PPO's two did
      before PPO took ppo2's one clip over both networks, and APO's two
      before APO took its official code's, into an optimiser without the
      clip: the clip holds no data, so the moments and step counts are
      kept. Another agent's is refused.
