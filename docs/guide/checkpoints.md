# Checkpoints and resuming

A run can stop and continue: in the same process by passing its state back
to `train`, or in another process through a checkpoint file. Either way the
continued run is the run that never stopped, up to rounding; the tests
check it for every agent ([below](#what-the-tests-guarantee)).

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

Pass the same seeds in every call. The state carries each run's random
streams, so a resumed run continues the same whatever seeds the call names:
they only name the runs in the logs and in `agent.members`, and `train`
cannot tell that they changed. A different number of seeds is refused. The
call may reuse the memory of the state you pass in, so read or save what
you need from it first.

## Save and load an agent

`agent.save` writes the state together with the agent's class and its
arguments, and `load` gives both back, so the loading script does not build
the agent again:

```python
import os
import tempfile

import ajax
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128, gamma=0.98)
state, _ = agent.train(seed=[0, 1], n_timesteps=2_048)
path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
agent.save(state, path)

# Later, in another process.
agent, state = ajax.load(path)  # or PPO.load(path), which refuses another class's file
print(agent.gamma)  # 0.98
state, _ = agent.train(seed=[0, 1], n_timesteps=2_048, initial_state=state)
```

- **What the file stores.** The state, which `load_checkpoint` and
  `restore_into` read too, and the arguments that are plain values: ids,
  numbers, strings, architectures, a memory configuration.
- **What you pass again.** An argument that is an object (a prebuilt
  environment, extensions, an expert policy) is not stored: `load` refuses,
  naming it, until you pass it again, as in
  `ajax.load(path, extensions=(...,))`. Any other argument given to `load`
  replaces the stored one.
- **Options of the original `train` call.** `train_options` goes to the
  0-step `train` call that rebuilds the state
  ([below](#across-processes)).

!!! warning "Load only files you trust"
    A checkpoint is a pickle, and unpickling can run code.

## Across processes

`ajax.checkpoint` has four functions:

| Function | What it does |
| --- | --- |
| `save_checkpoint(state, path)` | writes the state's arrays and counters to `path`, atomically: a crash mid-write leaves the previous file intact |
| `load_checkpoint(agent, path, **train_options)` | loads `path` into `agent`, an agent built with the same arguments: `restore_into` on a 0-step `train` call of as many seeds as the file holds, given `train_options` |
| `restore_into(skeleton, path)` | loads `path` into `skeleton`, a fresh state of the same agent |
| `checkpoint_exists(path)` | whether `path` holds a checkpoint |

A checkpoint stores data only. The optimisers' update rules and the
networks' functions are code, not data: they come from the skeleton, the
state an identically configured agent builds with a 0-step `train` call.
That call takes the run's `train` options, since an update rule may depend
on them, as APG's learning-rate schedule does ([below](#things-to-know)).
`load_checkpoint` passes on the options you give it; without any, it serves
a run trained with `train`'s defaults, and a policy to act with
([Train, evaluate, deploy](deploy.md)). The agent may be a new instance in a
new process.

A run that saves a checkpoint after each chunk of training, and picks up
from the last one when restarted:

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

done = int(state.collector_state.timestep[0])
while done < total:
    state, _ = agent.train(
        seed=seeds,
        n_timesteps=min(chunk, total - done),
        initial_state=state,
        total_timesteps=total,  # the schedules run over the whole run
    )
    save_checkpoint(state, path)
    done = int(state.collector_state.timestep[0])
print(state.collector_state.timestep)  # [2000 2000]
```

Arrays of 16 MiB or more, such as a large replay, are stored after the
pickle stream, and a restore copies each one once, from the file to where
the state keeps it: a resume holds no second copy of the replay.

## What a resume keeps

Everything in the state: the network parameters, the optimisers' moments
and step counts, the replay buffer, each environment copy where it stood
(in the middle of an episode, if it was), the observation and reward
statistics, the memory carries, a stateful expert's state, the random keys,
the extensions' states and the counters. Because the counters come back,
the evaluations keep their cadence, each from the key of its own count: a
run split in two logs and evaluates at the same steps, on the same
episodes, as the run left whole.

Schedules continue rather than restart, and run to the horizon
`total_timesteps`. Pass the whole run's horizon to every call, as the loop
above does, and a split run follows the schedules of one call; without it,
each call's horizon is its own end
([Continuing a run](training.md#continuing-a-run)).

## What the tests guarantee

The resume tests train a run in one call and again in two. Between the two
calls they save the state, restore it into a new agent's skeleton, and
continue from it on that new agent; then they compare the two end states:

| Agents | Check |
| --- | --- |
| DAgger, SAC, ASAC, RVI-SAC, REDQ, TD3, TQC, CrossQ, IQL, AVG, APO, PPO, REINFORCE, DQN, PQN, UDRL | every array of the final state equal, to a relative 1e-5 (CrossQ to an absolute 1e-4), with every normalisation the agent takes switched on and every call told the run's horizon; once more with a GRU memory for the agents that take one |
| APG | the same, with and without memory; with memory, the optimiser kept and a warm-up cosine learning rate |
| DreamerV3, TDMPC2, TDMPC2MultiTask | the same, in their own test files |

For SAC, DQN, PPO and PQN they also check that the two halves log and
evaluate at exactly the steps the whole run does.

The tolerance is there because two programs compiled apart may round
differently, and on physics tasks that rounding can grow. PPO on
HumanoidWalk, paused once on a CPU, ended with different weights from the
unbroken run, and so did PPO on CheetahRun on a GPU: each a valid run, but
not the same run. On a GPU, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
removed the gap on both tasks.

## Things to know

- **The seeds must stand at the same timestep.** A state whose seeds
  disagree on the timestep cannot be resumed and raises an error.
- **On-policy agents train whole rollouts.** PPO, APO, REINFORCE and PQN
  train the whole rollouts (`n_envs x n_steps` steps) a call's
  `n_timesteps` holds, at least one. A run split into calls of whole
  rollouts ends where one call does; otherwise it may end elsewhere
  ([The budget](training.md#the-budget)).
- **A resumed call logs as new runs.** Each `train` call opens a run per
  seed with a new run id, so a run made of two calls leaves two TensorBoard
  folders (or W&B runs) per seed; their steps continue, and each run's
  `config.json` records its seed.
- **APG restarts its optimiser on a resume** by default, which its staged
  curriculum relies on: its learning-rate schedule then spans each call.
  Pass `reset_optimizer_on_resume=False` to `train` to keep it. The kept
  optimiser runs the schedule it was built with, by the run's first call or
  the skeleton: give that call `reset_optimizer_on_resume=False` and the
  run's `total_timesteps` too, and the schedule spans the run.
- **Checkpoints follow the code.** A checkpoint restores into the state
  layout of the Ajax version that loads it. When a layout changes, Ajax
  converts older checkpoints for a while, then drops the conversion; a file
  it cannot convert raises, naming what differs.

??? note "The older checkpoints that restore today"
    - A replay buffer saved before it stored each row's next observation
      (`next_obs`, or the environment's own `raw_next_obs`): the field is
      rebuilt from the following row, which is what that replay read then.
    - A replay buffer saved with the raw observations and expert flags in a
      run without an expert, from before the buffer kept them only for an
      expert: it keeps them, and the resumed run keeps writing them.
    - A state saved before a field that the new layout leaves empty existed:
      the field is restored empty. A missing field that holds data is a
      mismatch and still raises.
    - An on-policy agent's state saved while its observation statistics
      lived in the environment's normaliser: they move to the collector (a
      gymnax run's statistics, one set per copy, merged into one), and a
      Brax state gets its raw observation back.
    - UDRL's segment buffer saved with `cum_rewards`, a running sum of its
      rewards that nothing read: the field is dropped. A saved field that
      the new layout lacks and does not list as removed is a mismatch and
      still raises.
    - A PPO's or an APO's optimiser state saved when each network's
      optimiser clipped its own gradient (`max_grad_norm`), before they took
      their reference codes' one clip over both networks: the clip holds no
      data, so the moments and step counts are kept. Another agent's is
      refused.
    - An optimiser state saved with one Adam moment array per parameter
      array, from before the optimisers held each moment as one vector over
      every parameter: the arrays are joined into that vector, every moment
      and step count kept. A state missing a parameter's moment is refused.
    - A state saved before collection drew on its own stream of keys
      (checkpoint format 2): its collector's key restarts from one derived
      from the updates' key, since the one it held was an ancestor of the
      updates' key and would hand out keys they had already drawn.
