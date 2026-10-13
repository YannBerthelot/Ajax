# TD-MPC2

TD-MPC2 learns a model of the environment in a latent space and plans with it
at every step. Its world model has no decoder: an encoder, latent dynamics, a
reward predictor and an ensemble of Q-functions, trained together on short
sub-trajectories from replay. To act, a sampling planner (MPPI) tries
action sequences in the latent model, guided by a learned policy prior, and
executes the first action of the best ones. `TDMPC2MultiTask` trains one such
model offline on the data of several tasks.

**Paper:** Hansen, Su and Wang, *TD-MPC2: Scalable, Robust World Models for
Continuous Control*, ICLR 2024
([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)). Official code:
[nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2).

Ajax reproduces the paper-era code (`b67b21c`, whose agent is identical to
`5f6fade`). The [world-model design notes](../world_models/README.md) give the
full specification, and [deviations.md](../world_models/deviations.md) lists
every version choice and every departure from the reference.

## Can I use?

| Feature | `TDMPC2` | `TDMPC2MultiTask` |
| --- | --- | --- |
| Discrete actions | no | no |
| Continuous actions | yes | yes |
| Memory (`memory=`) | no | no |
| Environments | gymnax, Brax, MuJoCo Playground | trains on a dataset; evaluates on any of them |
| Observations | flat vectors, raw (no normalisation) | flat vectors of different sizes per task |
| Terminating episodes | refused, unless `episodic=True` | no |
| Several seeds in one call | yes | yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics` | the same |
| Resume and checkpoints | yes; a checkpoint restored by a new agent equals the uninterrupted run (test) | yes, the same |

## Example

```python
# doc: skip (about two minutes on a CPU; tests/agents/TDMPC2/test_TDMPC2.py covers this path)
from ajax import TDMPC2
from ajax.logging.wandb_logging import LoggingConfig

# model_size=1 and a smaller planner than the paper's, for a quick CPU run.
agent = TDMPC2("Pendulum-v1", model_size=1, num_samples=64, num_elites=8,
               num_pi_trajs=4, iterations=2, seed_steps=400)
logging = LoggingConfig(config={}, log_frequency=600, use_wandb=False)
state, evaluations = agent.train(seed=0, n_timesteps=1_200, num_episode_test=2,
                                 logging_config=logging)
print(agent.gamma)  # 0.975, derived from Pendulum's 200-step episodes
print(evaluations["Eval/episodic mean reward"])
```

The paper's DMC protocol: action repeat 2 over 1,000 simulator steps (500
agent steps per episode), the 5M model, one environment. The paper's "1M
environment steps" are 500,000 agent steps:

```python
# doc: skip (the paper's full budget; benchmarks/world_models/paper_protocol.py runs it)
from ajax import TDMPC2

agent = TDMPC2("CheetahRun", model_size=5, action_repeat=2, episode_length=1000)
state, _ = agent.train(seed=0, n_timesteps=500_000)
```

## Key hyperparameters

Every default is the paper's (`config.yaml` at `5f6fade`).

| Argument | Default | What it does |
| --- | --- | --- |
| `model_size` | 5 | preset of 1, 5, 19, 48 or 317 M parameters; fills `enc_dim`, `mlp_dim`, `latent_dim`, `num_enc_layers`, `num_q` |
| `n_envs` | 1 | environments stepped in lockstep, one update per environment step |
| `horizon` | 3 | length of the training sub-trajectories and of the plans |
| `batch_size` | 256 | sub-trajectories per update |
| `buffer_size` | 1,000,000 | environment steps kept in replay |
| `seed_steps` | None | random steps before training; `None` is `max(1000, 5 T)` |
| `updates_per_step` | 1.0 | updates per environment step after the seed phase |
| `learning_rate` | 3e-4 | a float or a schedule; the encoder trains at `learning_rate * enc_lr_scale` (0.3) |
| `gamma` | None | `None` derives it from the episode length `T`: `(T/5 - 1) / (T/5)`, clipped to [0.95, 0.995] |
| `iterations`, `num_samples`, `num_elites` | 6, 512, 64 | MPPI iterations, sampled plans and kept plans |
| `num_pi_trajs` | 24 | plans proposed by the policy prior |
| `consistency_coef`, `reward_coef`, `value_coef` | 20, 0.1, 0.1 | world-model loss weights |
| `rho` | 0.5 | per-step decay of the loss weights along the horizon |
| `entropy_coef` | 1e-4 | policy prior's entropy weight |
| `tau` | 0.01 | target Q update rate |
| `episodic` | False | accept terminating tasks (the latest code's episodic mode) |
| `action_repeat`, `episode_length` | 1, 1000 | simulator steps per agent step, and per episode (Brax, Playground) |

## Notes

- **What `n_timesteps` counts.** Agent steps (environment steps after action
  repeat); logs also carry `env_frames`. With `episodic=True` it counts rows:
  environment steps plus one per finished episode.
- **Terminations.** The paper-era agent bootstraps through every episode end,
  so a terminating task would train on wrong targets. Ajax counts terminations
  and raises a `ValueError` after `train` if there were any (deviation T10).
  `episodic=True` grafts the latest official code's episodic mode (commit
  `eece801`) onto the paper-era agent instead: a termination head and
  variable-length episodes in replay (T29).
- **Evaluation** (with a `logging_config`): every `log_frequency` environment
  steps, `num_episode_test` episodes of the planner in evaluation mode, on a
  rebuilt environment, each evaluation from fresh initial states (the same
  ones when a seed is rerun; T9). The paper uses
  `log_frequency=50_000` with 10 episodes. `LoggingConfig`'s default of 1,000
  makes a 500-step-episode run spend about twice as long evaluating as
  training.
- **Replay memory** is sized at each `train` call for the steps the run will
  have taken: `min(buffer_size, total)` (T26), in the episodic mode as well
  (T29). At full capacity it takes 124 MB per seed on DMC Walker. A run
  trained in chunks, or restored from a checkpoint by a new agent, replays
  exactly what the uninterrupted run does.
- **Actions.** The planner's actions lie in `[-1, 1]`; Ajax maps them onto
  the environment's bounds, as the reference scales its actions.
- **Several environments.** `n_envs > 1` plans for each environment separately
  and takes `n_envs` updates per step (T7); the reference uses one.
- The DMC tasks run on MuJoCo Playground's MJX ports, whose physics differ
  from dm_control's (E18), so comparisons with the published curves are
  approximate.

## Multi-task: `TDMPC2MultiTask`

The paper's multi-task models are trained offline, on the pooled replay of
single-task agents. `TDMPC2MultiTask` does the same: a learnable task
embedding conditions the model, observations are zero-padded to the largest
task's size, the unused action dimensions are masked, and each task keeps the
discount of its own episode length. Training does no planning, and
`n_timesteps` counts updates. Build the dataset from single-task runs with
`export_episodes` (one run's replay), `concatenate_episodes` (several seeds of
a task) and `pool_tasks`; `save_dataset` and `load_dataset` store it as an
`.npz` file.

```python
# doc: skip (about five minutes on a CPU; tests/agents/TDMPC2/test_TDMPC2MultiTask.py covers this path)
from ajax import TDMPC2, TDMPC2MultiTask
from ajax.agents.TDMPC2.dataset import export_episodes, pool_tasks
from ajax.logging.wandb_logging import LoggingConfig

small = dict(model_size=1, num_samples=64, num_elites=8, num_pi_trajs=4, iterations=2)
tasks = ["Reacher-misc", "PointRobot-misc"]  # 8 and 6 observations, 2 actions, T = 100

episodes = []
for task in tasks:  # one single-task run per task; its replay is the task's data
    agent = TDMPC2(task, seed_steps=500, **small)
    state, _ = agent.train(seed=0, n_timesteps=1_000)
    episodes.append(export_episodes(agent, state, seed_index=0, name=task))
dataset = pool_tasks(episodes)  # every task's episodes must have the same length

multi = TDMPC2MultiTask(dataset, eval_envs=tasks, batch_size=256, **small)
logging = LoggingConfig(config={}, log_frequency=500, use_wandb=False)
state, history = multi.train(seed=0, n_timesteps=1_000, num_episode_test=2,
                             logging_config=logging)
print(history["Eval/Reacher-misc/episodic mean reward"])
print(history["Eval/episodic mean reward"])  # the mean over tasks
```

It also logs `Eval/normalized score`, the mean over tasks of `return / 10`
(the paper's normalisation of DMC returns). `multi.evaluate(state,
num_episodes=10)` runs the same evaluation outside training, on a state
`train` returned, and gives the same keys, one value per seed; without
`eval_envs` there is nothing to evaluate on, and it raises. Defaults are the single-task
agent's, except `batch_size=1024` and `task_dim=96`, the paper's (T14). The
evaluation follows the offline trainer's protocol (T16). Ajax's datasets are
generated by its own single-task runs, not the paper's mt30 and mt80 data, so
they validate the mechanisms rather than the paper's numbers (T17).

## Results

Recorded on CPU in [PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md),
with `model_size=1` and every other setting at the paper's default, two
seeds: Pendulum reaches -143 and -124 after 6,000 steps (a random policy
scores about -1,200), and Playground's CartpoleBalance (action repeat 2)
reaches 810 and 854 after 10,000 steps, still improving. The GPU validation
against the published curves is specified in
[VALIDATION.md](../world_models/VALIDATION.md) and has not been run yet. The
probes (`tests/probing`) train TD-MPC2 briefly on tiny environments whose
answers need its model (a payoff past its planning horizon; twin tasks for
the multi-task agent), and its own tests (`tests/agents/TDMPC2`) check that
a run split by a checkpoint equals the run left whole, for both agents. On
the CPU speed benchmark (`benchmarks/agent_bench.py`:
Pendulum, 4 environments, 8,000 steps, a smaller model than the paper's)
TD-MPC2 ran at 292 steps per second of execution (245 with compilation) at
main `a8a13c2`.

## API

::: ajax.agents.TDMPC2.TDMPC2.TDMPC2

::: ajax.agents.TDMPC2.TDMPC2MultiTask.TDMPC2MultiTask

::: ajax.agents.TDMPC2.dataset.export_episodes

::: ajax.agents.TDMPC2.dataset.concatenate_episodes

::: ajax.agents.TDMPC2.dataset.pool_tasks

::: ajax.agents.TDMPC2.dataset.save_dataset

::: ajax.agents.TDMPC2.dataset.load_dataset
