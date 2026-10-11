# SAC

Soft Actor-Critic learns a stochastic policy that maximises the reward plus
the policy's entropy, so it keeps exploring while it improves. Two critics
estimate the soft action value, and the target uses the smaller of the two
target critics to curb over-estimation. The entropy weight `alpha` tunes
itself towards a target entropy. SAC learns off-policy from a replay buffer,
so it reuses each transition many times.

SAC is also the agent with the most expert-guidance support in Ajax: it is
the only one that runs `action` extensions (an expert acting during
collection) and keeps a few expert options of its own (see
[Expert options](#expert-options)).

**Papers.** Haarnoja et al., *Soft Actor-Critic*, 2018
([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)). The learned
temperature with a target entropy follows Haarnoja et al., *Soft Actor-Critic
Algorithms and Applications*, 2018.

## Can I use?

| | SAC |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions or expert options |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss`, `action` |

The `action` phase runs only when the agent has an `expert_policy` and
`use_pid_policy` is off; SAC raises at construction otherwise. See
[Extensions](../guide/extensions.md) for what each phase does.

## Example

```python
from ajax import SAC
from ajax.logging.wandb_logging import LoggingConfig

agent = SAC("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)

print(evaluations["timestep"][0])                # when each evaluation ran
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.
`train` runs both seeds side by side in one compiled program. See
[Training](../guide/training.md) for resuming, and [Logging](../guide/logging.md)
for Weights & Biases or TensorBoard.

## Key hyperparameters

Defaults are read from `SAC.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `fixed_alpha` | `False` | keep `alpha` at `alpha_init` |
| `num_critics` | `2` | critics in the ensemble; the target takes their minimum |
| `num_critic_updates` | `1` | critic steps per update |
| `policy_update_start`, `alpha_update_start` | `2_000` | timestep from which the actor and `alpha` update |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping of the actor and critics |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation (see [Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

### Expert options

These options remain on SAC because they reach the network's input size, the
collection or the action pipeline, where extensions do not reach yet. Each
mirrors the matching extension's own setting. The expert always receives the
raw (unnormalised) observations. See [Expert guidance](../guide/expert-guidance.md).

| Argument | Default | What it does |
| --- | --- | --- |
| `expert_policy` | `None` | the expert used in training; `None` is plain SAC |
| `eval_expert_policy` | `None` | an expert used only for the evaluation's comparison metrics |
| `expert_buffer_n_steps` | `20_000` | expert steps per env written to the buffer before training |
| `expert_fraction` | `0.7` | during warm-up, the chance a step takes the expert's action instead of a uniform one |
| `pre_train_n_steps` | `0` | expert steps to behaviour-clone the actor on before training |
| `actor_cloning_epochs`, `actor_cloning_lr`, `actor_cloning_batch_size` | `10`, `1e-3`, `64` | that cloning's settings |
| `skip_actor_pretrain` | `False` | collect the cloning data but do not train the actor on it |
| `reset_log_std_after_bc`, `reset_actor_head_after_bc` | `False` | after cloning, reset the actor's log-std head (restoring exploration), or its whole output head |
| `use_pid_policy` | `False` | the actor outputs the gains of the expert's PID (needs `learnable_fields` on the expert) |
| `augment_obs_with_expert_action`, `augment_obs_with_expert_state` | `False` | append the expert's action or internal state to the observation |
| `jsrl_curriculum` | `False` | keep the per-episode step counter `JSRLCurriculum` needs |
| `use_train_frac` | `False` | append the training fraction to the observation |
| `target_entropy_initial_per_dim`, `target_entropy_ramp_frac` | `None`, `0.5` | ramp the target entropy up from a low start |
| `normalize_obs_running` | `False` | agent-side observation normalisation, which also covers the appended expert columns (see [Environments](../guide/environments.md#normalisation)) |
| `store_policy_action` | `False` | store the policy's sampled action in the replay instead of the executed one (an off-policy-correctness ablation) |

`action_scale` is accepted and ignored. `skip_critic_pretrain=True` is a no-op
kept for a downstream project; `False` raises, use the `MCPretrain`
extension instead.

## Notes

- **Actions.** The actor is a `tanh`-squashed Gaussian, so actions lie in
  `[-1, 1]`. Ajax clips them to the environment's bounds without rescaling
  them, so on wider bounds (gymnax Pendulum's torque is `[-2, 2]`) SAC
  reaches only `[-1, 1]`; see [Actions](../guide/environments.md#actions).
  Evaluation uses the mean action.
- **Updates per step.** Each iteration steps every env once, then makes one
  update: `num_critic_updates` critic steps, one actor step, one temperature
  step and one target update. With `n_envs=4` that is one update per four
  environment steps.
- **Time limits.** A transition cut by a time limit still bootstraps on its
  final observation; only a termination stops the bootstrap.
- **Temperature.** The `alpha` step is never gradient-clipped, as in the
  reference SAC, even with `max_grad_norm` set: clipping would turn it into a
  sign step.
- **Ajax additions, not in the paper.** `policy_update_start` and
  `alpha_update_start` hold the actor and `alpha` back until those
  timesteps; with the default `learning_starts=10_000` they change nothing.
  The target-entropy ramp and the expert options above are research features.
- **Memory.** With `memory=`, SAC replays sequences in the R2D2 way
  (Kapturowski et al. 2019): `burn_in` steps warm the memory up, then
  `sequence_length` steps are learned from. It refuses extensions,
  `expert_policy` and cloning pre-training.

## Results

The repository records no learning curves for SAC yet. The probing tests
(`tests/probing`) train SAC on tiny problems with exact answers: the soft
bootstrap with `alpha` pinned, discounting, time limits, reward scale,
memory, extension phases, and a run split by a checkpoint matching the run
left whole. The speed benchmark (`benchmarks/agent_bench.py`) runs SAC with
its defaults for 8,000 steps, below `learning_starts`, so its SAC figure
times collection, not updates.

## API

::: ajax.agents.SAC.SAC.SAC
