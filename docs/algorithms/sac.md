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
| Recurrent memory (`memory=`) | Yes, without extensions or `expert_policy` |
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
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (Adam's epsilon is TensorFlow's 1e-8) |
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
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, divide by the spread of the discounted return instead of the single reward's |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

### Expert options

These options remain on SAC because they reach the network's input size, the
collection or the action pipeline, where extensions do not reach yet. Most
concern the expert; the last four are research switches of SAC's own. The
expert is written in the environment's units and receives the raw
(unnormalised) observations; Ajax reads its actions in the agent's units. See
[Expert guidance](../guide/expert-guidance.md).

| Argument | Default | What it does |
| --- | --- | --- |
| `expert_policy` | `None` | the expert used in training; `None` is plain SAC |
| `eval_expert_policy` | `None` | an expert used only for the evaluation's comparison metrics |
| `expert_buffer_n_steps` | `20_000` | expert steps per env written to the buffer before training |
| `expert_fraction` | `0.7` | during warm-up, the chance a step takes the expert's action instead of a uniform one |
| `use_pid_policy` | `False` | the actor outputs the gains of the expert's PID (needs `learnable_fields` on the expert; refuses `augment_obs_with_expert_action`) |
| `augment_obs_with_expert_action`, `augment_obs_with_expert_state` | `False` | append the expert's action or internal state to the observation |
| `use_train_frac` | `False` | append the training fraction to the observation |
| `target_entropy_initial_per_dim`, `target_entropy_ramp_frac` | `None`, `0.5` | move the target entropy linearly from this start to `target_entropy_per_dim` over that fraction of training |
| `normalize_obs_running` | `False` | `normalize_observations` under its old name: warns, and will be removed ([Environments](../guide/environments.md#normalisation)) |
| `store_policy_action` | `False` | store the policy's sampled action in the replay instead of the executed one (an off-policy-correctness ablation) |

Cloning the expert into the actor before training is the `CloningPretrain`
extension, at the actor's learning rate; fitting a frozen expert critic on
the expert's Monte-Carlo returns is `MCPretrain`, which only SAC supports.
`jsrl_curriculum` is deprecated: it has no
effect and warns, since SAC counts each environment's steps in its episode
whenever an extension reads them (`JSRLCurriculum`).
`skip_critic_pretrain=True` is a no-op kept for a downstream project; `False`
raises.

## Notes

- **Actions.** The actor is a `tanh`-squashed Gaussian, so its actions lie
  in `[-1, 1]`; Ajax maps them onto the environment's bounds, so on gymnax
  Pendulum (torque in `[-2, 2]`) SAC reaches the whole range. The warm-up's
  uniform actions cover the whole box too. The replay stores the agent's
  `[-1, 1]` action. Evaluation uses the mean action.
- **Initialisation.** As both reference codes (`haarnoja/sac`, and
  `rail-berkeley/softlearning` for the learned temperature): Glorot-uniform
  kernels and zero biases in every layer. They draw the mean and the
  log-std from one output layer; Ajax's two heads each take Glorot's bound
  for their own width (about 1% wider with 256 inputs). With `memory=`, SAC
  keeps its previous initialisation until Ni et al.'s is audited.
- **Standard deviation.** The log-std is clipped to `[-20, 2]` and the
  standard deviation is its exponential, as in `haarnoja/sac` and in
  `softlearning` at the time of the learned-temperature paper. Later
  `softlearning` (its TensorFlow 2 port, 2020) uses `softplus(x) + 1e-5`
  instead; Ajax keeps the paper-era form.
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
  `sequence_length` steps are learned from. It refuses extensions and
  `expert_policy`.

## CQL and Cal-QL: offline

CQL (Kumar, Zhou, Tucker and Levine, *Conservative Q-Learning for Offline
Reinforcement Learning*, NeurIPS 2020,
[arXiv:2006.04779](https://arxiv.org/abs/2006.04779)) is SAC with a critic
term that pushes the critic down on actions away from the data's and up on
the data's. Cal-QL (Nakamoto et al., *Cal-QL: Calibrated Offline RL
Pre-Training for Efficient Online Fine-Tuning*, NeurIPS 2023,
[arXiv:2303.05479](https://arxiv.org/abs/2303.05479)) floors that push at
the data's Monte-Carlo returns, so fine-tuning online does not start from
values far too low. In Ajax both are extensions on SAC, trained with the
[offline mode](../guide/training.md#training-offline):

```python
from ajax import SAC
from ajax.extensions.offline import CalQL, ConservativeQ

agent = SAC(
    "Pendulum-v1",
    actor_learning_rate=1e-4,  # the references' policy rate
    policy_update_start=0,  # the offline mode counts updates: start at once
    alpha_update_start=0,
    learning_starts=0,
    expert_buffer_n_steps=0,
    extensions=(ConservativeQ(alpha=5.0),),  # or CalQL(alpha=5.0)
)
# state, evaluations = agent.train_offline(dataset, n_updates=1_000_000)
```

The term is the authors' importance-sampled CQL(H)
(aviralkumar2907/CQL at `d67dbe9`, `cql.py:234-264`): for each critic, the
logsumexp of its values at 10 uniform actions and 10 of the policy's at
`s` and at `s'`, each less its log-density, minus its value at the data's
action, times `alpha` (5 in the references' gym runs). Cal-QL takes the
maximum of each policy action's value and the row's return to go first
(nakamotoo/Cal-QL at `ac6eafe`, `JaxCQL/conservative_sac.py:227-240`). The
return to go is the transition dataset's, at its `gamma`. Online rows have
none, so they are not floored; the reference computes them as episodes
end. Deviation: SAC's target keeps its entropy term, which the CQL
reference's `deterministic_backup` drops.

## Results

The repository records no learning curves for SAC yet. Unit tests
(`tests/agents/SAC` and the contracts every agent shares in `tests/agents`)
check its pieces against exact contracts, without learning. The probes
(`tests/probing`) train it briefly on tiny environments with known answers:
the soft bootstrap with `alpha` pinned, discounting, time limits, reward
scale, memory, every extension phase it folds, and a run split by a
checkpoint matching the run left whole. On the CPU speed benchmark
(`benchmarks/agent_bench.py`: Pendulum, 4 envs, 8,000 steps, updates from
step 1,000) SAC ran at 2,218 steps per second of execution (1,635 with
compilation) at main `a8a13c2` (see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.SAC.SAC.SAC
