# Specifications

Written by `benchmarks/reference/refcmp_publish.py specs` from
`benchmarks/reference/refcmp_specs.py`, which a test keeps it equal to.
Each comparison is set before its runs and changed only as a new
version (see the [protocol](protocol.md)).

## PPO, version 2

Ajax's PPO against stable-baselines3's PPO, MlpPolicy.

- Environment: gymnax CartPole-v1, a port of Gymnasium's CartPole-v1 (dynamics, termination, 500-step limit); SB3 steps it through refcmp_sb3.gymnax_vec_env.
- Budget: 200,704 steps per seed.
- Evaluation: 10 episodes of the deterministic policy, every 14,336 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: ±25.
- Spec hash: `ab1379d3e1e9ebb6`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run PPO --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `n_envs` | `8` | `n_envs` | `8` |
| `n_steps` | `256` | `n_steps` | `256` |
| `batch_size` | `256` | - |  |
| `n_epochs` | `10` | `n_epochs` | `10` |
| `learning_rate` | `0.0003` | `actor_learning_rate`, `critic_learning_rate` | `0.0003`, `0.0003` |
| `gamma` | `0.99` | `gamma` | `0.99` |
| `gae_lambda` | `0.95` | `gae_lambda` | `0.95` |
| `clip_range` | `0.2` | `clip_range` | `0.2` |
| `ent_coef` | `0.0` | `ent_coef` | `0.0` |
| `vf_coef` | `0.5` | `vf_coef` | `1.0` |
| `max_grad_norm` | `0.5` | `max_grad_norm` | `0.5` |
| `net_arch` | `{'pi': [64, 64], 'vf': [64, 64]}` | - |  |
| `activation` | `'tanh'` | - |  |

Ajax settings without a reference counterpart: `num_minibatches=8`, `actor_architecture=('64', 'tanh', '64', 'tanh')`, `critic_architecture=('64', 'tanh', '64', 'tanh')`, `adam_eps=1e-05`, `encoder_kernel_init='orthogonal(1.4142135623730951)'`.

Not matched:


## PPO-inverted_pendulum, version 2

Ajax's PPO against brax.training.agents.ppo, with Brax's settings for inverted_pendulum (notebooks/training.ipynb) but the length.

- Environment: Brax's inverted_pendulum (generalized pipeline, 1000-step episodes), a MuJoCo InvertedPendulum simulated by Brax: the same Env on both sides.
- Budget: 819,200 steps per seed.
- Evaluation: 128 episodes of the deterministic policy, every 163,840 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: ±50.
- Spec hash: `5f421eaf04254d20`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run PPO-inverted_pendulum --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `num_envs` | `2048` | `n_envs` | `2048` |
| `batch_size` | `1024` | - |  |
| `num_minibatches` | `32` | `num_minibatches` | `32` |
| `unroll_length` | `5` | `unroll_length` | `5` |
| `num_updates_per_batch` | `4` | `n_epochs` | `4` |
| `learning_rate` | `0.0003` | `actor_learning_rate`, `critic_learning_rate` | `0.0003`, `0.0003` |
| `entropy_cost` | `0.01` | `ent_coef` | `0.01` |
| `discounting` | `0.97` | `gamma` | `0.97` |
| `gae_lambda` | `0.95` | `gae_lambda` | `0.95` |
| `clipping_epsilon` | `0.3` | `clip_range` | `0.3` |
| `vf_loss_coefficient` | `0.5` | `vf_coef` | `0.5` |
| `max_grad_norm` | `None` | `max_grad_norm` | `None` |
| `normalize_advantage` | `True` | `normalize_advantage` | `True` |
| `normalize_observations` | `True` | `normalize_observations` | `True` |
| `reward_scaling` | `10` | - |  |
| `episode_length` | `1000` | `episode_length` | `1000` |
| `action_repeat` | `1` | `action_repeat` | `1` |

Ajax settings without a reference counterpart: `n_steps=80`, `adam_eps=1e-08`, `actor_architecture=('32', 'swish', '32', 'swish', '32', 'swish', '32', 'swish')`, `critic_architecture=('256', 'swish', '256', 'swish', '256', 'swish', '256', 'swish', '256', 'swish')`, `encoder_kernel_init='lecun_uniform'`, `mean_kernel_init='lecun_uniform'`, `squash=True`, `log_std_state_independent=False`, `log_std_init=-0.3651`.

Not matched:

- rewards: Brax scales them by 10 (reward_scaling); Ajax's PPO cannot
- time limits: Brax leaves a truncated step out of the advantage (bootstrap_on_timeout off: this env marks every fall as a time out, so it would bootstrap falls); Ajax bootstraps it
- auto-reset: Brax restarts each env from its own first state; Ajax draws a new one
- std, computed from the observation on both sides: softplus(x) + 0.001 in Brax, exp(x) clipped to [-20, 2] in Ajax; its weights start lecun_uniform in Brax, zero in Ajax
- the value head's initialisation (lecun_uniform in Brax)

## DQN, version 1

Ajax's DQN against stable-baselines3's DQN, MlpPolicy.

- Environment: gymnax CartPole-v1, a port of Gymnasium's CartPole-v1 (dynamics, termination, 500-step limit); SB3 steps it through refcmp_sb3.gymnax_vec_env.
- Budget: 100,000 steps per seed.
- Evaluation: 10 episodes of the deterministic policy, every 5,000 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: none registered yet.
- Spec hash: `74aef7221269ae79`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run DQN --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `learning_rate` | `0.001` | `learning_rate` | `0.001` |
| `buffer_size` | `100000` | `buffer_size` | `100000` |
| `learning_starts` | `1000` | `learning_starts` | `1000` |
| `batch_size` | `64` | `batch_size` | `64` |
| `gamma` | `0.99` | `gamma` | `0.99` |
| `train_freq` | `1` | - |  |
| `gradient_steps` | `1` | `n_gradient_steps` | `1` |
| `target_update_interval` | `500` | `target_update_interval` | `500` |
| `exploration_fraction` | `0.2` | `epsilon_decay_frac` | `0.2` |
| `exploration_initial_eps` | `1.0` | `epsilon_start` | `1.0` |
| `exploration_final_eps` | `0.05` | `epsilon_end` | `0.05` |
| `max_grad_norm` | `10.0` | `max_grad_norm` | `10.0` |
| `net_arch` | `[128, 128]` | - |  |

Ajax settings without a reference counterpart: `n_envs=1`, `tau=1.0`, `architecture=('128', 'relu', '128', 'relu')`, `td_loss='huber'`.

Not matched:

- weight initialisation (torch default vs Ajax's)
- Adam eps: 1e-5 in Ajax, 1e-8 in the reference

## SAC, version 1

Ajax's SAC against stable-baselines3's SAC, MlpPolicy.

- Environment: gymnax Pendulum-v1, a port of Gymnasium's Pendulum-v1 (200-step episodes); SB3 steps it through refcmp_sb3.gymnax_vec_env.
- Budget: 20,000 steps per seed.
- Evaluation: 10 episodes of the deterministic policy, every 1,000 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: none registered yet.
- Spec hash: `3639b4ebc6ea4944`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run SAC --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `learning_rate` | `0.001` | `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `0.001`, `0.001`, `0.001` |
| `buffer_size` | `20000` | `buffer_size` | `20000` |
| `learning_starts` | `100` | `learning_starts` | `100` |
| `batch_size` | `256` | `batch_size` | `256` |
| `tau` | `0.005` | `tau` | `0.005` |
| `gamma` | `0.99` | `gamma` | `0.99` |
| `train_freq` | `1` | - |  |
| `gradient_steps` | `1` | - |  |
| `ent_coef` | `'auto'` | - |  |
| `net_arch` | `[256, 256]` | - |  |

Ajax settings without a reference counterpart: `n_envs=1`, `alpha_init=1.0`, `target_entropy_per_dim=-1.0`, `policy_update_start=0`, `alpha_update_start=0`, `actor_architecture=('256', 'relu', '256', 'relu')`, `critic_architecture=('256', 'relu', '256', 'relu')`.

Not matched:

- weight initialisation
- initial policy std: exp(-1) in Ajax, about 1 in SB3
- Adam eps: 1e-5 in Ajax, 1e-8 in the reference

## TD3, version 1

Ajax's TD3 against stable-baselines3's TD3, MlpPolicy.

- Environment: gymnax Pendulum-v1, a port of Gymnasium's Pendulum-v1 (200-step episodes); SB3 steps it through refcmp_sb3.gymnax_vec_env.
- Budget: 20,000 steps per seed.
- Evaluation: 10 episodes of the deterministic policy, every 1,000 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: none registered yet.
- Spec hash: `7d9a3c8804e712ba`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run TD3 --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `learning_rate` | `0.001` | `actor_learning_rate`, `critic_learning_rate` | `0.001`, `0.001` |
| `buffer_size` | `20000` | `buffer_size` | `20000` |
| `learning_starts` | `100` | `learning_starts` | `100` |
| `batch_size` | `256` | `batch_size` | `256` |
| `tau` | `0.005` | `tau` | `0.005` |
| `gamma` | `0.99` | `gamma` | `0.99` |
| `train_freq` | `1` | - |  |
| `gradient_steps` | `1` | - |  |
| `policy_delay` | `2` | `policy_delay` | `2` |
| `target_policy_noise` | `0.2` | `target_policy_noise` | `0.2` |
| `target_noise_clip` | `0.5` | `target_noise_clip` | `0.5` |
| `action_noise_std` | `0.1` | `exploration_noise` | `0.1` |
| `net_arch` | `[400, 300]` | - |  |

Ajax settings without a reference counterpart: `n_envs=1`, `actor_architecture=('400', 'relu', '300', 'relu')`, `critic_architecture=('400', 'relu', '300', 'relu')`.

Not matched:

- weight initialisation
- Adam eps: 1e-5 in Ajax, 1e-8 in the reference

## PQN, version 1

Ajax's PQN against purejaxql 47af6d7: vendor/purejaxql/pqn_gymnax.py.

- Environment: gymnax CartPole-v1, a port of Gymnasium's CartPole-v1.
- Budget: 491,520 steps per seed.
- Evaluation: 128 episodes of the deterministic policy, every 24,576 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: ±25.
- Spec hash: `87990c92b943e1a8`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run PQN --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `TOTAL_TIMESTEPS` | `491520` | - |  |
| `TOTAL_TIMESTEPS_DECAY` | `491520` | - |  |
| `NUM_ENVS` | `32` | `n_envs` | `32` |
| `NUM_STEPS` | `64` | `n_steps` | `64` |
| `EPS_START` | `1.0` | `epsilon_start` | `1.0` |
| `EPS_FINISH` | `0.2` | `epsilon_end` | `0.2` |
| `EPS_DECAY` | `0.2` | `epsilon_decay_frac` | `0.2` |
| `NUM_MINIBATCHES` | `16` | `num_minibatches` | `16` |
| `NUM_EPOCHS` | `4` | `n_epochs` | `4` |
| `NORM_INPUT` | `False` | - |  |
| `HIDDEN_SIZE` | `256` | - |  |
| `NUM_LAYERS` | `2` | - |  |
| `NORM_TYPE` | `'layer_norm'` | - |  |
| `LR` | `0.0001` | `learning_rate` | `0.0001` |
| `MAX_GRAD_NORM` | `10` | `max_grad_norm` | `10.0` |
| `LR_LINEAR_DECAY` | `True` | - |  |
| `REW_SCALE` | `0.1` | `reward_scale` | `0.1` |
| `GAMMA` | `0.99` | `gamma` | `0.99` |
| `LAMBDA` | `0.95` | `q_lambda` | `0.95` |
| `ENV_NAME` | `'CartPole-v1'` | - |  |
| `ENV_KWARGS` | `{}` | - |  |
| `TEST_DURING_TRAINING` | `True` | - |  |
| `TEST_INTERVAL` | `0.05` | - |  |
| `TEST_NUM_ENVS` | `128` | - |  |
| `EPS_TEST` | `0.0` | - |  |
| `WANDB_MODE` | `'disabled'` | - |  |

Ajax settings without a reference counterpart: `architecture=('256', 'relu', '256', 'relu')`, `lr_decay=('linear', 1e-20, 15360)`, `td_loss='half_squared'`.

Not matched:

- optimiser: RAdam (eps 1e-8) in purejaxql, Adam (eps 1e-5) in Ajax
- weight initialisation (lecun_normal vs Ajax's)
- time limits: the reference treats them as terminations
- evaluation: purejaxql scores the last episode each of its 128 test envs ends within one 500-step window, Ajax one episode per env

Deviations found by the conformance checks (`tests/conformance`):

- per-step epsilon (ajax, collection, epsilon-greedy actions): Ajax reads epsilon at the collector's timestep, so it moves at every env step; purejaxql reads it once per update. While epsilon decays, Ajax explores slightly less within a rollout. Code: src/ajax/agents/DQN/train_DQN.py linear_epsilon; vendor/purejaxql/pqn_gymnax.py:189. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_exploration_differs_only_by_the_per_step_epsilon`.
- time limits (convention, Q(lambda) targets): purejaxql treats a time-limit cut as a termination (the target is the reward); Ajax bootstraps it. Only the targets of episodes cut by a time limit differ. Code: refcmp_jax._FiveValueStep; src/ajax/agents/PQN/utils.py compute_q_lambda_targets. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_time_limit_cuts_change_only_their_episodes_targets`.
- Q(lambda) off-by-one (reference defect, Q(lambda) targets): purejaxql's backward pass starts with the last step's bootstrap, V(next_obs) of the last transition, as the next value of the second-to-last one, whose next value is V(obs) of the last; that target, and through lambda those before it in its episode, differ from Q(lambda). Ajax computes Q(lambda). Reported upstream as purejaxql issue #12. Code: vendor/purejaxql/pqn_gymnax.py:246-254. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_targets_differ_only_by_purejaxqls_off_by_one`.

## PQN-Asterix, version 1

Ajax's PQN against purejaxql 47af6d7: vendor/purejaxql/pqn_gymnax.py with config/alg/pqn_minatar.yaml but the length. That config is for pqn_minatar.py, whose CNN Ajax's CNN encoder cannot build (VALID padding, LayerNorm after the convolution); both sides use pqn_gymnax.py's MLP on the flattened image instead.

- Environment: gymnax Asterix-MinAtar, a port of MinAtar's Asterix with a 1000-step limit and no sticky actions.
- Budget: 1,228,800 steps per seed.
- Evaluation: 128 episodes of the deterministic policy, every 61,440 steps; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: ±5% of the reference's mean score.
- Spec hash: `cba75a779763e3f9`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run PQN-Asterix --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `TOTAL_TIMESTEPS` | `1228800` | - |  |
| `TOTAL_TIMESTEPS_DECAY` | `1228800` | - |  |
| `NUM_ENVS` | `128` | `n_envs` | `128` |
| `NUM_STEPS` | `32` | `n_steps` | `32` |
| `EPS_START` | `1.0` | `epsilon_start` | `1.0` |
| `EPS_FINISH` | `0.05` | `epsilon_end` | `0.05` |
| `EPS_DECAY` | `0.1` | `epsilon_decay_frac` | `0.1` |
| `NUM_MINIBATCHES` | `32` | `num_minibatches` | `32` |
| `NUM_EPOCHS` | `2` | `n_epochs` | `2` |
| `NORM_TYPE` | `'layer_norm'` | - |  |
| `LR` | `0.0005` | `learning_rate` | `0.0005` |
| `MAX_GRAD_NORM` | `10` | `max_grad_norm` | `10.0` |
| `LR_LINEAR_DECAY` | `True` | - |  |
| `REW_SCALE` | `1.0` | `reward_scale` | `1.0` |
| `GAMMA` | `0.99` | `gamma` | `0.99` |
| `LAMBDA` | `0.65` | `q_lambda` | `0.65` |
| `ENV_NAME` | `'Asterix-MinAtar'` | - |  |
| `ENV_KWARGS` | `{}` | - |  |
| `TEST_DURING_TRAINING` | `True` | - |  |
| `TEST_INTERVAL` | `0.05` | - |  |
| `TEST_NUM_ENVS` | `128` | - |  |
| `EPS_TEST` | `0.0` | - |  |
| `WANDB_MODE` | `'disabled'` | - |  |

Ajax settings without a reference counterpart: `architecture=('128', 'relu', '128', 'relu')`, `lr_decay=('linear', 1e-20, 19200)`, `td_loss='half_squared'`.

Not matched:

- optimiser: RAdam (eps 1e-8) in purejaxql, Adam (eps 1e-5) in Ajax
- weight initialisation (lecun_normal vs Ajax's)
- time limits: the reference treats them as terminations
- evaluation: purejaxql scores the last episode each of its 128 test envs ends within one 1000-step window, Ajax one episode per env

Deviations found by the conformance checks (`tests/conformance`):

- per-step epsilon (ajax, collection, epsilon-greedy actions): Ajax reads epsilon at the collector's timestep, so it moves at every env step; purejaxql reads it once per update. While epsilon decays, Ajax explores slightly less within a rollout. Code: src/ajax/agents/DQN/train_DQN.py linear_epsilon; vendor/purejaxql/pqn_gymnax.py:189. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_exploration_differs_only_by_the_per_step_epsilon`.
- time limits (convention, Q(lambda) targets): purejaxql treats a time-limit cut as a termination (the target is the reward); Ajax bootstraps it. Only the targets of episodes cut by a time limit differ. Code: refcmp_jax._FiveValueStep; src/ajax/agents/PQN/utils.py compute_q_lambda_targets. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_time_limit_cuts_change_only_their_episodes_targets`.
- Q(lambda) off-by-one (reference defect, Q(lambda) targets): purejaxql's backward pass starts with the last step's bootstrap, V(next_obs) of the last transition, as the next value of the second-to-last one, whose next value is V(obs) of the last; that target, and through lambda those before it in its episode, differ from Q(lambda). Ajax computes Q(lambda). Reported upstream as purejaxql issue #12. Code: vendor/purejaxql/pqn_gymnax.py:246-254. Pinned by `tests/conformance/test_pqn_purejaxql.py::test_targets_differ_only_by_purejaxqls_off_by_one`.

## APG, version 1

Ajax's APG against brax.training.agents.apg, on refcmp_brax.GymnaxAsBrax.

- Environment: gymnax Pendulum-v1 with transition gradients, a port of Gymnasium's Pendulum-v1.
- Budget: 2,048,000 steps per seed.
- Evaluation: 10 episodes of the deterministic policy, once, at the end; a seed's score is the last one's mean return.
- AdaStop: n = 4, K = 24, alpha = 0.05.
- Margin: none registered yet.
- Spec hash: `c74efa6c58e75716`.
- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py run APG --cache DIR`

| Reference setting | Value | Ajax setting | Value |
|---|---|---|---|
| `episode_length` | `200` | - |  |
| `policy_updates` | `1000` | - |  |
| `horizon_length` | `32` | `horizon` | `32` |
| `num_envs` | `64` | `n_envs` | `64` |
| `learning_rate` | `0.0001` | `learning_rate` | `0.0001` |

Ajax settings without a reference counterpart: `beta_1=0.7`, `beta_2=0.95`, `max_grad_norm=None`, `lr_decay=('exponential', 0.997)`, `actor_architecture=('32', 'relu', '32', 'relu', '32', 'relu', '32', 'relu')`.

Not matched:

- episodes: Brax continues them across updates from scrambled start times and resets each env to its own first state (64 starts); Ajax restarts them from fresh states at every update
- Brax trains on sampled actions (tanh-Gaussian, var_scale 0.1); Ajax's controller is deterministic
- Brax clips each gradient entry to [-1, 1]; its loss is the mean reward over the horizon, Ajax's the summed return
- network: ELU with LayerNorm and orthogonal(0.01) in Brax, ReLU in Ajax
- Adam eps: 1e-5 in Ajax, 1e-8 in the reference
