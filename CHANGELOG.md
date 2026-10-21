# Changelog

Versions follow semantic versioning; before 1.0, a breaking change bumps
the minor version. Each release is tagged `vX.Y.Z` on main and published to
PyPI as `ajax-rl`.

## Unreleased

- **ASAC** now does what the authors' code does (`darer/ASAC.py`): a
  termination drops the next value in the critic's target (it bootstrapped
  through it); the termination penalty follows their rule, `p_0` times the
  larger reward of the batch's first two rows; `theta` averages the reward
  without the penalty, at the actions the actor step sampled, which the
  temperature step reads too; the entropy terms are measured from
  `log(1/nA)`; and the policy's log-probability takes stable-baselines3's
  `+1e-6` in the `tanh` correction. Runs with terminations learn other
  values; continuing tasks change less.
- **REDQ** now does what the authors' code does (`watchernyu/REDQ`): its
  first update's critic targets and actor loss read the temperature 0.2,
  which the code's constructor leaves until its first temperature step, and
  the policy's log-probability takes the code's `+1e-6` in the `tanh`
  correction. Later updates read `exp(log_alpha)` as before.
- **AVG** reads its log-std head as a log-variance, as the authors' code
  does (`gauthamvasan/avg`): the policy's standard deviation is
  `exp(out / 2)`, its clip bounding the log-variance. Runs start with a
  different spread and learn otherwise.
- **PQN** now does what purejaxql does: a time limit ends an episode in
  its returns (it bootstrapped there), epsilon is read once per update
  instead of at every step, and the rollout's second-to-last step reads
  the last step's bootstrap, as purejaxql's backward pass does. Its
  reproduction comparisons become version 2.
- **APO** ends an episode at a time limit, its next value dropped, and
  leaves that step out of the loss, as xtma/apo does (`bootstrap_timelimit`):
  it bootstrapped on the final observation and trained on the step. With
  `normalize_advantage`, the advantages use the other steps' mean and
  unbiased standard deviation, floored at `1e-6`.
- `NetworkConfig.squash_log_det_eps` sets the guard inside the `tanh`
  log-det, `log(1 - tanh(u)^2 + eps)`, for an agent whose reference uses
  one; 0, the exact log-det, by default. `NetworkConfig.log_variance_head`
  reads the log-std head as a log-variance, for the same reason.

## 0.4.0

The first release since 0.3.1 (April 2025), which held SAC alone. Ajax has
since been reworked; the [documentation](https://yannberthelot.github.io/Ajax/)
describes the result. What a user of 0.3.1 meets:

- **Requirements**: Python 3.12 or newer; JAX 0.10 and flax 0.12.8 or newer,
  with no upper bounds; gymnax 1.0.1 from PyPI instead of a fork. Ajax
  declares only the packages it imports, so gymnasium, mujoco, mujoco-mjx,
  pandas, matplotlib, pydantic and scipy are no longer installed with it.
  pip installs Ajax from PyPI or from its repository.
- **Distribution name**: Ajax is on PyPI as `ajax-rl` (`pip install
  ajax-rl`); the import is unchanged, `import ajax`. A project that declares
  `ajax` from git or a path renames it `ajax-rl`.
- **Breaking**: Brax and MuJoCo Playground are no longer installed with
  Ajax. A project that uses their environments asks for `ajax-rl[playground]`
  (which brings Brax) or `ajax-rl[brax]`; without them, building one of their
  environments raises an error that names the extra.
- **Extras**: `ajax-rl[cuda]` (JAX with CUDA, Linux), `ajax-rl[brax]`,
  `ajax-rl[playground]`, `ajax-rl[compare]`, `ajax-rl[reproduction]` and
  `ajax-rl[wandb]`.
- **Agents**: sixteen, each from a published paper: SAC, ASAC, REDQ, TD3,
  AVG, DQN, PQN, PPO, APO, UDRL, APG, DreamerV3, TDMPC2, TDMPC2MultiTask,
  IQL and DAgger, on one training loop, with several seeds in one compiled
  program. Each starts from its reference code's initialisation and
  optimiser settings where its algorithm page says so. TD3 makes the update
  of sfujim/TD3, its critics reading actions in the environment's units, as
  REDQ's do.
- **Extensions**: research features passed as `extensions=(...)`; an agent
  refuses, when built, one it cannot run.
- **Learning from data**: `train_offline(dataset, n_updates)` trains a replay
  agent (SAC, TD3, REDQ, ASAC, DQN, IQL) on a transition dataset before it
  acts, then `train` fine-tunes it online. On it run TD3+BC
  (`ImitationLoss(None, target="replay", alpha=2.5)`, the paper's actor
  loss), ReBRAC (with `TargetBC`), CQL and Cal-QL (`ConservativeQ`, `CalQL`,
  for a policy with a density) and IQL. GAIL (`AdversarialReward` on PPO)
  and DAgger imitate with no reward, and the expert-guidance extensions
  learn from an existing controller.
- **New modules**: `ajax.compare` (AdaStop and confidence sequences),
  transition datasets (`ajax.buffers.dataset`), checkpoints
  (`ajax.checkpoint`), recurrent memory, and the reference comparisons
  under `benchmarks/reference`.
- **Development**: dependency floors without caps, tested weekly at the
  newest and the oldest versions allowed; `uv.lock` refreshed monthly.
  Conformance checks (`tests/conformance`) compare one update of an agent
  with its original implementation's: PQN, TD3, TD3+BC, REDQ and AVG so
  far. The tests come in
  two tiers (CONTRIBUTING.md, Testing): the slow tier, every test that
  trains, also runs daily on main, on a pull request labelled
  `run-slow-tier` and before each release; tags come only from the Release
  workflow, once the slow tier passed on main's head.

### Default changes

For projects that pin an Ajax commit: the merges since 2026-10-17 that move
default runs. Each pull request says how.

| Agent | What changed | Pull requests |
| --- | --- | --- |
| SAC, TD3, REDQ, ASAC, DQN, IQL, DAgger | the replay also draws each environment's newest row | #210 |
| SAC | Xavier-uniform layers, as softlearning; Adam eps 1e-8 | #193, #197 |
| ASAC | torch's layer init; temperature fixed at 0.2; learning rates 1e-4 and 5e-4, gradient clip 10; Adam eps 1e-8 | #193, #196, #197 |
| REDQ | Xavier-uniform layers; Adam eps 1e-8; critics read actions in the environment's units | #193, #197, #227 |
| TD3 | the 2019 code's settings (256-256, learning rate 3e-4, batch 256, 25,000 warm-up steps); torch's layer init; Adam eps 1e-8; summed critic losses, first actor step at the `policy_delay`-th update, critics in the environment's units | #193, #196, #197, #220, #224 |
| TD3+BC, ReBRAC | `ImitationLoss(alpha=...)` is the paper's actor loss, `λ` times the agent's own plus the cloning term, actions compared in the environment's units | #225 |
| AVG | orthogonal layers; Adam eps 1e-8 | #193, #197 |
| DQN | torch's layer init; centered RMSProp with a Huber loss, as Mnih et al. | #193, #197 |
| PQN | flax's `lecun_normal` layers; RAdam after a gradient clip of 10 | #193, #197 |
| PPO | flat minibatches, GAE once per rollout; ppo2's orthogonal layers; Adam eps 1e-5, one gradient clip over both networks, `vf_coef` 0.5 | #146, #159, #183 |
| PPO with memory | PureJaxRL's layer init; `num_minibatches` sets the sequence length | #170, #188 |
| PPO, APO, UDRL, discrete actions | the logits head starts at orthogonal(0.01) | #147 |
| APO | its official code's actor head, 64-64 tanh layers, joint optimiser step and entropy bonus 0.01 | #192 |
| APG | Adam eps 1e-8; `APG.contextual_controller` does not clip its gradient | #175 |
| twin and ensemble critics, with `max_grad_norm` | each critic clipped on its own | #196 |
