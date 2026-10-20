# Changelog

Versions follow semantic versioning; before 1.0, a breaking change bumps
the minor version. Each release is tagged `vX.Y.Z` on main.

## Unreleased

- **Breaking**: brax and MuJoCo Playground are no longer installed with
  Ajax. A project that uses their environments asks for `ajax[playground]`
  (or `ajax[brax]` for Brax alone); without them, building one raises an
  error that names the extra. A gymnax-only install drops 34 of its 93
  packages.
- **TD3** now makes sfujim/TD3's update: its critics' losses are summed
  (they were averaged, which doubled Adam's eps against the gradient), its
  first actor step comes with the `policy_delay`-th update (it came with the
  first), and on boxes other than `[-1, 1]` its critics read actions in the
  environment's units (on gymnax Pendulum, torque). Default TD3 runs differ
  from 0.4.0's; no setting restores the old update.
- **Extras**: `ajax[brax]`, `ajax[playground]` and `ajax[wandb]` (Weights &
  Biases logging).
- **Dependencies**: brax is no longer capped below 0.15 (Ajax uses its
  public `get_environment`); mujoco, mujoco-mjx and gymnasium are no longer
  declared, since Ajax does not import them.

## 0.4.0

The first release since 0.3.1 (April 2025), which held SAC alone. Ajax has
since been reworked; the [documentation](https://yannberthelot.github.io/Ajax/)
describes the result. What a user of 0.3.1 meets:

- **Requirements**: Python 3.12 or newer; JAX 0.10 and flax 0.12.8 or newer,
  with no upper bounds; gymnax 1.0.1 from PyPI instead of a fork. pip
  installs Ajax from its repository as uv does.
- **Agents**: sixteen, each from a published paper: SAC, ASAC, REDQ, TD3,
  AVG, DQN, PQN, PPO, APO, UDRL, APG, DreamerV3, TDMPC2, TDMPC2MultiTask,
  IQL and DAgger, on one training loop, with several seeds in one compiled
  program.
- **Defaults**: each agent starts from its reference code's initialisation
  and optimiser settings where its algorithm page says so, so default runs
  differ from 0.3.1's SAC.
- **Extensions**: research features passed as `extensions=(...)`; an agent
  refuses, when built, one it cannot run.
- **Learning from data**: `train_offline` trains an agent on a transition
  dataset before it acts (TD3+BC through `ImitationLoss`, IQL), and the
  expert-guidance extensions learn from an existing controller.
- **New modules**: `ajax.compare` (AdaStop and confidence sequences),
  transition datasets (`ajax.buffers.dataset`), checkpoints
  (`ajax.checkpoint`), recurrent memory, and the reference comparisons
  under `benchmarks/reference`.
- **Development**: dependency floors without caps, tested weekly at the
  newest and the oldest versions allowed; `uv.lock` refreshed monthly.
