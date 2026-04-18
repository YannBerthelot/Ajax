# Contributing to AJAX

Thanks for your interest in contributing. This document covers:

1. [Development setup](#development-setup)
2. [Repository layout](#repository-layout)
3. [The hook API](#the-hook-api)
4. [Adding a new agent](#adding-a-new-agent)
5. [Adding a new composable module (hook)](#adding-a-new-composable-module-hook)
6. [Testing](#testing)
7. [Style and CI](#style-and-ci)

---

## Development setup

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
poetry install
poetry run pre-commit install
```

All commands below assume `poetry run` or an activated `poetry shell`.

---

## Repository layout

```
src/ajax/
├── agents/
│   ├── base.py              # ActorCritic base class (env prep, network args, .train())
│   ├── cloning.py           # BC utilities (pre-train actor/critic from expert transitions)
│   └── <AGENT>/
│       ├── <AGENT>.py       # Public class — __init__, get_make_train
│       ├── train_<AGENT>.py # make_train, update_<step> functions, loss functions
│       ├── state.py         # flax.struct.dataclass state types
│       └── utils.py         # Agent-specific utilities
├── buffers/, environments/, logging/, modules/, networks/
├── state.py                 # Shared config dataclasses (EnvironmentConfig, NetworkConfig, …)
├── evaluate.py, log.py      # Eval loop + metric logging
├── schedule.py              # Schedulable scalars (constant, linear, exponential, polynomial)
└── wrappers.py              # Env wrappers
```

### Agent anatomy

Every agent follows the same two-file split:

- **`<AGENT>.py`** — the public class. Inherits `ActorCritic` (see [src/ajax/agents/base.py](src/ajax/agents/base.py)), stores hyperparameters, and exposes `get_make_train()` returning a `functools.partial` over `make_train`.
- **`train_<AGENT>.py`** — the JIT-compiled training logic: `make_train(…)` builds a closure; `training_iteration` is the body of the `jax.lax.scan`; loss functions live here.

---

## The hook API

Agents replace boolean feature flags with **composable `Optional[Callable]` hooks** passed at `__init__`. When `None`, the feature is inactive. When provided, the callable is forwarded into the compiled `make_train` and JIT-compiled as a static argument (one compilation per unique hook signature).

### Hooks exposed today

| Hook | Applies to | Purpose |
| --- | --- | --- |
| `pid_actor_config` | all | Use a PID-gain-predicting actor instead of raw-action actor |
| `action_pipeline` | SAC, ASAC, REDQ, AVG, PPO, APO | Custom action selection during `collect_experience` (exploration, expert mixing, residual RL, PID override) |
| `obs_preprocessor` | SAC family | Transform observations before the actor/critic sees them (e.g. stop-gradient on expert-action dims) |
| `policy_action_transform` | SAC family | Transform actions in the policy loss (e.g. residual RL inside Q evaluation) |
| `target_modifier` | SAC family | Modify the Bellman target (IBRL, critic blend, MC variance correction) |
| `eval_action_transform` | SAC family | Rewrap `apply_fn` at eval time (residual / PID at deploy) |
| `runtime_maintenance` | SAC | Periodic maintenance every K steps (φ* refresh) |

See [tests/modules/test_hook_composition.py](tests/modules/test_hook_composition.py) for the full contract each agent must uphold:

1. Every hook defaults to `None`.
2. Passed hooks are stored on the instance with the same name.
3. `get_make_train()` forwards every stored hook via the `functools.partial`.

### Factories

Each hook has a corresponding `make_<hook>(…)` factory in `src/ajax/agents/SAC/train_SAC.py`. These factories turn primitive flags into a composed callable and return `None` when all branches are inactive. Example:

```python
from ajax.agents.SAC.train_SAC import make_action_pipeline

pipeline = make_action_pipeline(
    expert_policy=expert_fn,
    recurrent=False,
    env_args=env_args,
    use_expert_guided_exploration=True,
    exploration_decay_frac=0.5,
    total_timesteps=1_000_000,
)
```

---

## Adding a new agent

Let's say you want to add an agent called `FOO`.

### 1. Create the directory

```
src/ajax/agents/FOO/
├── __init__.py
├── FOO.py
├── train_FOO.py
└── state.py
```

### 2. Define the state

In `state.py`, use `flax.struct.dataclass` to bundle the agent's mutable state:

```python
from flax import struct
import jax.numpy as jnp
from ajax.state import BaseAgentState

@struct.dataclass
class FOOState(BaseAgentState):
    # actor_state, critic_state, collector_state, … come from BaseAgentState
    # Add agent-specific fields here:
    my_field: jnp.ndarray
```

### 3. Write `train_FOO.py`

This file holds:

- **Loss functions** — pure functions taking `params` and returning `(loss, aux)`.
- **Update steps** — `@jax.jit`-compatible functions taking a state and returning the updated state.
- **`training_iteration`** — the scan body: `collect_experience` → update(s) → log.
- **`make_train(env_args, network_args, …, hook1=None, hook2=None, …)`** — builds and returns `train(seed, n_timesteps, …)`.

Pattern:

```python
def make_train(
    env_args, network_args, optimizer_args, cloning_args,
    # … hyperparameters …
    action_pipeline: Optional[Callable] = None,
):
    def train(seed, n_timesteps, logging_config, num_episode_test):
        agent_state = init_FOO(…)

        def training_iteration(runner_state, _):
            # collect_experience
            # update
            # log
            return new_runner_state, metrics

        final_state, metrics = jax.lax.scan(
            training_iteration, (agent_state, …), None, length=n_timesteps // n_envs,
        )
        return final_state, metrics

    return train
```

### 4. Write `FOO.py`

The public class:

```python
from ajax.agents.base import ActorCritic
from ajax.agents.FOO.train_FOO import make_train

class FOO(ActorCritic):
    name = "FOO"

    def __init__(
        self,
        env_id,
        # … base args forwarded to super().__init__() …
        # … FOO-specific hyperparameters …
        pid_actor_config: Optional[PIDActorConfig] = None,
        action_pipeline: Optional[Callable] = None,
    ):
        super().__init__(env_id=env_id, …)
        self.pid_actor_config = pid_actor_config
        self.action_pipeline = action_pipeline
        # store other hyperparameters on self

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            # … forward stored hyperparameters + hooks …
            action_pipeline=self.action_pipeline,
        )
```

### 5. Export

In `src/ajax/__init__.py`:

```python
from ajax.agents.FOO.FOO import FOO
__all__ = [..., "FOO"]
```

### 6. Tests

Add the agent to `tests/agents/test_probing.py` — this exercises value-net, discounting (if applicable), and policy learning on the 3 probing environments. Add the agent to `AGENT_HOOKS` in [tests/modules/test_hook_composition.py](tests/modules/test_hook_composition.py) to enforce the hook API contract.

---

## Adding a new composable module (hook)

If the feature you want doesn't fit an existing hook, add a new one. Guideline: a hook should replace N scattered boolean flags with one callable that composes them cleanly.

### 1. Design the callable signature

Keep the signature narrow and name the hook for what it *does*, not what it *is*. For example:

```python
# In train_<AGENT>.py
def make_my_hook(flag_a, flag_b, some_context):
    if not flag_a and not flag_b:
        return None

    def hook(state, batch, rng):
        # compose behavior here
        return transformed_batch, aux
    return hook
```

Return `None` when every branch is inactive — callers check with `if hook is not None` at Python level, so the compiled graph only contains active branches.

### 2. Thread the hook through `make_train`

Add `my_hook: Optional[Callable] = None` to `make_train`'s signature. Add it to the inner function's `static_argnames` wherever it's used. Apply it inside the training iteration when non-`None`.

### 3. Expose it on the agent class

Add an `Optional[Callable] = None` parameter to `<AGENT>.__init__`, store on `self`, and forward in `get_make_train()`.

### 4. Test the contract

Add the hook name to `tests/modules/test_hook_composition.py`'s `AGENT_HOOKS` tuple for each agent that supports it. The three contract tests (default-None, stored-correctly, forwarded-by-get_make_train) will automatically cover the new hook.

### 5. (Optional) Demonstrate it

If the hook captures a published technique, add a probing test that shows the technique measurably changes behavior — e.g. `tests/agents/REDQ/test_redq_subset.py` exercises the random-subset target.

---

## Testing

```bash
poetry run pytest                                         # full suite
poetry run pytest tests/modules/test_hook_composition.py  # hook API contract
poetry run pytest tests/agents/test_probing.py -v         # cross-agent probing (slow)
```

Structure:

- **`tests/modules/test_hook_composition.py`** — API contract for the hook system.
- **`tests/agents/test_probing.py`** — cross-agent behavioral tests on 3 probing environments (from the user's `ProbingEnvironments` repo). This is the main regression catcher.
- **`tests/agents/<AGENT>/`** — agent-specific unit tests (loss functions, update steps, agent-specific mechanics like REDQ's subset target).
- **`tests/environments/`, `tests/buffers/`, `tests/logging/`, `tests/networks/`** — shared-utility tests.

Coverage gate: `--fail-under=70` in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## Style and CI

- **Formatter**: `ruff-format` (configured in [pyproject.toml](pyproject.toml)).
- **Linter**: `ruff check` with rule set `I F E W B C RUF`.
- **Types**: `mypy` (optional but encouraged — run `make mypy`).
- **Pre-commit**: `poetry run pre-commit run --all-files` — runs ruff + mypy.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) enforces pre-commit and runs the test suite with coverage on every PR.

### Conventions

- **No new boolean flags.** If you feel the urge, build a hook.
- **Every scalar hyperparameter must be schedulable** — accept either a `float` or `Callable[[int], float]`. See [src/ajax/schedule.py](src/ajax/schedule.py) and existing agents for the pattern.
- **Probing first.** When adding a feature, run `tests/agents/test_probing.py` to verify no agent regressed before opening a PR.
- **Composable modules, not inheritance.** Prefer passing a hook over subclassing an agent.
