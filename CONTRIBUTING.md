# Contributing to AJAX

Thanks for your interest in contributing. This document covers:

1. [Development setup](#development-setup)
2. [Repository layout](#repository-layout)
3. [The Extension framework](#the-extension-framework)
4. [Adding a new agent](#adding-a-new-agent)
5. [Adding a new Extension](#adding-a-new-extension)
6. [Testing](#testing)
7. [Style and CI](#style-and-ci)

> **Heads-up:** Ajax went through an agent-architecture rework
> (Phases 0–5, branch `agent-architecture-rework`). The pre-rework
> "composable hook" API — `target_modifier`, `runtime_maintenance`,
> `action_pipeline` etc. as `Optional[Callable]` kwargs on each agent
> — has been superseded by the [Extension framework](#the-extension-framework)
> for every research feature; only hooks with a live user remain (see
> [Legacy hook API](#legacy-hook-api-back-compat-only)). **All new
> features should be Extensions, not new hooks.**

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
│   ├── base.py              # ActorCritic base class (env prep, network args,
│   │                        #   .train(), extensions= plumbing)
│   ├── loop.py              # TrainLoop: the shared training loop (collect,
│   │                        #   update gate, post_update, eval + log, resume)
│   ├── recurrent.py         # Replay sampling + R2D2-style sequence burn-in,
│   │                        #   actor_dist / q_values over batch or sequence
│   ├── cloning.py           # BC utilities (pre-train actor/critic from expert)
│   └── <AGENT>/
│       ├── <AGENT>.py       # Public class — __init__, get_make_train
│       ├── train_<AGENT>.py # make_train, update_<step>, loss functions
│       ├── core.py          # (SAC, PPO) The maths that descendants
│       │                    #   (ASAC, REDQ, AVG; APO) import
│       ├── state.py         # flax.struct.dataclass state types
│       └── utils.py         # Agent-specific utilities
├── extensions/
│   ├── base.py              # Extension + ExtensionStack + ExtensionContext
│   │                        #   + fold_<phase> / fold_init helpers
│   ├── expert.py            # ExpertGuidance / OnlineBC / ResidualPolicy /
│   │                        #   ExpertObsAugmentation / JSRLCurriculum
│   ├── target_mods.py       # IBRL / LCBGatedBootstrap / CriticBlend /
│   │                        #   MCVarianceCorrection / ValueBox
│   ├── exploration.py       # EDGEExploration (6 gates)
│   ├── pretrain.py          # MCPretrain / BellmanPretrain / PhiRefresh
│   ├── ensemble.py          # KernelRepulsion
│   └── instrumentation.py   # EVarEst-style measurement: ConditioningMetrics
│                            #   / BiasVoreDecomposition /
│                            #   DiagnosticSnapshots / BiasVorePenalty
├── buffers/, environments/, logging/, modules/, networks/
│   environments/system_class.py   # SystemClass: distributions over EnvParams
│   environments/differentiable.py # closed_loop_rollout: BPTT through gymnax envs
│   environments/model_reference.py# ModelReferenceWrapper: tracking tasks
│   environments/row_collector.py  # obs-aligned row collector (static /
│                                  #   dynamic resets) for agents that own
│                                  #   their collection (world models)
│   modules/pid_head.py            # learnable PID layer on a network output
├── state.py                 # BaseAgentState (carries ext_state: tuple),
│                            #   BaseAgentConfig, shared config dataclasses
├── evaluate.py, log.py      # Eval loops (evaluate, evaluate_policy) + metric
│                            #   logging (evaluate_and_log, maybe_eval_and_log)
├── perf_utils.py            # build_resumable_train (init-or-resume + scan
│                            #   skeleton, resume offset), final_aux_scan/_fori
├── schedule.py              # Optimizer step schedules (warmup_cosine_schedule)
└── wrappers.py              # Env wrappers
```

### Agent anatomy

Every agent follows the same split:

- **`<AGENT>.py`** — the public class. Inherits `ActorCritic` (see [src/ajax/agents/base.py](src/ajax/agents/base.py)), stores algorithm-specific hyperparameters, accepts `extensions: Sequence[Extension] = ()`, and exposes `get_make_train()` returning a `functools.partial` over `make_train`.
- **`train_<AGENT>.py`** — the algorithm: its losses and update steps, and a `make_train(…)` that hands `init` and `update` to the shared `TrainLoop` ([src/ajax/agents/loop.py](src/ajax/agents/loop.py)). The update folds the ExtensionStack at its phases via `stack.fold_<phase>(...)`; the loop folds the rest (`init_state` / `pretrain`, `post_update`, `eval_metrics`). Agents with a loop of their own (APG, UDRL, the world models) build on `build_resumable_train` directly.
- **`core.py`** (SAC and PPO only) — SAC's soft actor-critic maths (init, bootstrap sampling, TD target, critic and actor losses, actor step, temperature, target update); PPO's clipped surrogate (log-prob recompute, entropy bonus, minibatch epochs). Lineage descendants (ASAC, REDQ, AVG; APO, which also builds its state with `init_PPO`) import from here rather than duplicating.
- **`state.py`** — `<AGENT>State` and `<AGENT>Config` extending `BaseAgentState` / `BaseAgentConfig`.

---

## The Extension framework

An **`Extension`** is a composable research feature attachable to any
agent. Each Extension is a `@dataclass(frozen=True)` that holds its own
hyperparameters and overrides only the lifecycle *phases* it touches.
The agent's training loop folds the static `ExtensionStack` through each
phase via `stack.fold_<phase>(...)`. An empty stack is a true no-op
(zero JIT trace cost).

```python
from ajax import SAC
from ajax.extensions.target_mods import IBRL
from ajax.extensions.expert import ExpertGuidance, JSRLCurriculum
from ajax.extensions.instrumentation import (
    ConditioningMetrics, BiasVoreDecomposition, BiasVorePenalty,
)

agent = SAC(
    env_id="Pendulum-v1",
    expert_policy=my_expert,                # SAC keeps a small set of
                                            # deeply-threaded kwargs.
    extensions=[
        ExpertGuidance(expert_policy=my_expert, expert_buffer_n_steps=20_000),
        IBRL(expert_policy=my_expert),
        JSRLCurriculum(expert_policy=my_expert, episode_length=200, decay_frac=0.5),
        BiasVorePenalty(alpha=0.2),
        ConditioningMetrics(),
        BiasVoreDecomposition(),
    ],
)
state, metrics = agent.train(seed=0, n_timesteps=1_000_000)
```

### Phase contract

Each Extension may override any of the following methods (defaults are
all no-op / identity / `0.0` / `{}` / unchanged-state, so an Extension
is exactly as invasive as the phases it overrides):

| Phase | When it fires | Used by (examples) |
| --- | --- | --- |
| `init_state(agent_state, rng) -> pytree` | once, on fresh init | stateful Extensions; `()` default = stateless |
| `pretrain(agent_state, ext_state, ctx)` | once, before training loop | MC pre-train, BC pre-train |
| `on_obs(obs, ext_state, ctx) -> obs` | before network consumes obs | obs augmentation, stop-grad |
| `on_batch(batch, ext_state, ctx) -> batch` | after replay sample | HER relabel, expert mixing |
| `on_target(agent_state, ext_state, batch, target, ctx) -> target` | TD target | IBRL, CriticBlend, LCBGatedBootstrap, MCVarianceCorrection |
| `critic_loss(agent_state, ext_state, batch, ctx) -> scalar` | extra critic-loss term | BiasVorePenalty |
| `actor_loss(agent_state, ext_state, batch, ctx) -> scalar` | extra actor-loss term | OnlineBC |
| `action(agent_state, ext_state, obs, rng, ctx) -> action \| None` | collection-time action | EDGEExploration, ValueBox, JSRLCurriculum |
| `eval_action(agent_state, ext_state, obs, rng, ctx) -> action \| None` | eval-time action | ResidualPolicy |
| `post_update(agent_state, ext_state, ctx) -> (agent_state, ext_state)` | after each update step | PhiRefresh, target-entropy schedules |
| `eval_metrics(agent_state, ext_state, rng, ctx) -> dict` | each eval | ConditioningMetrics, BiasVoreDecomposition, DiagnosticSnapshots |

An agent can declare the phases its training loop folds in
`supported_extension_phases` (a class attribute of `ActorCritic`); an
extension implementing any other phase is then rejected when the agent is
constructed instead of being silently ignored. The check is opt-in: the
default (every phase) checks nothing, and the existing agents have not
opted in yet, so on them an extension implementing a phase they do not
fold (e.g. `critic_loss` on APG) is still silently ignored. New agents
that own their training loop declare their phases.

### Self-binding

When an Extension needs runtime context the agent owns (env / network
config / buffer / `expert_policy`), it implements
`bind_to_agent(env_args, network_args, buffer, agent_config, expert_policy)`
and returns a new (frozen) instance with the context attached. The agent
calls `stack = stack.bind_to_agent(...)` once at init; from that point
on the Extension is fully self-contained. **The base agent must never
hardcode an Extension's hyperparameters or know about a specific
Extension** — see [CLAUDE.md](CLAUDE.md) §"Extensions are self-contained".

### State threading

Stateful Extensions keep their state in `BaseAgentState.ext_state`
(a tuple, one entry per Extension; `()` = stateless). Extensions never
hold mutable state on `self` — only frozen config (e.g. a frozen
expert network).

### Reference: extensions shipped today

| File | Extensions |
| --- | --- |
| `extensions/expert.py` | `ExpertGuidance`, `OnlineBC`, `ImitationLoss`, `ResidualPolicy`, `ExpertObsAugmentation`, `JSRLCurriculum` |
| `extensions/target_mods.py` | `IBRL`, `LCBGatedBootstrap`, `CriticBlend`, `MCVarianceCorrection`, `ValueBox` |
| `extensions/exploration.py` | `EDGEExploration` (6 gates) |
| `extensions/pretrain.py` | `MCPretrain`, `BellmanPretrain`, `PhiRefresh` |
| `extensions/ensemble.py` | `KernelRepulsion` |
| `extensions/instrumentation.py` | `ConditioningMetrics`, `BiasVoreDecomposition`, `BiasVorePenalty`, `DiagnosticSnapshots` |

See [tests/extensions/](tests/extensions/) for behaviour-pinning
tests on each.

### Legacy hook API (back-compat only)

The surviving pre-rework hooks (`Optional[Callable]` kwargs) are those
with a live user: TD3's `action_pipeline`, PPO's `reward_shaping_fn` and
the DQN / PQN variants (`td_target_fn`, `td_loss_fn`, `q_network_cls`).
The rest (`target_modifier`, `obs_preprocessor`, `policy_action_transform`,
`eval_action_transform`, `extra_actor_loss_fn`, `extra_critic_loss_fn`,
`her_relabel_fn`, `init_transform`, `auxiliary_update`,
`extra_eval_metrics`, SAC's `early_termination_condition`, …) were
removed, as was the
online-imitation keyword `imitation_coef` (now the `ImitationLoss`
extension); the `runtime_maintenance` surface and the `use_X` boolean
flags (`ibrl_bootstrap`, `use_critic_blend`,
`use_expert_guided_exploration`, …) went in Phase 5 of the architecture
rework. Use the matching Extensions instead. Tests for the surviving
callable hooks live in
[tests/modules/test_hook_composition.py](tests/modules/test_hook_composition.py).

---

## Adding a new agent

The split-line: **boilerplate goes in shared backbone, RL essence
goes in one file per agent.** A practitioner should be able to read
`train_<AGENT>.py` top-to-bottom like the paper's pseudocode. The
shared backbone (`agents/base.py`, `agents/loop.py`, `perf_utils.py`,
`log.py`, `environments/interaction.py`, `extensions/base.py`) carries
everything that isn't algorithm-specific.

**Lineage rule:** if your agent descends from an existing one (e.g.
REDQ from SAC), **import** the parent's reusable mechanisms from its
`core.py`; do not copy-paste. SAC's `core.py` exports
`init_soft_actor_critic`, `create_alpha_train_state`,
`sample_next_actions`, `compute_td_target`, `critic_loss_fn`,
`soft_policy_loss`, `soft_actor_step`, `update_temperature` and
`update_target_networks` for descendants; PPO's exports
`recompute_log_prob`, `policy_entropy`, `clipped_surrogate`,
`rollout_actions`, `resolve_num_minibatches`, `resolve_clip_coef` and
`run_epochs`; `agents/loop.py` has the replay agents' `critic_step`.

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

In `state.py`, use `flax.struct.dataclass`:

```python
from flax import struct
import jax.numpy as jnp
from ajax.state import BaseAgentState, BaseAgentConfig

@struct.dataclass
class FOOState(BaseAgentState):
    # actor_state, critic_state, collector_state, ext_state, last_rollout, …
    # come from BaseAgentState. Add agent-specific fields here:
    my_field: jnp.ndarray

# BaseAgentConfig already carries gamma, tau, learning_starts, target_entropy,
# reward_scale, expose_recent_rollout. Subclass to add your own.
```

### 3. Write `train_FOO.py`

The algorithm's maths, then a `make_train` that hands it to the shared
loop. An off-policy agent collecting one step per env (TD3, REDQ, ASAC,
AVG):

```python
from collections.abc import Sequence
from ajax.agents.loop import TrainLoop, gradient_step
from ajax.agents.recurrent import sample_replay
from ajax.extensions.base import Extension, ExtensionStack

def update_agent(agent_state, buffer, agent_config, extension_stack, total_timesteps):
    """FOO's update, as in the paper: sample, critic step, actor step, ..."""
    sample_key, rng = jax.random.split(agent_state.rng)
    batch, carries = sample_replay(agent_state, buffer, sample_key)
    ...  # fold extension_stack.fold_on_target / fold_critic_loss / fold_actor_loss
    return agent_state, AuxiliaryLogs(...)

def make_train(
    env_args, network_args, actor_optimizer_args, critic_optimizer_args,
    buffer, agent_config, total_timesteps, num_episode_test,
    run_ids=None, logging_config=None,
    extensions: Sequence[Extension] = (),
):
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key, pretrain_key):  # pretrain_key: one-shot pretraining (cloning)
        return init_FOO(key, ...)

    def update(agent_state, _transition):  # a replay agent samples its buffer
        return update_agent(agent_state, buffer, agent_config, loop.stack, total_timesteps)

    return loop.off_policy(
        init, update, AuxiliaryLogs, agent_config.learning_starts,
        collect_kwargs={"buffer": buffer},
    )
```

An on-policy agent uses `loop.on_policy(init, update, n_steps)`, its
`update(agent_state, rollout, start)` receiving the `(n_steps, n_envs)`
rollout and the state it was collected from, whose carries a recurrent
agent replays it from (PPO, APO, PQN). The loop owns the rest: the extensions' initial state and
pretraining on a fresh run (`ExtensionStack.fold_init`), resuming from a
checkpoint, the warm-up gate before `learning_starts` (the update's metrics
then NaN), `post_update`, evaluation and logging with the extensions'
`eval_metrics`. With a `LoggingConfig` the run evaluates every
`log_frequency` steps and `train` returns `(state, evaluations)`, each
logged key's values per seed and evaluation, whether or not a backend
(`use_wandb`, `use_tensorboard`) records them; the logging worker starts
only for a backend. Without one, `train` returns `(state, None)`.

Key points:
- `extensions=` is the **only** research-feature surface. No per-feature kwargs on `make_train`.
- `stack.fold_<phase>(agent_state, …, step, rng, total_steps)` builds the context and threads `ext_state` in one call, and is a no-op on an empty stack — never inline that boilerplate, and a fold needs no guard. Guard (`if stack:`) a key split drawn only for the extensions, so an agent without extensions keeps its random stream.
- `compose_eval_metrics(user_fn, stack, total_steps)` collapses to `None` when both inputs are no-ops, preserving `evaluate_and_log`'s zero-overhead path.

### 4. Write `FOO.py`

```python
from collections.abc import Sequence
from functools import partial
from typing import Optional
from ajax.agents.base import ActorCritic
from ajax.agents.FOO.train_FOO import make_train
from ajax.extensions.base import Extension

class FOO(ActorCritic):
    name = "FOO"

    def __init__(
        self,
        env_id,
        n_envs: int = 1,
        # … the SAME args ``ActorCritic`` accepts (forwarded to super) …
        # … FOO-specific algorithm hyperparameters (gamma, tau, …) …
        extensions: Sequence[Extension] = (),
    ):
        super().__init__(env_id=env_id, n_envs=n_envs, …, extensions=extensions)
        # store FOO-specific hyperparameters on self

    def get_make_train(self):
        return partial(
            make_train,
            env_args=self.env_args,
            …,
            extensions=tuple(self.extension_stack.extensions),
        )
```

**Anti-pattern:** do NOT add a per-Extension kwarg (`ibrl_bootstrap=True`,
`use_critic_blend=...`, …) to `FOO.__init__`. That is the SAC
back-compat shim Phase 5 deleted. If your feature needs a new
hyperparameter, it belongs as a field on an Extension, not on the agent.

### 5. Export

In `src/ajax/__init__.py`:

```python
from ajax.agents.FOO.FOO import FOO
__all__ = [..., "FOO"]
```

### 6. Tests

- Add to `tests/agents/test_probing.py` — exercises value-net, discounting (if applicable), policy learning on the 3 probing environments.
- Add a per-agent test dir `tests/agents/FOO/test_FOO.py` with a tiny-config smoke run.
- Add a smoke test in `tests/extensions/test_phase3b_extension_smoke.py` (or its successor) exercising `FOO(..., extensions=[CounterExt()])` end-to-end. Mirror the pattern of the other agents there.
- Capture a perf baseline: `JAX_PLATFORMS=cpu poetry run python benchmarks/agent_bench.py --only FOO --out benchmarks/agent_<phase>.jsonl` and check within ±10% of the relevant baseline (see `agent_baseline.jsonl`).

---

## Adding a new Extension

If the feature you want doesn't already exist, add an `Extension` —
**never a new boolean flag or kwarg on an agent**.

### 1. Pick the phase(s)

Map the feature to the [phase contract](#phase-contract). Most
research features touch one or two phases; very rarely three or more.

| Phase | Example feature |
| --- | --- |
| `on_target` | TD-target reshaping (IBRL, CriticBlend) |
| `critic_loss` | Extra additive term (EVarEst penalty) |
| `actor_loss` | BC term, KL regulariser |
| `action` | Exploration override (EDGE, ValueBox) |
| `eval_action` | Eval-time policy modification (residual) |
| `pretrain` | One-shot offline pre-train |
| `post_update` | Periodic state maintenance (φ\* refresh) |
| `eval_metrics` | Pure observability / instrumentation |
| `on_obs` | Pre-network obs transform (stop-grad, encoder adapter) |

### 2. Define the dataclass

```python
# src/ajax/extensions/<topic>.py
from dataclasses import dataclass
from typing import Callable
from ajax.extensions.base import Extension

@dataclass(frozen=True)
class MyFeature(Extension):
    """One-sentence summary of what this Extension does (paper, eq. N).

    Owns its own hyperparameters — never expose them on the agent's
    __init__. The agent only needs to know it has an extensions= list
    to fold.
    """

    # Frozen config fields — extensions are JIT static-arg-friendly only
    # if hashable. @dataclass(frozen=True) makes them hashable for free.
    coefficient: float = 1.0
    schedule_frac: float = 0.5
    expert_policy: Callable | None = None

    # Override only the phases you touch:
    def critic_loss(self, agent_state, ext_state, batch, ctx):
        # Return a scalar (added on top of the agent's vanilla critic loss).
        train_frac = ctx.step / max(ctx.total_steps, 1)
        coeff = self.coefficient * jnp.maximum(1.0 - train_frac / self.schedule_frac, 0.0)
        residual = predict_q(...) - batch.target
        return coeff * jnp.mean(residual ** 2)
```

### 3. (Optional) Bind agent context

If your Extension needs runtime context the agent owns (env config,
network config, buffer, expert_policy), implement `bind_to_agent` —
the agent calls it once at init and from that point the Extension is
fully self-contained.

```python
import dataclasses

@dataclass(frozen=True)
class MyFeatureNeedingBuffer(Extension):
    coefficient: float = 1.0
    _buffer: object = None  # filled by bind_to_agent

    def bind_to_agent(self, *, buffer, **_):
        return dataclasses.replace(self, _buffer=buffer)
```

### 4. (Optional) Stateful Extensions

If your Extension carries mutable state across updates, override
`init_state(agent_state, rng)` to return its initial pytree, and
mutate it via `post_update(agent_state, ext_state, ctx)`. The state
lives in `agent_state.ext_state[i]`, never on `self`.

```python
@dataclass(frozen=True)
class Counter(Extension):
    def init_state(self, agent_state, rng):
        return jnp.asarray(0)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1
```

### 5. Test the contract

Add a behaviour-pinning test in `tests/extensions/`:

```python
def test_my_feature_changes_critic_target():
    agent = SAC(..., extensions=[MyFeature(coefficient=0.2)])
    state, _ = agent.train(seed=0, n_timesteps=80)
    # Assert the feature actually fired (e.g. metric appears, weights
    # changed in the expected direction, fingerprint matches a golden).
```

For features that shadow a previous flag-driven path, capture a
golden checksum and pin numerics at `1e-4` (matches the
`tests/extensions/test_sac_extensions_equivalence.py` contract).

### 6. (No agent code changes needed)

You should **never** need to edit an agent file to make a new
Extension work. If you do, you're either:
- Using the wrong phase (re-read the phase contract), or
- Adding agent-knowledge of your Extension (anti-pattern — re-read
  [CLAUDE.md](CLAUDE.md) §"Extensions are self-contained").

---

## Testing

```bash
poetry run pytest                                         # full suite
poetry run pytest tests/modules/test_hook_composition.py  # hook API contract
poetry run pytest tests/agents/test_probing.py -v         # cross-agent probing (slow)
```

Structure:

- **`tests/extensions/`** — Extension framework + per-Extension behaviour pinning. Includes:
  - `test_extension_framework.py` — base class / `ExtensionStack` / `fold_<phase>` semantics.
  - `test_sac_extensions_equivalence.py` — 20 byte-identical goldens for the migrated SAC features (pinned at `1e-4` against `_sac_equivalence_goldens.json`).
  - `test_instrumentation_extensions.py` — EVarEst measurement extensions on a tiny config.
  - `test_ppo_dqn_pqn_extension_smoke.py`, `test_phase3b_extension_smoke.py` — end-to-end smoke per agent.
- **`tests/modules/test_hook_composition.py`** — API contract for the surviving legacy callable hooks (the small set documented in §"Legacy hook API").
- **`tests/agents/test_probing.py`** — cross-agent behavioral tests on 3 probing environments (from the `ProbingEnvironments` repo). The main regression catcher.
- **`tests/agents/<AGENT>/`** — agent-specific unit tests (loss functions, update steps, agent-specific mechanics).
- **`tests/agents/test_recent_rollout.py`** — Gap A contract for on-policy agents (`expose_recent_rollout` → `agent_state.last_rollout`).
- **`tests/environments/`, `tests/buffers/`, `tests/logging/`, `tests/networks/`** — shared-utility tests.

Coverage gate: `--fail-under=70` in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

### Performance benchmarks

CPU-only standardized harness per agent at
[`benchmarks/agent_bench.py`](benchmarks/agent_bench.py). Baseline at
`benchmarks/agent_baseline.jsonl`; per-phase records at
`benchmarks/agent_phase*.jsonl`. Any agent that regresses beyond ~10%
on the standardized run is investigated to root cause **before** the
change lands.

```bash
JAX_PLATFORMS=cpu poetry run python benchmarks/agent_bench.py \
    --only SAC --out benchmarks/my_change.jsonl --warmup 0 --trials 3
```

**Never share GPUs with running experiments.** Default to
`JAX_PLATFORMS=cpu` for all test + bench runs; use GPU only when the
experiment specifically needs it and the device is idle.

---

## Style and CI

- **Formatter**: `ruff-format` (configured in [pyproject.toml](pyproject.toml)).
- **Linter**: `ruff check` with rule set `I F E W B C RUF`.
- **Types**: `mypy` (optional but encouraged — run `make mypy`).
- **Pre-commit**: `poetry run pre-commit run --all-files` — runs ruff + mypy.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) enforces pre-commit and runs the test suite with coverage on every PR.

### Conventions

- **No new boolean flags. No new agent-level kwargs for features.** If you feel the urge, build an Extension.
- **Extensions are self-contained.** The base agent must never hardcode any Extension's hyperparameters or know about a specific Extension. See [CLAUDE.md](CLAUDE.md) §"Extensions are self-contained" for the full rule + history.
- **Every scalar hyperparameter must be schedulable** — accept either a `float` or `Callable[[int], float]`. See `FloatOrCallable` in [src/ajax/types.py](src/ajax/types.py) and existing agents for the pattern.
- **Probing first.** When adding a feature, run `tests/agents/test_probing.py` to verify no agent regressed before opening a PR.
- **Composable modules, not inheritance.** Prefer adding an Extension over subclassing an agent. The lineage exception is for genuine algorithmic descent (e.g. REDQ extends SAC's actor/temperature machinery) — and even then, import the shared pieces from `core.py` rather than copy-pasting.
- **Heavy changes go on a dedicated branch** with per-commit CI green; see [CLAUDE.md](CLAUDE.md) §"Refactoring & code-quality standards".
