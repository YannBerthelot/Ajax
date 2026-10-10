# Contributing to Ajax

This guide is for changing Ajax: where the code lives, how an agent trains,
how to add an agent or an extension, and how the tests check them. To use
Ajax, read the [README](README.md). [CLAUDE.md](CLAUDE.md) holds the checks a
change must pass and the refactoring rules; this guide does not repeat them.

## Development setup

Install as the [README](README.md#install) says, then run
`uv run pre-commit install` once, so ruff and mypy check every commit. Prefix
commands with `uv run`, or activate `.venv`.

## Where things live

The README's [project layout](README.md#project-layout) gives the overview.
These are the pieces every agent shares:

```
src/ajax/
├── agents/base.py       ActorCritic: environment and network setup, extensions=, train()
├── agents/loop.py       TrainLoop, the shared loop; critic_step and gradient_step
├── agents/recurrent.py  replay buffers and R2D2-style sequence replay for memory=
├── agents/cloning.py    behaviour-cloning pre-training from an expert
├── extensions/base.py   Extension, ExtensionStack, ExtensionContext, the phases
├── environments/interaction.py  collect_experience: act, step and store
├── perf_utils.py        build_resumable_train: initialise or resume, then one jitted scan
├── log.py, evaluate.py  when to evaluate and log; the evaluation episodes
├── state.py             BaseAgentState, BaseAgentConfig, the configuration dataclasses
└── checkpoint.py        save_checkpoint, restore_into
```

**Agent anatomy.** Each folder `src/ajax/agents/<AGENT>/` holds `<AGENT>.py`,
the public class (the paper's hyperparameters and `extensions=`;
`get_make_train()` returns `make_train` with them bound); `train_<AGENT>.py`,
the algorithm and a `make_train` that hands its initialisation and update to
`TrainLoop`; and `state.py`, `<AGENT>State` and `<AGENT>Config`. SAC, PPO and
TDMPC2 also have a `core.py`: the maths their descendants import instead of
copying (ASAC, REDQ and AVG from SAC; APO from PPO; TDMPC2MultiTask from it).

## The shared training loop

The agents share one loop, `TrainLoop`, so collecting experience, waiting for
`learning_starts`, applying the extensions, evaluating, logging and resuming
work the same way everywhere and are fixed in one place. An agent supplies
only its algorithm:

- `init(key, pretrain_key) -> agent_state`: a fresh state; one-shot
  pre-training (behaviour cloning) draws on `pretrain_key`.
- `update(...) -> (agent_state, aux)`: one update. `aux` holds the metrics as
  flax dataclasses nested exactly one level, as TD3's
  `AuxiliaryLogs(policy=..., value=...)`, logged as `value/critic_loss`; the
  logger expects that shape.

and picks one of two iterations:

- **`loop.off_policy(init, update, aux_cls, learning_starts, ...)`** (SAC,
  TD3, REDQ, ASAC, AVG, DQN): each iteration collects one step per
  environment, then, from `learning_starts`, runs `update(agent_state,
  transition)` and folds the extensions' `post_update`. Before
  `learning_starts` the actions are uniform and the metrics are `aux_cls`
  filled with NaN, which the logger drops.
- **`loop.on_policy(init, update, n_steps, ...)`** (PPO, APO, PQN): each
  iteration collects an `n_steps` rollout per environment from the state
  `start`, then runs `update(agent_state, rollout, start)` and folds
  `post_update`. A recurrent agent replays the rollout from `start`'s memory.

The loop does the rest: `init` and the extensions' initial state and
pre-training (`ExtensionStack.fold_init`) on a fresh run, the given state on a
resumed one, then the iterations in `build_resumable_train`'s jitted scan,
evaluating and logging every `log_frequency` steps (with `eval_metrics`).

The world models (DreamerV3, TDMPC2) also give `off_policy` their own
`collect(agent_state, tick)`, `n_updates(tick)` (such as a train ratio) and
`Evaluation`. UDRL runs its own iteration on `TrainLoop.train`; APG and the
offline `TDMPC2MultiTask` call `build_resumable_train` themselves, with
`fresh_state` and the `post_update` fold.

`ActorCritic.train` runs the seeds at once (`jax.vmap`). With a
`LoggingConfig` it returns `(state, evaluations)`, each logged key's values
per seed and evaluation, kept whether or not a backend (`use_wandb`,
`use_tensorboard`, the only case that starts the logging worker) records them.
Without one it does not evaluate and returns `(state, None)`; APG returns
every update's metrics instead. `initial_state=state` continues a run.

## Extensions

An extension is a research feature (an expert to learn from, an exploration
rule, an extra loss term, a measurement) that plugs into any agent without
changing it, so an agent's surface stays its algorithm's hyperparameters plus
`extensions=()` (see CLAUDE.md, "Extensions are self-contained"). The README
[lists the extensions](README.md#extensions) Ajax ships.

An extension is a frozen dataclass subclassing `Extension`. It holds its own
settings and overrides only the *phases* it touches; the other phases keep
their defaults, which change nothing. An agent folds its `ExtensionStack`
through a phase in list order, mostly with a `stack.fold_<phase>(...)` helper
(`fold_init`, `fold_on_target`, `fold_critic_loss`, `fold_actor_loss`,
`fold_post_update`, `fold_eval_metrics`) that builds the `ExtensionContext`
(`step`, `rng`, `total_steps`) and threads the extensions' state. On an empty
stack a helper changes nothing (a loss term is `0.0`), so an agent without
extensions traces nothing extra and never guards a fold.

| Phase | What it does | Folded today by |
| --- | --- | --- |
| `init_state(agent_state, rng)` | returns the extension's state (`()`: none) | every agent, fresh runs only |
| `pretrain(agent_state, ext_state, ctx)` | one-shot step before training | every agent, fresh runs only |
| `on_obs(obs, ext_state, ctx)` | transforms an observation | SAC, in its actor loss only |
| `on_batch(batch, ext_state, ctx)` | transforms a sampled batch | no agent yet |
| `on_target(agent_state, ext_state, batch, target, ctx)` | transforms the TD or value target | SAC, REDQ, ASAC, AVG, TD3, DQN, PQN, PPO, APO |
| `critic_loss(agent_state, ext_state, batch, ctx)` | adds a term to the critic loss | SAC, REDQ, ASAC, AVG, TD3, PPO, APO; DQN (without gradient) |
| `actor_loss(agent_state, ext_state, batch, ctx)` | adds a term to the actor loss | SAC, REDQ, ASAC, AVG, TD3, PPO, APO, APG, UDRL |
| `action(agent_state, ext_state, obs, rng, ctx)` | overrides the collection action (`None` defers) | SAC with an `expert_policy`, for extensions with an `action_slot` |
| `eval_action(agent_state, ext_state, obs, rng, ctx)` | overrides the evaluation action | no agent generically (SAC wires `ResidualPolicy` itself) |
| `post_update(agent_state, ext_state, ctx)` | runs after each update | every agent |
| `eval_metrics(agent_state, ext_state, rng, ctx)` | adds metrics to each evaluation | every agent |

`tests/probing/test_extensions.py` checks the phases on the agents with
test-only extensions of known effect, and keeps each gap above visible as a
strict xfail naming it.

An agent may list the phases it folds in `supported_extension_phases`, so an
extension implementing another phase is rejected when the agent is built
instead of being silently ignored. DreamerV3, TDMPC2 and TDMPC2MultiTask do;
the others keep the default (every phase), so they still ignore such phases.

What changes during training lives in `agent_state.ext_state` (one entry per
extension), never on `self`: extensions are static arguments of the compiled
program, so they stay hashable and unchanged. An extension that needs what
the agent builds (environment, network config, buffer, discount) overrides
`bind_to_agent(**agent_context)` to return a new instance holding it, as
`ExpertObsAugmentation` does; only SAC and DreamerV3 call it. The batch a
phase receives is a dictionary whose keys differ by agent: read the agent's
fold call before relying on one.

## Adding an agent

**Boilerplate goes in the shared code; the algorithm goes in one file**, so a
reader can follow `train_<AGENT>.py` like the paper's pseudocode. **Lineage
rule:** an agent descending from another (REDQ from SAC) imports the parent's
maths from its `core.py`. Use `critic_step` (a replay critic step) and `gradient_step` (an
optimiser step) from `agents/loop.py` rather than writing your own.

These templates for an agent `FOO` follow the replay agents TD3 and REDQ; an
on-policy agent follows APO instead (no buffer, `loop.on_policy`).

**1. `state.py`.** `BaseAgentState` already holds the random key, the actor,
critic and collector states, the update count and the extensions' state. A
replay agent's config extends `RecurrentReplayConfig`, a `BaseAgentConfig`
for sequence replay; `kw_only` lets fields without defaults follow others.

```python
@partial(struct.dataclass, kw_only=True)  # Partial from jax.tree_util
class FOOState(BaseAgentState):
    """FOO carries only actor + critic + collector."""


@partial(struct.dataclass, kw_only=True)
class FOOConfig(RecurrentReplayConfig):
    gamma: float
    learning_starts: int = 100
    reward_scale: float = 1.0
```

**2. `train_FOO.py`.** The paper's maths, then `make_train`. The critic step
goes through `critic_step`, which folds `on_target` and `critic_loss`:

```python
def update_value_functions(agent_state, batch, agent_config, extension_stack, total_timesteps):
    key, rng = jax.random.split(agent_state.rng)
    target_q = compute_foo_td_target(agent_state, key, batch, agent_config)  # the paper's target

    def value_loss(params, target_q):  # -> (loss, aux)
        return value_loss_function(params, agent_state.critic_state, batch.obs, batch.action, target_q)

    critic_state, aux = critic_step(
        agent_state, batch, target_q, value_loss, extension_stack, key, total_timesteps,
        rewards=batch.reward, gamma=agent_config.gamma, reward_scale=agent_config.reward_scale,
    )
    return agent_state.replace(rng=rng, critic_state=critic_state), aux


def make_train(
    env_args, actor_optimizer_args, critic_optimizer_args, network_args, buffer,
    agent_config, total_timesteps, num_episode_test,
    run_ids=None, logging_config=None, extensions=(),
):
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key, pretrain_key):
        return init_FOO(key, env_args, actor_optimizer_args, critic_optimizer_args, network_args, buffer)

    def update(agent_state, _transition):
        # The step just collected is in the buffer: FOO samples it from there.
        return update_agent(agent_state, buffer, agent_config, loop.stack, total_timesteps)

    return loop.off_policy(
        init, update, AuxiliaryLogs, agent_config.learning_starts,
        collect_kwargs={"buffer": buffer},
    )
```

`update_agent` samples a batch (`sample_replay`), takes the critic step, then
the actor step, whose loss adds `extension_stack.fold_actor_loss(...)`, and
returns the state with an `AuxiliaryLogs` of both steps' metrics.
`ActorCritic.train` calls `make_train` with the keyword names above, so keep
them. Split a random key for the extensions only under `if stack:`, so an
agent without extensions keeps its random stream.

**3. `FOO.py`.**

```python
class FOO(ActorCritic):
    """FOO (Author et al., year) for continuous action spaces; defaults as in the paper."""

    name: str = "FOO"
    # The phases the loop folds, plus those FOO's update folds.
    supported_extension_phases: frozenset = frozenset(
        {"pretrain", "post_update", "eval_metrics", "on_target", "critic_loss", "actor_loss"}
    )

    def __init__(
        self, env_id: str | EnvType, n_envs: int = 1, gamma: float = 0.99,
        buffer_size: int = int(1e6), batch_size: int = 100, learning_starts: int = int(1e4),
        extensions: Sequence[Extension] = (),
    ) -> None:
        self.config = {**locals()}  # what train() logs with the run
        self.config.update({"algo_name": "FOO"})
        super().__init__(env_id=env_id, n_envs=n_envs, extensions=extensions)
        self.agent_config = FOOConfig(gamma=gamma, learning_starts=learning_starts)
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )

    def get_make_train(self) -> Callable:
        return partial(make_train, buffer=self.buffer, extensions=tuple(self.extension_stack.extensions))
```

Do **not** add a keyword for a feature (`use_critic_blend=True`, ...): it
belongs on an extension. Set `supports_memory = True` only once the agent
trains with `memory=`; until then `ActorCritic` rejects it.

**4. Export and document.** Import it in `src/ajax/__init__.py`, add it to
`__all__`, and add a row to the README's agent table.

**5. Tests and benchmark.** `tests/agents/FOO/`: the maths on hand-computed
values and a tiny run. `tests/probing/`: presets in `agents.py`, then the
cases that fit (`test_framework.py` fails while an exported agent has none).
`tests/agents/test_agent_pins.py`: a tiny run's fingerprint, so a later
restructuring can show it changed nothing. `benchmarks/agent_bench.py`: an
entry in `AGENTS` and a baseline (see [Performance](#performance)).

## Adding an extension

Add an extension, never a flag or a keyword on an agent.

1. **Pick the phases** from the [table](#extensions) and check that the agents
   you target fold them. If one does not, adding the fold to that agent is a
   change of its own, and the agent still must not know about your extension.
2. **Write the dataclass** in `src/ajax/extensions/<topic>.py`, its settings
   as fields. This one counts updates and logs the count; it runs as is:

```python
from dataclasses import dataclass

import jax.numpy as jnp

from ajax import PPO
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import LoggingConfig


@dataclass(frozen=True)
class UpdateCounter(Extension):
    """Counts the updates and logs the count at each evaluation."""

    name: str = "update-counter"

    def init_state(self, agent_state, rng):
        return jnp.asarray(0)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"Counter/updates": ext_state}


agent = PPO("CartPole-v1", n_steps=128, extensions=(UpdateCounter(),))
logging = LoggingConfig(config={}, log_frequency=1024, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4096, logging_config=logging)
print(state.ext_state[0])  # the updates each seed made: [9 9]
print(evaluations["Counter/updates"])  # per seed, at each evaluation: [2 4 6 8]
```

3. **Test it** in `tests/extensions/`: the feature fires and does what its
   paper says (a metric appears, a term moves the loss the right way).

**Callable hooks.** A few `Optional[Callable]` keywords predate extensions and
stay because downstream projects use them: TD3's `action_pipeline`, PPO's
`reward_shaping_fn`, DQN's `td_target_fn` and `td_loss_fn`, PQN's
`td_loss_fn`. `tests/modules/test_hook_composition.py` pins them. Add none.

## Testing

*Unit tests* (`tests/<area>/`, `tests/agents/<AGENT>/`) check pieces in
isolation and that every agent and extension runs. *Pins* record behaviour so
a restructuring can show it changed nothing: `tests/agents/test_agent_pins.py`
(tiny runs' fingerprints), `tests/extensions/test_sac_extensions_equivalence.py`
and `tests/test_downstream_api.py` (the names downstream projects import).
*Probing tests* check that agents learn the right thing: `tests/probing` and,
older, `tests/agents/test_probing.py`. *Slow tests* (`@pytest.mark.slow`) are
the long trainings.

```bash
uv run pytest --collect-only -q   # everything imports and collects
uv run pytest tests/agents/TD3    # the tests of what you changed
uv run pytest -m "not slow"       # skip the long trainings
make ci   # what CI runs (ci-precommit, ci-test, ci-slow, ci-probe), GPU hidden
```

The `make` targets hide the GPU (`CPU_ENV`) so tests never compete with a
running experiment; elsewhere set `JAX_PLATFORMS=cpu` while a GPU is busy.

**CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on CPU for
every pull request and push to `main`: *Pre-commit*; *Tests* (three shards:
the tests not marked slow, probing excluded, under coverage; the last shard
runs whatever the first two do not name, so a new test file always runs);
*Coverage* (the shards combined, at least 70%); *Slow tests* (without
coverage, which would lengthen them); *Probing* (five shards; the last takes
any new probe file). *Lint and Test*, the one check branch protection
requires, passes only when all of them passed.

### The probing tests

A unit test shows that code runs, not that an agent learns what it should. A
probe trains an agent on a tiny problem whose answer is known exactly (a
value, a best action, a count) and compares what it learned with the right
answer and named wrong ones (a discount applied twice), so a failure names the
mistake.

`envs.py` builds the tiny problems, `agents.py` holds each agent's presets
(pinned by `DIGEST`), `runs.py` trains every seed in one compiled program,
`readouts.py` reads values and actions off a trained agent and `oracles.py`
computes the right and wrong answers. In `verdict.py`, a `Query` is a
reading's right and named wrong answers and a `Case` adds the budget and
tolerances; a check trains seeds 0-7: 7 or more within tolerance pass, 4 or
fewer fail, otherwise seeds 8-15 run and 13 of 16 must pass. `faults.py` and
`faults/<module>.jsonl` plant known bugs in a scratch copy of the source, each
naming the tests that must catch it. The `test_*.py` modules group the cases
by subject (value chains, episode ends, bookkeeping, extensions, memory, world
models, resuming a split run); `test_framework.py` tests the harness.

A new case's budget and tolerances come from calibration on seeds no check
uses (`uv run python -m tests.probing.verdict ladder|choose|certify ...`;
`verdict.py` describes the procedure); then its module's `ANSWER_DIGEST` is
re-pinned. A known bug stays visible as a strict xfail naming the defect,
never as a looser tolerance. The directory is capped at 7,000 lines.

## Performance

CLAUDE.md ("Performance no-regression guardrail") sets the rule; here is the
check. `benchmarks/agent_bench.py` times a fixed run per agent; `--compare`
fails when one is slower than `--tol` (10% by default) against a baseline.
Always pass `--out`: it defaults to the baseline file.

```bash
export JAX_PLATFORMS=cpu
uv run python benchmarks/agent_bench.py --only TD3 --out benchmarks/my_change.jsonl
uv run python benchmarks/agent_bench.py --compare benchmarks/agent_baseline.jsonl benchmarks/my_change.jsonl
```

## Style

Pre-commit ([.pre-commit-config.yaml](.pre-commit-config.yaml)) runs on every
commit and in CI: ruff lint with rules `I F E W B C RUF ARG` (`ARG`, unused
arguments, is off under `tests/`), ruff-format (settings in
[pyproject.toml](pyproject.toml)), and mypy on everything but `benchmarks/`.
