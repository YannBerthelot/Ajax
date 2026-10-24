# Contributing to Ajax

This guide is for changing Ajax: where the code lives, how an agent trains,
how to add an agent or an extension, and how the tests check them. To use
Ajax, read the [README](README.md). [CLAUDE.md](CLAUDE.md) holds the checks a
change must pass and the refactoring rules; this guide does not repeat them.

## Development setup

Install as the [README](README.md#installation) says, then run
`uv run pre-commit install` once, so ruff and mypy check every commit. Prefix
commands with `uv run`, or activate `.venv`.

## Where things live

[Architecture](docs/developer/architecture.md#the-package) gives the overview.
These are the pieces every agent shares:

```
src/ajax/
├── agents/base.py       ActorCritic: environment and network setup, extensions=, train()
├── agents/families.py   declared agents: fields as hyperparameters, the replay and on-policy families
├── agents/loop.py       TrainLoop, the shared loop (brax's forced resets included)
├── agents/steps.py      critic_step, actor_step, gradient_step, joint_step, when, the target update
├── agents/minibatch.py  the on-policy rollout split, minibatches and epochs
├── agents/critics.py    the Q-critics' regression (continuous and discrete actions)
├── agents/diagnostics.py  KL, explained variance, policy std
├── agents/recurrent.py  replay buffers and R2D2-style sequence replay for memory=
├── agents/cloning.py    the behaviour-cloning loss and fit (CloningPretrain's)
├── extensions/base.py   Extension, ExtensionStack, ExtensionContext, the phases
├── environments/interaction.py  collect_experience: act, step and store
├── perf_utils.py        build_resumable_train: initialise or resume, then one jitted scan
├── log.py, evaluate.py  when to evaluate and log; the evaluation episodes
├── state.py             BaseAgentState, BaseAgentConfig, the configuration dataclasses
└── checkpoint.py        save_checkpoint, restore_into
```

**Agent anatomy.** A declared agent (IQL, CrossQ, TD3, PPO, APO, PQN) is one file,
`src/ajax/agents/<AGENT>/<AGENT>.py`: its hyperparameters as dataclass
fields, its `init` and `update`, its equations, its state and metrics (and,
until 0.5.0, the `<AGENT>Config` view it had before), on a family of
`agents/families.py` that builds the rest (the constructor, the run config,
the replay or the rollouts, the train function); its old modules are shims.
The other agents' folders hold `<AGENT>.py`, the public class (the paper's
hyperparameters and `extensions=`; it names its `make_train` and fills
`train_kwargs`, which `get_make_train()` binds);
`train_<AGENT>.py`, the algorithm and a `make_train` that hands its
initialisation and update to `TrainLoop`; and `state.py`, `<AGENT>State` and
`<AGENT>Config`. SAC and TDMPC2 also have a `core.py`: the maths their
descendants import instead of copying (ASAC, AVG, CrossQ, REDQ and TQC from
SAC; TDMPC2MultiTask from TDMPC2), under the lineage rule below. A declared
parent's equations are in its one file: APO imports PPO's from
`PPO/PPO.py`.

## The shared training loop

The agents share one loop, `TrainLoop`, so collecting experience, waiting for
`learning_starts`, applying the extensions, evaluating, logging and resuming
work the same way everywhere and are fixed in one place. An agent supplies
only its algorithm:

- `init(key) -> agent_state`: a fresh state. One-shot pretraining (behaviour
  cloning) is an extension's, on a key the loop splits off.
- `update(...) -> (agent_state, aux)`: one update. `aux` holds the metrics as
  flax dataclasses nested exactly one level, as TD3's
  `AuxiliaryLogs(policy=..., value=...)`, logged as `value/critic_loss`; the
  logger expects that shape.

and picks one of two iterations:

- **`loop.off_policy(init, update, learning_starts, ...)`** (SAC,
  TD3, REDQ, ASAC, AVG, DQN): each iteration collects one step per
  environment, then, from `learning_starts`, runs `update(agent_state,
  transition)` and folds the extensions' `post_update`. Before
  `learning_starts` the actions are uniform and the metrics have the
  update's structure, filled with NaN, which the logger drops.
- **`loop.on_policy(init, update, n_steps, ...)`** (PPO, APO, PQN): each
  iteration collects an `n_steps` rollout per environment from the state
  `start`, then runs `update(agent_state, rollout, start)` and folds
  `post_update`. A recurrent agent replays the rollout from `start`'s memory.

The loop does the rest: `init` and the extensions' initial state and
pre-training (`ExtensionStack.fold_init`) on a fresh run, the given state on a
resumed one, then the iterations in `build_resumable_train`'s jitted scan,
evaluating and logging every `log_frequency` steps (with `eval_metrics`).

Continuous agents act in `[-1, 1]`; never scale to the env's bounds yourself.
Every Ajax env step maps the action onto them (`agent_action_to_env`), and
`ActorCritic` reads an expert, written in env units, in the same `[-1, 1]`
(`ExpertInAgentUnits`): replay, losses and extensions see one unit.

The world models (DreamerV3, TDMPC2) also give `off_policy` their own
`collect(agent_state, tick)`, `n_updates(tick, agent_state)` (such as a train
ratio) and `Evaluation`. UDRL runs its own iteration on `TrainLoop.train`; APG
and the offline `TDMPC2MultiTask` call `build_resumable_train` themselves,
with `fresh_state` and the `post_update` fold.

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
`extensions=()` (see CLAUDE.md, "Extensions are self-contained"). The
[extensions guide](docs/guide/extensions.md#catalogue) lists the ones Ajax
ships.

An extension is a frozen dataclass subclassing `Extension`. It holds its own
settings and overrides only the *phases* it touches; the other phases keep
their defaults, which change nothing. An agent applies a phase of its
`ExtensionStack` in list order with one helper per phase (`apply_reward`,
`apply_on_target`, `apply_critic_loss`, `apply_actor_loss`,
`apply_post_update`; the loop calls `fold_init` and `fold_eval_metrics`)
that builds the `ExtensionContext` (`step`, `rng`, `total_steps`) and
threads the extensions' state. On an empty stack a helper changes nothing
(a loss term is `0.0`), so an agent without extensions traces nothing extra
and never guards a call.

| Phase | What it does | Folded today by |
| --- | --- | --- |
| `init_state(agent_state, rng)` | returns the extension's state (`()`: none) | every agent, fresh runs only |
| `pretrain(agent_state, ext_state, ctx)` | one-shot step before training | every agent, fresh runs only |
| `on_target(agent_state, ext_state, batch, target, ctx)` | transforms the TD or value target | SAC, REDQ, ASAC, AVG, TD3, DQN, PQN, PPO, APO |
| `critic_loss(agent_state, ext_state, batch, ctx)` | adds a term to the critic loss | SAC, REDQ, ASAC, AVG, TD3, DQN, PQN, PPO, APO |
| `actor_loss(agent_state, ext_state, batch, ctx)` | adds a term to the actor loss | SAC, REDQ, ASAC, AVG, TD3, PPO, APO, APG, UDRL |
| `action(agent_state, ext_state, obs, rng, ctx)` | overrides the collection action (`None` defers) | SAC with an `expert_policy`, for extensions with an `action_slot` |
| `eval_action(agent_state, ext_state, obs, rng, ctx)` | overrides the evaluation action | no agent yet |
| `post_update(agent_state, ext_state, ctx)` | runs after each update | every agent |
| `eval_metrics(agent_state, ext_state, rng, ctx)` | adds metrics to each evaluation | every agent |

`tests/probing/test_extensions.py` checks the phases on the agents with
test-only extensions of known effect, and that an extension implementing a
phase the agent does not fold is rejected.

Every agent lists the phases it folds in `supported_extension_phases`: the
shared loop's `LOOP_PHASES` (`pretrain`, `post_update`, `eval_metrics`, in
`ajax.agents.loop`) plus those its update folds; the base class's default is
empty. An extension implementing any other phase is rejected when the agent
is built instead of being silently ignored. It declares in
`phase_batch_keys` the keys its folds put in the extensions' batches beyond
the common ones (`COMMON_BATCH_KEYS`): an agent on `critic_step` and
`actor_step` declares their sets (`REPLAY_BATCH_KEYS`, or `TARGET_KEYS`,
`CRITIC_LOSS_KEYS` and `ACTOR_STEP_KEYS` with its own rows'), plus any it
adds. An extension reading an undeclared key is refused when the agent is
built.

What changes during training lives in `agent_state.ext_state` (one entry per
extension), never on `self`: extensions are static arguments of the compiled
program, so they stay hashable and unchanged. An extension that needs what
the agent builds (environment, network config, buffer, discount) overrides
`bind_to_agent(**agent_context)` to return a new instance holding it, as
`MCPretrain` does; SAC, TD3, REDQ, APO and DreamerV3 call it. The batch a
phase receives is a dictionary: the keys every agent puts in it are listed in
`ajax/extensions/base.py`; read the agent's fold call before relying on any
other.

## Adding an agent

**Boilerplate goes in the shared code; the algorithm goes in one file**, so a
reader can follow the agent's file (`<AGENT>/<AGENT>.py` for a declared
agent) like the paper's pseudocode. **Lineage rule:** an agent imports from
another agent only when it is a variant of that agent's algorithm (REDQ of
SAC, APO of PPO), and then only the parent's equations. Anything else two
agents share (an init, a network, a loss helper, an exploration schedule)
goes to a shared module: `agents/steps.py`, `agents/minibatch.py`,
`agents/critics.py`, `agents/diagnostics.py`, `networks/`. A shared module
imports no agent.
`tests/agents/test_agent_imports.py` holds the lineages and each parent's
equations, and fails on any other cross-agent import. Use `critic_step` (a
replay critic step), `actor_step` (an actor step) and `gradient_step` (an
optimiser step) from `agents/steps.py` rather than writing your own.

A new agent `FOO` declares itself on a family of `agents/families.py`: a
replay agent on `ReplayAgent`, as IQL, CrossQ and TD3 do (read
`agents/IQL/IQL.py` first); an on-policy agent on `OnPolicyAgent`, as PPO,
APO and PQN do (read `agents/PPO/PPO.py` first, and the on-policy family
below). The other agents keep the older three-file form (`state.py`,
`train_<AGENT>.py` with a `make_train`, `<AGENT>.py`) until their batch
moves them.

**1. `FOO/FOO.py`, the one file.** Its hyperparameters are the fields of a
dataclass, declared once with the paper's defaults: the family generates the
constructor (keyword-only after `env_id`), logs the run config, builds the
optimisers, the replay and the train function (`ReplayAgent.make_train`, on
`loop.off_policy`; an on-policy agent's is `OnPolicyAgent.make_train`, on
`loop.on_policy`). The file reads top-down: the class, its equations in the
order `update` calls them, then its state and metrics. The template below
is a replay agent's; an on-policy agent's file has the same parts.

```python
@dataclass(kw_only=True, eq=False, repr=False)  # literal: the API docs read it
class FOO(GradientClipping, ExpertGuided, ReplayAgent):
    """FOO (Author et al., year): the paper's algorithm box, each line one
    function below."""

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(1e4)

    def init(self, key, ctx):
        rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
        actor_state, critic_state = ctx.actor_critic(
            init_key, continuous=True, action_value=True, num_critics=2
        )
        return FOOState(
            rng=rng, eval_rng=eval_rng, actor_state=actor_state,
            critic_state=critic_state, collector_state=ctx.collector(collector_key),
        )

    def update(self, state, ctx):
        state, sample_key, critic_key, actor_key = state.draw_keys(3)
        batch, _ = sample_replay(state, ctx.buffer, sample_key)
        target = foo_target(state, batch, self.gamma, self.reward_scale)
        critic_state, critic_aux, _, critic_metrics = steps.critic_step(
            state, batch, target, partial(foo_value_loss, state.critic_state, batch),
            ctx.extensions, critic_key, ctx.total_timesteps,
            rewards=batch.reward, dones=batch.terminated,
            gamma=self.gamma, reward_scale=self.reward_scale,
        )
        actor_state, actor_aux, _, actor_metrics = steps.actor_step(
            state, partial(foo_policy_loss, state), ctx.extensions, actor_key,
            ctx.total_timesteps, observations=batch.obs, actions=batch.action,
        )
        state = state.replace(
            actor_state=actor_state,
            critic_state=critic_state.soft_update(tau=self.tau),
            n_updates=state.n_updates + 1,
        )
        metrics = merge_metrics(critic_metrics, actor_metrics)
        return state, AuxiliaryLogs(value=critic_aux, policy=actor_aux, extensions=metrics)

    # The contract, unannotated (an annotation would make it a field).
    name = "FOO"
    capabilities = frozenset("actor discounted off_policy q_critic replay".split())


# FOO's equations, in the order update calls them.
def foo_target(state, batch, gamma, reward_scale): ...       # the paper's target
def foo_value_loss(critic, batch, params, target): ...       # -> (loss, aux)
def foo_policy_loss(state, params): ...                      # -> (loss, aux, pi_mean)


@struct.dataclass(kw_only=True)
class FOOState(BaseAgentState):
    """FOO carries only actor + critic + collector."""
```

A new agent declares no `<AGENT>Config`: its fields are its configuration,
and `agent.agent_config` is a plain `BaseAgentConfig`. Only the agents that
had a Config before their fields were declared (IQL, CrossQ, TD3, PPO, APO, PQN) keep one, a
view of the fields (`declare_config`), until 0.5.0.

`init` splits the run's key on the agent's own layout, which its pins hold.
`steps.critic_step` applies `on_target` and `critic_loss`; its keywords are
what the target used (`dones`: where it cut the bootstrap). `steps.actor_step`
applies `actor_loss`; its keywords are the rows the extensions read. Both derive
the extensions' keys from the step's
([Randomness](docs/developer/writing-an-extension.md#randomness)), so the
agent's own random stream is the same with or without extensions.

An agent writes `init(key, ctx)` and `update(state, ctx)` (an on-policy
agent's `update(state, rollout, start, ctx)`), and may add these hooks; each
is a no-op on the family, which does its own part outside them, so none
calls `super()`:

- `check()` refuses arguments before anything is built;
- `actor_critic_args()` returns `ActorCritic` arguments the paper fixes or
  derives from the fields (initialisations, heads; CrossQ's learning rates
  and widths); it may only fill a field the caller left None (TypeError
  otherwise), as PPO's `encoder_kernel_init` does;
- `optimizer_settings()` returns the `OptimizerConfig` settings of the
  reference's optimiser (its Adam's `eps`, its betas): the family sets none,
  so an agent states its reference's;
- `setup()` derives or checks what needs the built env and networks (IQL's
  value optimiser, PPO's rollout split checks), after the family has built
  the replay;
- `behaviour(ctx)` replaces collection's action pipeline.

The family reads some fields by name. `ReplayAgent` requires `buffer_size`,
`batch_size` and `learning_starts`; an agent claiming the `discounted`
capability declares `gamma` and `reward_scale`, which the evaluation's value
estimate reads (construction refuses it otherwise); `expert_policy` (the
`ExpertGuided` mixin) is the agent's expert; and every field named like an
`ActorCritic.__init__` argument (`actor_learning_rate`,
`critic_architecture`, ...) reaches it, so a misspelt one is a field that
reaches nothing. The shared knobs an algorithm takes beyond those every agent
takes are mixins
(`GradientClipping`, `ObservationNormalization`, `RewardNormalization`,
`ExpertGuided`), and `RecurrentReplayAgent` adds `memory=` with R2D2's
sequence replay. Once built, an agent's fields and `agent_config` are fixed:
`dataclasses.replace(agent, gamma=0.9)` builds another.

**The on-policy family.** `OnPolicyAgent.make_train` runs `loop.on_policy`:
each iteration collects `n_steps` per env, then calls `update(state,
rollout, start, ctx)` on the `(n_steps, n_envs)` rollout and the state it
was collected from (a recurrent agent replays the rollout from its
memories). The family requires `n_steps`, `n_epochs` and
`num_minibatches`, reads brax's `num_resets_per_eval` and `num_evals` where
an agent declares them, and adds the `evaluation_options()` hook (APO's
`avg_reward_mode`). `agents/minibatch.py` runs the passes (`run_epochs`);
PPO's `choose_rollout_split` returns the `RolloutSplit` that says how its
rollout is cut and where the advantages are computed, while APO and PQN
shuffle single steps. `ImageObservations` adds the CNN encoder's fields;
`train_offline` refuses an on-policy agent.

Do **not** add a field for a feature (`use_critic_blend=True`, ...): it
belongs on an extension.

**2. Export and document.** Import it in `src/ajax/__init__.py`, add it to
`__all__`, and add a row to the README's agent table.

**3. Tests and benchmark.** `tests/agents/FOO/`: the maths on hand-computed
values. `tests/agents/test_agent_contract.py`: a row, so FOO builds, refuses
what it cannot act in and folds extensions on tiny runs. `tests/probing/`:
presets in `agents.py`, then the
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
        # A float: the steps that do not evaluate log NaN in its place.
        return {"Counter/updates": ext_state.astype(jnp.float32)}


agent = PPO("CartPole-v1", n_steps=128, extensions=(UpdateCounter(),))
logging = LoggingConfig(config={}, log_frequency=1024, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4096, logging_config=logging)
print(state.ext_state[0])  # the updates each seed made: [9 9]
print(evaluations["Counter/updates"])  # per seed, at each evaluation: [2. 4. 6. 8.]
```

3. **Test it** in `tests/extensions/`: the feature fires and does what its
   paper says (a metric appears, a term moves the loss the right way).

**Callable hooks.** A few `Optional[Callable]` keywords predate extensions and
stay: TD3's `action_pipeline` (AjaxExperiments' TD3 variants), DQN's
`td_target_fn` and `td_loss_fn` and PQN's `td_loss_fn` (Double DQN and
Huber, which their agents' tests run). `tests/modules/test_hook_composition.py`
pins them. Add none.

## Testing

Two kinds of tests check the code, each answering its own question
([Testing](docs/developer/testing.md) is the full guide):

- *Unit tests* (`tests/<area>/`, `tests/agents/<AGENT>/`) check one component
  against an exact contract, with no learning. Where agents or extensions
  share a contract, the test is generic (`tests/agents/test_*_contract.py`,
  `tests/extensions/test_phase_contract.py`, ...): it runs every one of them,
  and a new one fails it until it joins.
- *Probing tests* (`tests/probing`) train an agent briefly on a tiny generic
  environment whose answer is known, and check only what learning shows: the
  right value, the right action, the right count.

A convention that differs from a reference implementation (an ordering, a
default, a constant) is for code review, not for a test. *Pins* record what a
tiny run produces today, so a restructuring can show it changed nothing:
`tests/agents/test_agent_pins.py` (tiny runs' fingerprints),
`tests/agents/test_agent_surface.py` (every public name the agent modules
define, and each agent's run config and `agent_config`),
`tests/extensions/test_sac_extensions_equivalence.py` and
`tests/test_downstream_api.py` (the names downstream projects import). The
fingerprints hold to 1e-3 across platforms; on one machine, `uv run python -m
tests.exact_pins record DIR`, then `compare DIR` after the change, shows a
restructuring bitwise: every leaf of tiny runs' final states, their logged
evaluations and the text of their training programs' jaxprs. *Slow
tests* (`@pytest.mark.slow`) are the long trainings.

The tests come in two tiers. The *fast tier* is the tests not marked slow:
every pull request must pass it. A fast test may run a tiny training, of at
most 128 gradient steps: `tests/conftest.py` counts each test's gradient steps
(the larger of the agent's update count and each train state's step) and
fails a fast test past the line (`AJAX_FAST_MAX_UPDATES`); a test that trains
belongs in `tests/probing`, or is marked slow. It also times each test and
lists those past 30 s (`AJAX_FAST_MAX_SECONDS`) without failing them:
compiles, not trainings, set those times. The *slow tier* is everything that
trains: `tests/probing`, which `tests/conftest.py` marks slow but for the
harness's own checks, and the tests marked slow elsewhere; a workflow of its
own runs it (below).

Every directory of `tests/` that holds Python files is a package (an empty
`__init__.py`; `tests/test_repo_hygiene.py` checks it), so tests import shared
helpers by their full name: `from tests.agents.extension_stubs import
CounterExt`. The benchmark scripts the tests import are on the path through
`pythonpath` in `pyproject.toml`.

```bash
uv run pytest --collect-only -q   # everything imports and collects
uv run pytest tests/agents/TD3    # the tests of what you changed
uv run pytest -m "not slow" -n auto   # the fast tier, a worker per core
make ci       # the pull-request gate: ci-precommit, ci-test, ci-conformance, docs
make ci-all   # the gate and the slow tier (ci-slow)
```

The `make` targets hide the GPU (`CPU_ENV`) so tests never compete with a
running experiment; elsewhere set `JAX_PLATFORMS=cpu` while a GPU is busy.

**CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the fast
tier on CPU for every pull request and push to `main`: *Pre-commit*; *Tests*
(three shards: the tests not marked slow, the probing harness's checks among
them, under coverage, a pytest-xdist worker per vCPU, at XLA's optimisation
level 0 since these programs are tiny; the last shard runs whatever the first
two do not name, so a new test file always runs); *Coverage* (the shards
combined, at least 90%); *Conformance* (each agent's training step against
its original implementation); *Docs* (the documentation site, `mkdocs build
--strict`). *Lint and Test*, the one check branch protection requires, passes
only when all of them passed. They run on GitHub's arm64 runners, which all
come on one CPU model, so a job finds main's compile cache (JAX keys it by
the CPU).

The *Slow tier* workflow ([slow.yml](.github/workflows/slow.yml)) runs
everything that trains (`pytest -m slow`) on x86, the platform most users
train on, in six shards by recorded duration (`--shard I/N`,
`tests/.durations`, `make durations`): daily on main when main changed, by
hand, and on a pull request each time the label `run-slow-tier` is added. A
pull request that changes how an agent trains gets the label before it
merges. A red run on main opens an issue, and bisects the merges since the
last green run when that is cheap.

As the gate runs no probe, every planted fault
(`tests/probing/faults/<family>.jsonl`) names in its `fast` field the fast
tests that catch it, measured by planting the fault in a scratch copy and
running the fast tests that execute its lines (`uv run python -m
tests.fault_catches SCRATCH [FAMILY ...]`); `tests/probing/test_framework.py`
checks that each names live tests outside `tests/probing`. A new fault comes
with its fast catch.

### The probing tests

A unit test checks a contract; it cannot show that an agent learns what it
should. A probe trains an agent on a tiny problem whose answer is known
exactly (a value, a best action, a count) and compares what it learned with
the right answer and named wrong ones (a discount applied twice), so a failure
names the mistake.

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
A planted fault quotes the source it edits, so any change under `src/ajax`
can break one: such a change runs `tests/probing/test_framework.py` locally
before the push (text and collection only, no training).

A new case's budget and tolerances come from calibration on seeds no check
uses (`uv run python -m tests.probing.verdict ladder|choose|certify ...`;
`verdict.py` describes the procedure); then its module's `ANSWER_DIGEST` is
re-pinned. A known bug stays visible as a strict xfail naming the defect,
never as a looser tolerance. The directory is capped at 7,000 lines.

## Performance

CLAUDE.md ("Performance no-regression guardrail") sets the rule; here is the
check. `benchmarks/agent_bench.py` times a fixed run per agent; `--compare`
fails when one is slower than `--tol` (10% by default) against a baseline.
Capturing requires `--out`, so a quick check never overwrites the baseline.

```bash
export JAX_PLATFORMS=cpu
uv run python benchmarks/agent_bench.py --only TD3 --out benchmarks/my_change.jsonl
uv run python benchmarks/agent_bench.py --compare benchmarks/agent_baseline.jsonl benchmarks/my_change.jsonl
```

## Dependencies and releases

Ajax is a library: `pyproject.toml` says which versions work, and `uv.lock`
fixes the ones CI and development use. Users' installers never read the
lock.

- **Floors only.** Each lower bound is the oldest version Ajax works with.
  Raise it when Ajax needs something newer or an older version is known to
  break, with a comment saying why. Add an upper bound only after a known
  break, with its reason, and remove it once Ajax is fixed.
- **The ranges are tested.** The Compatibility workflow
  ([compat.yml](.github/workflows/compat.yml)) runs the tests not marked
  slow every Monday, and on pull requests that change the dependencies,
  twice without the lock: at the newest versions allowed, on the newest
  Python, and with every direct dependency at its floor
  (`uv sync --resolution lowest-direct`), on the oldest.
- **The lock is refreshed monthly** by Dependabot
  ([dependabot.yml](.github/dependabot.yml)), the JAX stack in one pull
  request.
- **Python**: 3.12 and newer, the versions JAX supports. A version goes when
  JAX drops it.
- **Releases.** Versions follow semantic versioning; before 1.0, a breaking
  change (a raised floor, a removed keyword) bumps the minor version. A
  release is an entry in [CHANGELOG.md](CHANGELOG.md) and a `vX.Y.Z` tag on
  main, which downstream projects can pin. Public names change by expand
  then contract (CLAUDE.md, rule 10). Tags come only from the *Release*
  workflow ([release.yml](.github/workflows/release.yml), by hand, with the
  version): it tags main's head once the slow tier has passed there,
  running it first if needed, then publishes that commit to PyPI as
  `ajax-rl`.

## Style

Pre-commit ([.pre-commit-config.yaml](.pre-commit-config.yaml)) runs on every
commit and in CI: ruff lint with rules `I F E W B C RUF ARG` (`ARG`, unused
arguments, is off under `tests/`), ruff-format (settings in
[pyproject.toml](pyproject.toml)), and mypy on everything but `benchmarks/`,
run in the project's environment (`uv run mypy`) so it sees jax and flax.
