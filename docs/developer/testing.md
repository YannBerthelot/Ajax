# Testing

Ajax checks its code with two kinds of tests, and each answers a different
question:

- **Unit tests** check one component against an exact contract, with no
  learning. Where agents or extensions share a contract, the test is
  generic: it runs every one of them, and a new agent or extension fails it
  until it joins.
- **Probes** are tiny generic environments with known answers. An agent
  trains on one briefly, and the probe checks a capability of the whole
  agent: that it learns the right value, picks the right action, counts the
  right steps.

A convention that differs from a reference implementation (an ordering, a
default, a constant) is a matter for code review only, not for a test.

The two kinds do not overlap. A unit test can show that a TD target takes a
`max` rather than a `mean`; it cannot show that the trained agent's values
come out right, which needs the target, the buffer, the episode-end handling,
the optimiser and the network to work together. A smoke test that trains on
CartPole and checks the return proves little either: a broken agent often
still solves CartPole.

Beside them, *pins* record what a tiny run produces today, so that a
restructuring can show it changed nothing.

## Where the tests live

| Kind | Where | CI job |
| --- | --- | --- |
| Unit, one component | `tests/<area>/` (environments, networks, buffers, logging, ...), `tests/agents/<AGENT>/` | Tests |
| Unit, generic contracts | `tests/agents/test_*_contract.py`, `tests/extensions/test_phase_contract.py`, `tests/extensions/test_no_silent_noop.py`, `tests/environments/test_wrapper_contract.py`, ... (below) | Tests, Slow tests |
| Pins | `tests/agents/test_agent_pins.py`, `tests/agents/SAC/test_sac_parity.py`, `tests/extensions/test_sac_extensions_equivalence.py`, `tests/test_downstream_api.py` | Tests |
| Slow | anywhere, marked `@pytest.mark.slow`: the option matrix, the resume contract, long trainings | Slow tests |
| Probes | `tests/probing/` | Probing |
| Learning checks | `benchmarks/world_models/learning_checks.py`: a world model learns its task at a small preset; hours on a CPU, run before merging such an agent | none |

Every directory of `tests/` that holds Python files is a package, so tests
import shared helpers by their full name (`from tests.environments.identity_env
import IdentityEnv`); `tests/test_repo_hygiene.py` checks it.

## The generic contract tests

These tests are written once and run over every agent, extension or wrapper
that shares the contract. Each finds its subjects from the code (the
exported agents, the subclasses of `Extension`, the signatures), so a new
one cannot slip past: it fails a completeness check until it has a row.
Several run on the *identity task* (`tests/environments/identity_env.py`),
whose observation `[t, c]` says which step of which episode a row comes from:
an episode with `c > 0` terminates at `t = 3`, the others are truncated at
`t = 5`.

| Test | The contract |
| --- | --- |
| `tests/agents/test_agent_contract.py` | every agent on the shared loop builds its configs on each action space it acts in and refuses the other; a default run keeps no extension state; a run of two seeds and two envs folds its extensions in step and adds their metrics to every evaluation; the measurement extensions find its critic by themselves; the run config drops the instance; the evaluation key is apart from the training key, and each evaluation draws fresh keys; a continuous agent acts on the env's whole box; `episode_length` and `action_repeat` reach the env |
| `tests/extensions/test_phase_contract.py` | for every agent and every phase it folds (`on_target`, `critic_loss`, `actor_loss`, `post_update`): each batch holds the keys `PHASE_BATCH_KEYS` documents; each fold gets a key of its own, never one the agent draws on; `critic_loss` gets the target `on_target` returned; a loss term moves the network it names, and that one only |
| `tests/extensions/test_no_silent_noop.py` | every extension implementing `on_target`, `critic_loss` or `actor_loss` raises on a batch without what it reads, rather than return its input or 0 |
| `tests/extensions/test_extension_phase_support.py` | each agent accepts exactly the phases it folds; a reader of phi\* needs `MCPretrain` |
| `tests/environments/test_wrapper_contract.py` | every Ajax env wrapper survives copy, deepcopy and pickle, and reports the episode-end flags each step raised, alone and in the stacks `create.py` builds |
| `tests/agents/test_transition_contract.py` | the transitions every agent hands `on_target` continue their own episode (a time-limited row's next observation is the final one, not the next episode's first), carry the flags the step raised, and carry the normalised rewards under reward normalisation |
| `tests/agents/test_expert_consumers.py` | every agent taking an `expert_policy` gives it the raw observation, and a stateful expert its state; the cloning dataset holds its actions in agent units; the evaluation's expert return is its closed form |
| `tests/test_evaluate.py`, `tests/extensions/test_instrumentation_extensions.py` | every output of the evaluation, and every metric of the measurement extensions, meets its closed form on a known rollout |
| `tests/agents/test_reward_normalisation.py` | every agent taking `normalize_rewards` hands the normaliser the discount it documents (none by default, `reward_normalization_gamma` when given) and reports its Train and Eval returns in the env's units (UDRL's Train return is a strict xfail: a known bug) |
| `tests/agents/test_carry_restart.py` | every recurrent agent's actor remembers only its current episode, for every memory kind: the carry it acts with is the actor's run from a fresh carry over the episode's observations (one row per agent and per kind here, the rest in the slow tier) |
| `tests/agents/test_option_smoke.py` (slow) | the option matrix: every pair of an agent's options, and every extension stacked into an agent that folds it, trains a tiny logged run that updates the policy and keeps every parameter finite |
| `tests/agents/test_resume_contract.py` (slow) | a run split in two, its second half on a new agent instance, ends in the state one uninterrupted call reaches, every leaf alike (floats to a relative 1e-5): every agent of the contract table and APG, with the normalisation it takes and with memory where it takes one (the world models' own tests hold theirs) |

The option matrix and the resume contract train, so their main tests run in
the slow tier; their completeness checks (every switch of a signature is in
a row, every extension has a builder, every agent has a row) are fast and run
with the other unit tests.

What each agent computes is tested in its own directory, `tests/agents/<AGENT>/`:
GAE on a known trajectory, a loss's sign, a target on a hand-built batch,
and that every configuration value reaches the code that runs. That last
kind exists because of a real bug: PPO once treated `num_minibatches=32` as
1, and its CartPole smoke test still passed
(`tests/agents/PPO/test_PPO_code.py`).

## Pins

The pins are change detectors, not truth. `test_agent_pins.py` compares a
fingerprint of a tiny run (the sum of squares of each parameter tree, and
the state of a test extension) with the one recorded, under a relative
tolerance that absorbs the floating-point drift between CI's Linux x86 and
macOS ARM. A behaviour-neutral change that moves a pin only by that drift
re-records it; a change that moves the numbers on purpose re-records it in
its own commit, saying why. `test_downstream_api.py` pins the names and
keywords the downstream projects import, so that removing one is a
deliberate step (CLAUDE.md, rule 10).

## Running tests locally

The full suite costs about an hour of CPU, so CI runs it on the pull request.
Before pushing, run the cheap checks
([CLAUDE.md](../../CLAUDE.md),
"Commit requirement"):

```bash
uv run pre-commit run --all-files            # ruff lint, ruff-format, mypy
uv run pytest --collect-only -q              # everything imports and collects
uv run pytest tests/agents/TD3               # the tests of what you changed
uv run pytest tests/probing/test_framework.py  # any change under src/ajax
uv run pytest tests/probing -k TD3           # a change to TD3's training: all its probe cells
```

Two of these are process rules. A change under `src/ajax` runs
`tests/probing/test_framework.py`, which trains nothing: every planted fault
quotes a line of the source, so any source edit can break one. A change to an
agent's training runs all of that agent's probe cells (`-k <Agent>` over
`tests/probing`).

`make ci` runs everything CI runs (`ci-precommit`, `ci-test`, `ci-slow`,
`ci-probe`, and `docs`, the strict build of this site). The `make` targets
hide the GPU (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`), so tests never
compete with an experiment running on it. Outside `make`, set
`JAX_PLATFORMS=cpu` while a GPU is busy.

`pyproject.toml` registers one marker, `slow`, and runs pytest with
`--strict-markers`, so a misspelt marker is an error rather than a test that
silently runs in the wrong job.

## How CI splits the suite

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
runs on CPU for every pull request and every push to `main`:

| Job | What it runs |
| --- | --- |
| Pre-commit | ruff lint, ruff-format and mypy on every file |
| Tests (3 shards) | the tests not marked slow, probing excluded, under coverage. Shards 1 and 2 name directories; shard 3 runs everything else, so a new test file always runs somewhere |
| Coverage | the three shards' data combined; fails below 70% |
| Slow tests | `-m slow`, probing excluded, without coverage (its tracing would lengthen them) |
| Probing (5 shards) | `tests/probing`, slow cases included. Shards 1-3 split the memory probes by memory kind; the last shard takes any probe file the others do not name |
| Docs | `mkdocs build --strict` with the `docs` group alone: a broken link, a page left out of the navigation or a docstring naming a parameter the signature lacks fails it |
| Lint and Test | passes only when every job above passed; the one check branch protection requires |

The test, slow and probing jobs restore Ajax's compile cache as the last push
to `main` left it for that job, so a program a change leaves alone is loaded
instead of compiled. A new push to a pull request cancels the run it
supersedes. Nothing merges while any job is red; a failure that is
pre-existing and unrelated to the change is called out explicitly instead of
merged over.

## The probing framework

A probe trains an agent on a tiny Markov decision process whose answer is
known exactly (a value, a best action, a count) and compares what the agent
learned with the right answer and with the answers that named bugs would
produce. When it fails, it says which mistake the agent's reading is closest
to.

The directory [`tests/probing`](../../tests/probing) holds:

| File | Role |
| --- | --- |
| `envs.py` | `Spec`: a small episodic environment from its transition function, with gymnax auto-reset, a time limit and, on request, a record of the executed actions |
| `agents.py` | each agent's frozen probe presets, pinned by `DIGEST`, and `make`, which checks every override reached the agent |
| `runs.py` | trains every seed of a case in one compiled program, and caches the readings so the tests of one case share its training |
| `readouts.py` | reads values and actions off a trained agent, through the same paths training uses |
| `oracles.py` | the right answers, and the wrong ones under named faults, from each estimator's fixed point or a closed form |
| `verdict.py` | `Query`, `Case`, the two-stage verdict, calibration and answer digests |
| `faults.py`, `faults/*.jsonl` | the planted-fault catalogue and its runner |
| `test_*.py` | the cases, grouped by subject (below); `test_framework.py` tests the harness itself |

| Module | What it probes |
| --- | --- |
| `test_value_chains.py` | values and actions on short chains: max bootstrap and action indexing, discounting, entropy bonuses, normalisation, gradients through a transition (APG) |
| `test_episode_ends.py` | terminations against time limits, rollout ends, parallel environments out of step, reward scale, average reward |
| `test_bookkeeping.py` | what an agent counts, logs, stores, draws and executes: step counters, UDRL's commands, an expert's warm-up, a declared action box |
| `test_extensions.py` | every extension phase an agent claims to fold, through test-only extensions of known effect; the expert extensions' numbers |
| `test_memory.py` | a recurrent agent acts on what an episode showed it and restarts at the episode's end |
| `test_world_models.py` | answers reachable only through a learned model (TD-MPC2's horizon, multi-task twins, DreamerV3's imagination) |
| `test_p4_split_run.py` | a run checkpointed, restored and resumed equals the run never interrupted |

### Known-answer cases

A `Query` names one reading, its right answer and the wrong answers that
named faults produce. A `Case` adds the readings function, the training
budget and the calibrated tolerances. This one, from `test_value_chains.py`,
is a two-step chain: state `s1` pays 0, 0.5 or 1 by action, and
`Q(s0) = 0.8 max_a Q(s1, a)`:

```python
# doc: skip (an excerpt of tests/probing/test_value_chains.py; it needs the test package)
R3, G3 = (0.0, 0.5, 1.0), 0.8  # s1's rewards by action; the discount
Q_S0_WRONG = {"mean bootstrap (D01)": G3 * 0.5, "no discount": 1.0}

CASES["p0a-DQN"] = Case(
    "p0a-DQN",
    (*Q_S1, *(Query(f"Q(s0, a={a})", G3, Q_S0_WRONG) for a in range(3)), GREEDY),
    _three_run("DQN"),  # train, then read Q at both states and the greedy action
    2500,  # budget, in environment steps
    (0.026, 0.021, 0.033, 0.037, 0.032, 0.029, 0.02),  # one tolerance per query
    note="1250 qualified but 2 seeds of 3000-3031 read Q(s0) 0.047 off; cert 32/32",
)
```

The tolerance can never reach a wrong answer: it is at most half the gap to
the nearest one, and at most 0.1. So a reading within tolerance is closer to
the truth than to any named bug. A margin query asks only for the right sign
by a margin, for readings such as the gap between two actions' values.

Some cases are exact rather than judged. The bookkeeping and split-run probes
assert equalities (a counter, a logged timestep, a whole state) on two seeds
trained in one program.

### The two-stage seed rule

Learning is noisy, so a judged case trains several seeds and counts how many
land within tolerance:

1. Train seeds 0-7 in one program. Seven or more within tolerance: pass.
   Four or fewer: fail.
2. Five or six: train seeds 8-15 as well, and pass when 13 of the 16 are
   within.

A clear result costs one training; only an ambiguous one pays for a second.
A precondition (a reading of the wrong shape, an override that did not
reach the agent) raises `RuntimeError`, never `AssertionError`, so a strict
xfail cannot absorb it.

### Calibration and certification

A case's budget and tolerances are measured, never guessed, and on seeds no
check uses:

1. **Ladder.** Read two blocks of 32 seeds (1000-1031 and 2000-2031)
   untrained and at a series of budgets.
2. **Choose.** Pool the blocks and take the smallest budget at which twice
   the worst seed's error fits under the cap (the smaller of 0.1 and half
   the gap to the nearest wrong answer); that doubled error, at least 0.02,
   becomes the tolerance. For a margin query, half the weakest seed's margin
   must reach 0.25.
3. **Certify.** Check the chosen budget and tolerances on a third block,
   3000-3031, and record the result in the case's `note`.

```bash
uv run python -m tests.probing.verdict ladder test_value_chains p0a-DQN 1000 1250,2500,5000 a.json
uv run python -m tests.probing.verdict ladder test_value_chains p0a-DQN 2000 1250,2500,5000 b.json
uv run python -m tests.probing.verdict choose test_value_chains p0a-DQN a.json b.json
uv run python -m tests.probing.verdict certify test_value_chains p0a-DQN   # seeds 3000-3031
```

`test_framework.py` checks that every calibrated tolerance lies inside its
gap.

### Answer digests

A calibration holds only for the configuration and the answers it was made
on. Two digests pin both:

- `agents.DIGEST` hashes every agent's probe presets.
- Each scenario module's `ANSWER_DIGEST` hashes every query's right answer,
  wrong answers, tolerance and kind, to six decimals.

`test_presets_and_answers_are_pinned` fails when either moves. Change a
preset or an answer only together with a fresh calibration, then re-pin the
digest. To see what a module's digest covers:

```bash
uv run python -m tests.probing.verdict answers test_value_chains
```

### Planted faults

A probe that has never caught a bug has not shown it can. Each scenario
module has a catalogue, `faults/<module>.jsonl`, with one line per fault:

```json
{"id": "P0a-D01-DQN", "path": "agents/DQN/train_DQN.py",
 "old": "    max_q_next = jnp.max(q_next, axis=-1, keepdims=True)\n",
 "new": "    max_q_next = jnp.mean(q_next, axis=-1, keepdims=True)\n",
 "catches": ["test_value_chains.py::test_learns[p0a-DQN]"]}
```

`path` is under `src/ajax`; `old` must occur exactly once; `catches` lists
every test node that fails with the fault planted. The runner copies the
repository to a scratch directory, plants the fault there (your checkout is
never modified), checks that Python imports the copy, and runs the listed
nodes with the compile cache off. A non-zero exit means caught:

```bash
JAX_PLATFORMS=cpu uv run python -m tests.probing.faults /tmp/faults P0a-D01-DQN P0a-gather-up
# {'P0a-D01-DQN': 'caught', 'P0a-gather-up': 'caught'}
```

CI does not plant faults; `test_framework.py` runs
`test_every_planted_fault_applies_once_and_names_live_nodes`, which checks
that each `old` text still occurs once and that each listed node still
exists. This is why a change under `src/ajax` runs `test_framework.py`
before the push: when a change moves the code a fault edits, update the
fault's line and run it again. The catalogue holds 188 faults across seven
modules.

### Known defects

A probe that finds a live bug keeps failing, visibly. The case gets a
`defect`, which turns it into a strict xfail whose reason names the defect
(`LIVE defect: ...`). When the bug is fixed, the xfail passes, strict mode
fails the suite, and the annotation has to be removed. A known bug is never
hidden by loosening a tolerance.

### Size budgets

The framework is meant to stay readable. `test_framework.py` caps each file's
length (`CAPS`) and the whole directory at 7,000 lines, and requires at least
one case for every agent `ajax` exports.

## The bugs the probes target

The catalogue reads as a list of the classic ways reinforcement-learning code
goes wrong without crashing:

| Bug | Example faults |
| --- | --- |
| The bootstrap averages instead of maximising; the taken action's value, or the greedy action, is read at the wrong index | `P0a-D01-DQN`, `P0a-gather-up`, `P0a-greedy-up` |
| The discount is squared, replaced by `1 - gamma`, or dropped | the value-chain cases: at the probe discount 0.62 those wrong answers sit about 0.24 or more from the right one |
| A time limit is treated as a termination; the reset observation replaces the episode's final one; truncation is dropped | `P6-PPO-OR`, `P6-RESET-OBS`, `P6-GYMNAX-TRUNC-DROPPED` |
| The reward scale is applied twice | `P5-S13`, `P5-REDQ-scaled-twice` |
| GAE runs forwards in time; the value bootstraps on the wrong observation; the configured lambda is ignored | `Q9-GAE-FORWARD`, `P09`, `Q9-PQN-LAMBDA` |
| The entropy term has the wrong sign or is missing; the tanh log-probability misses its Jacobian, in the policy or in the target | `P07`, `S-entropy-sign`, `S-no-jacobian`, `S-pre-tanh-target` |
| The observation normaliser is skipped in the target or at evaluation | `P1-sac-target-skips-agent-norm`, `P1-eval-skips-agent-norm` |
| TD3's target policy noise is not clipped | `T02` |
| Parallel environments share one random key; the step counter counts collection steps, not environment steps | `B02`, `P9-shared-step-keys`, `B01` |
| Every seed gets the same key | `B11` |
| A recurrent memory is never reset at an episode start, or reset every step; the burn-in is too short | `P0b-never-restart`, `P0b-reset-every-step`, `Q2-short-burn` |
| A resumed run loses its random key, optimiser state, counter or buffer | `P4-rng`, `P4-optimizer`, `P4-counter`, `P4-buffer` |
| An extension's term has no gradient, its target is ignored, or `post_update` runs before the update | `P3-ppo-critic-term-no-gradient`, `P3-dqn-target-ignored`, `P3-post-update-first` |
| Logs go to the wrong run; the last evaluation is dropped | `W01`, `L01` |
| A planner forgets its terminal value; a world model ignores the action | `Q12-terminal`, `Q13-action-cut` |

## Adding a probe

1. **Pick the module** by subject, and write the environment as an
   `envs.Spec`: a transition function returning the next state, the reward
   and the termination flag.
2. **Write the queries.** Give each reading its exact right answer and the
   wrong answers the bugs you are after would produce. Choose the problem's
   numbers so the wrong answers sit well apart from the right one: the
   tolerance can be at most half that gap.
3. **Write the readings** with `runs.readings(build, read)`: `build` makes
   the agent with `agents.make` (its probe preset plus your overrides), and
   `read` applies `readouts` to the trained state.
4. **Calibrate and certify** as above, and record the budget, the
   tolerances and the certification in the case's `note`.
5. **Re-pin** the module's `ANSWER_DIGEST` (and `agents.DIGEST` if you
   changed a preset). The failing pin test shows the new value.
6. **Plant the faults.** Add a line per bug to `faults/<module>.jsonl`, run
   them with the fault runner, and keep only lines it reports as caught. A
   fault that survives means the probe is weaker than you thought.
7. **Mind the caps.** If a file outgrows its cap, shorten the code before
   raising the cap.

A new agent needs probe presets in `agents.py` and at least one case; a new
probe file runs in CI's last probing shard without any workflow change.
