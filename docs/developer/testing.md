# Testing

Ajax tests at three levels. Unit tests show that each piece computes what it
should. Pins show that a restructuring changed nothing. Probing tests show
that an agent *learns the right thing*. That last level is what makes
reinforcement-learning code trustworthy: a broken agent often still solves
CartPole, so a smoke test that trains and checks the return proves little.

## The tiers

| Tier | Where | What it shows | CI job |
| --- | --- | --- | --- |
| Unit | `tests/<area>/`, `tests/agents/<AGENT>/` | a function, a loss, a network or a wrapper on hand-computed values; every agent and extension runs | Tests |
| Pins | `tests/agents/test_agent_pins.py`, `tests/extensions/test_sac_extensions_equivalence.py`, `tests/test_downstream_api.py` | a restructuring reproduces a tiny run's fingerprint; the names downstream projects import still exist | Tests |
| Slow | anywhere, marked `@pytest.mark.slow` | long trainings: an agent reaches a return, a wrapper holds over many episodes | Slow tests |
| Probing | `tests/probing/` | an agent's values and actions on tiny problems with exact answers | Probing |
| Learning checks | `benchmarks/learning_checks.py` | a world model learns its task at a small preset; too long for CI, run before merging such an agent | none |

Within an agent's folder the unit tests are split by kind, as
`tests/agents/conftest.py` describes: `test_<agent>_code.py` for programmatic
invariants (a configuration value reaches its consumer),
`test_<agent>_structure.py` for shapes and parameter trees, and
`test_<agent>_rl.py` for the algorithm on synthetic data (GAE on a known
trajectory, loss signs, bootstrap conventions). The plain `test_<agent>.py`
smoke tests are the weakest layer. That split exists because of a real bug:
PPO once ran one full-batch update where it should have run 32 minibatch
updates, and its CartPole smoke test still passed.

The pins are change detectors, not truth. A fingerprint is the sum of squares
of each parameter tree after a tiny run, compared with a relative tolerance
that absorbs the floating-point drift between CI's Linux x86 and macOS ARM.
A behaviour-neutral change that moves a pin only by that drift re-records it;
a change that moves the numbers on purpose re-records it in its own commit,
saying why.

## Running tests locally

The full suite costs about an hour of CPU, so CI runs it on the pull request.
Locally, run the cheap checks first:

```bash
uv run pre-commit run --all-files   # ruff lint, ruff-format, mypy
uv run pytest --collect-only -q     # everything imports and collects
uv run pytest tests/agents/TD3      # the tests of what you changed
uv run pytest -m "not slow"         # skip the long trainings
```

`make ci` runs everything CI runs (`ci-precommit`, `ci-test`, `ci-slow`,
`ci-probe`, and `docs`, the strict build of this site). The `make` targets hide the GPU (`CUDA_VISIBLE_DEVICES=""`,
`JAX_PLATFORMS=cpu`), so tests never compete with an experiment running on
it. Outside `make`, set `JAX_PLATFORMS=cpu` while a GPU is busy.

`pyproject.toml` registers one marker, `slow`, and runs pytest with
`--strict-markers`, so a misspelt marker is an error rather than a test that
silently runs in the wrong job.

## How CI splits the suite

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on CPU for
every pull request and every push to `main`:

| Job | What it runs |
| --- | --- |
| Pre-commit | ruff lint, ruff-format and mypy on every file |
| Tests (3 shards) | the tests not marked slow, probing excluded, under coverage. Shards 1 and 2 name directories; shard 3 runs everything else, so a new test file always runs somewhere |
| Coverage | the three shards' data combined; fails below 70% |
| Slow tests | `-m slow`, probing excluded, without coverage (its tracing would lengthen them) |
| Probing (5 shards) | `tests/probing`, slow cases included. Shards 1-3 split the memory probes by memory kind; the last shard takes any probe file the others do not name |
| Docs | `mkdocs build --strict` with the `docs` group alone: a broken link, a page left out of the navigation or a docstring naming a parameter the signature lacks fails it |
| Lint and Test | passes only when every job above passed; the one check branch protection requires |

A new push to a pull request cancels the run it supersedes. Nothing merges
while any job is red; a failure that is pre-existing and unrelated to the
change is called out explicitly instead of merged over.

## The probing framework

A unit test can check that a TD target uses `max` rather than `mean`. It
cannot check that the trained agent's values come out right, which depends on
the target, the buffer, the episode-end handling, the optimiser and the
network all working together. A probe trains an agent on a tiny Markov
decision process whose answer is known exactly (a value, a best action, a
count) and compares what the agent learned with the right answer and with
the answers that named bugs would produce. When it fails, it says which
mistake the agent's reading is closest to.

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
budget and the calibrated tolerances. This one, shortened from
`test_value_chains.py`, is a two-step chain: state `s1` pays 0, 0.5 or 1 by
action, and `Q(s0) = 0.8 max_a Q(s1, a)`:

```python
# doc: skip (an excerpt of tests/probing/test_value_chains.py; it needs the test package)
Q_S0_WRONG = {"mean bootstrap (D01)": 0.8 * 0.5, "no discount": 1.0}

CASES["p0a-DQN"] = Case(
    "p0a-DQN",
    (*Q_S1, *(Query(f"Q(s0, a={a})", 0.8, Q_S0_WRONG) for a in range(3)), GREEDY),
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

1. **Ladder.** Read two blocks of 32 seeds (1000-1031 and 2000-2031) untrained
   and at a series of budgets.
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

CI does not plant faults; it runs `test_every_planted_fault_applies_once_and_names_live_nodes`,
which checks that each `old` text still occurs once and that each listed node
still exists. When a refactor moves the code a fault edits, update its line
and run it again. The catalogue holds about 180 faults across seven modules.

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
| The discount is squared, replaced by `1 - gamma`, or dropped | the value-chain cases: at the probe discount 0.62 those wrong answers sit at least 0.24 from the right one |
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
   changed a preset). The failing pin test prints the new value.
6. **Plant the faults.** Add a line per bug to `faults/<module>.jsonl`, run
   them with the fault runner, and keep only lines it reports as caught. A
   fault that survives means the probe is weaker than you thought.
7. **Mind the caps.** If a file outgrows its cap, shorten the code before
   raising the cap.

A new agent needs probe presets in `agents.py` and at least one case; a new
probe file runs in CI's last probing shard without any workflow change.
