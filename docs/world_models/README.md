# World-model agents: DreamerV3 and TD-MPC2

Reference material for Ajax's implementations of

- **DreamerV3**: Hafner, Pasukonis, Ba, Lillicrap, *Mastering Diverse Domains through
  World Models*, arXiv:2301.04104v2 (2024), published in Nature (2025).
  Official code: <https://github.com/danijar/dreamerv3>.
- **TD-MPC2**: Hansen, Su, Wang, *TD-MPC2: Scalable, Robust World Models for Continuous
  Control*, ICLR 2024, arXiv:2310.16828v2.
  Official code: <https://github.com/nicklashansen/tdmpc2>.

| File | Contents |
|---|---|
| `DESIGN.md` | Architecture of the Ajax implementation: units, shared blocks, collector, replay, schedules, testing strategy, milestones. |
| `deviations.md` | Which code version Ajax follows where the paper-era and latest official code differ, and every place Ajax departs from the reference. |
| `dreamerv3_spec.md` | Component-by-component specification of DreamerV3 with exact values, paper and code citations, paper-vs-code conflicts and version notes. |
| `tdmpc2_spec.md` | The same for TD-MPC2, single-task and multi-task. |
| `shared_blocks.md` | The building blocks both algorithms use (symlog, two-hot, percentile normalisers, normed MLPs, ...), with one parameterisation covering both papers and the unit tests that pin them. |
| `VALIDATION.md` | The GPU validation against the published curves (M9): reference data, paper-protocol runner, report and acceptance criteria, multi-task pipeline, GPU commands and resource estimates. |
| `parity/` | Fixture generators: scripts that run the pinned reference code itself at tiny sizes (in a throwaway venv, see each script's docstring) and write the `.npz` parity fixtures committed under `tests/agents/<Agent>/fixtures/` (`DESIGN.md` §10). |
| `reference_comparison/` | DreamerV3 against the real reference code (29eb964) on gymnax CartPole-v1, three pre-registered rounds: the harness (run by hand, the reference side in a throwaway venv), protocol, pre-registrations, root-cause investigation and compact results, with a script that recomputes every table (`reference_comparison/README.md`; conclusions in `PERFORMANCE_REPORT.md`). |

## How the specifications were produced

Each component was extracted from the paper and the pinned official code by one reader,
then re-checked line by line against the same sources by an independent adversarial
verifier whose corrections were all applied (about 50 across both papers). Paper
citations are to the PDF pages of the arXiv versions above; code citations are
`path:line` at the pinned commits:

- DreamerV3 `e3f02248693a79dc8b0ebd62c93683888ddaccfe` (latest checked) with version
  notes for the paper-era commit `2411f7d` and the bug fix `29eb964`;
- TD-MPC2 `e9f59321933cbc8e11a002b842adc7d4ffae8ff1` (latest checked) with version notes
  for the paper-era commits `b67b21c` and `5f6fade`.

Where the specs list a value as the latest code's and the paper-era code differs, the
"Version note" column gives the paper-era value; `deviations.md` §1–2 says which one
Ajax implements (the fidelity rule in `DESIGN.md` §0).
