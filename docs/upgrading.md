# Upgrading

How to move a project from one Ajax release to the next. The notes for each
release are in the [changelog](changelog.md); this page says where to look
and what to check.

## Which version you have

```bash
python -c "import ajax; print(ajax.__version__)"
pip install --upgrade ajax-rl
```

The documentation has one version per release, chosen at the top of the
page. `latest` is the newest release and `dev` is built from main, with
changes not released yet.

## What a version number tells you

Versions follow semantic versioning. Before 1.0, a breaking change bumps
the minor version (0.5.0 to 0.6.0) and a fix bumps the patch (0.5.0 to
0.5.1). Breaking means one of:

- a public name removed or renamed;
- a default changed, or a result that a fixed seed gave before;
- a dependency moved into an extra, or its oldest allowed version raised;
- a task or environment definition changed.

A project that needs stable results pins the minor version, for example
`ajax-rl>=0.5,<0.6`.

## The notes of each release

Each release's changelog entry has a section listing what can break code or
results that worked on the release before, with what to do about each:

| From | To | Notes |
| --- | --- | --- |
| 0.5.0 | the next release (0.6.0), on main | [Upgrading from 0.5.0](changelog.md#upgrading-from-050) |
| 0.4.0 | 0.5.0 | [Upgrading from 0.4.0](changelog.md#upgrading-from-040) |
| 0.3.1 | 0.4.0 | [0.4.0](changelog.md#040): a rework, read as a new library |

Where a default changed, the entry ends with an "Upgrade:" line giving the
arguments that restore the earlier behaviour.

## What to check after upgrading

- **Results.** A release that changes a default or the random numbers a seed
  draws moves results. Compare runs of the same version, and re-run a
  baseline after a minor upgrade.
- **Checkpoints.** A [checkpoint](guide/checkpoints.md) holds arrays, and
  loads into an agent built with the same arguments. When a release changes
  a default that shapes the networks or the replay, a file saved before
  loads only into an agent given the earlier values. The release's notes
  name those cases.
- **Deprecations.** A deprecated argument still works and warns. Run your
  tests with warnings shown (`python -W default`) to see them before the
  argument goes.
