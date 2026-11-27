"""The workflows' token permissions, as GitHub checks them before a run.

Contract (GitHub's rule for reusable workflows): a called workflow's token
can only be given fewer permissions than its caller's, never more, so GitHub
refuses to start a run in which a called job asks for more than the calling
job grants (a "startup failure"). A job's permissions are its own block, or
else its workflow's top-level one. The Release workflow, which runs once a
release, failed this way at 0.6.0: slow.yml's slow job had come to ask for
``actions: write`` and release.yml's call granted ``actions: read``.
"""

import pathlib

import pytest
import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
LOCAL = "./.github/workflows/"
LEVELS = {"none": 0, "read": 1, "write": 2}


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def permissions(job: dict, workflow: dict) -> dict:
    """A job's token permissions: its own block, or else its workflow's."""
    return job.get("permissions", workflow.get("permissions", {}))


def calls() -> list:
    """(caller, calling job, called) for each job that calls a workflow here."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in load(path.name)["jobs"].items():
            if job.get("uses", "").startswith(LOCAL):
                found.append((path.name, name, job["uses"].removeprefix(LOCAL)))
    return found


def requests() -> list:
    """Every permission a called job asks for, beside what its call grants:
    (caller, calling job, called, called job, scope, asked, granted)."""
    found = []
    for caller, name, called in calls():
        caller_workflow, called_workflow = load(caller), load(called)
        granted = permissions(caller_workflow["jobs"][name], caller_workflow)
        for called_job, body in called_workflow["jobs"].items():
            for scope, asked in permissions(body, called_workflow).items():
                given = granted.get(scope, "none")
                found.append((caller, name, called, called_job, scope, asked, given))
    return found


@pytest.mark.parametrize("case", requests(), ids=lambda case: "-".join(case[:5]))
def test_a_called_job_asks_for_no_more_than_its_call_grants(case):
    caller, name, called, called_job, scope, asked, given = case
    assert LEVELS[asked] <= LEVELS[given], (
        f"{caller}'s {name} grants {scope}: {given}, but {called}'s {called_job}"
        f" asks for {scope}: {asked}; GitHub would not start the run"
    )
