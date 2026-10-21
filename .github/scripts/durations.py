"""Refresh tests/.durations from junit reports (``make durations``):

    uv run python .github/scripts/durations.py REPORT.xml [REPORT.xml ...]

Each test's time replaces its entry (per node id); tests under one second are
left out (tests/shards.py counts them as one second). Entries the reports do
not name are kept, so the slow and fast tiers' reports can be merged in turn.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "tests", ".durations")


def nodeid(classname: str, name: str) -> str:
    """junit's ``tests.agents.test_x[.TestClass]`` and ``test_y[...]`` back to
    pytest's ``tests/agents/test_x.py[::TestClass]::test_y[...]``."""
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        module = "/".join(parts[:cut]) + ".py"
        if os.path.exists(os.path.join(ROOT, module)):
            return "::".join([module, *parts[cut:], name])
    return "::".join(["/".join(parts) + ".py", name])


def main(reports: list[str]) -> None:
    durations = {}
    if os.path.exists(PATH):
        with open(PATH) as fh:
            durations = json.load(fh)
    for report in reports:
        for case in ET.parse(report).iter("testcase"):
            node = nodeid(case.get("classname", ""), case.get("name", ""))
            seconds = round(float(case.get("time", 0)), 1)
            if seconds >= 1.0:
                durations[node] = seconds
            else:
                durations.pop(node, None)
    with open(PATH, "w") as fh:
        json.dump(dict(sorted(durations.items())), fh, indent=0)
        fh.write("\n")
    print(f"{len(durations)} durations in {PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
