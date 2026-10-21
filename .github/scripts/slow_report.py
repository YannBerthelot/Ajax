"""The slow tier's report (.github/workflows/slow.yml): on main, a red run
opens an issue labelled ``slow-tier-red`` (or comments on the open one) and a
green run closes it; it also decides whether a bisect is cheap enough.

    python3 .github/scripts/slow_report.py JUNIT_DIR --result RESULT --sha SHA
        --last-green SHA --run-url URL [--dry-run]

Writes ``issue``, ``bisect`` and ``nodes`` to $GITHUB_OUTPUT. ``--dry-run``
prints the gh commands instead of running them.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from durations import nodeid

LABEL = "slow-tier-red"
#: A bisect is cheap: at most this many failing tests, taking at most this
#: long together, over 2 to 16 first-parent commits (at most 4 steps).
MAX_NODES, MAX_SECONDS, MAX_COMMITS = 5, 600.0, 16


def failures(junit_dir: str) -> dict[str, float]:
    """The failing tests' node ids and seconds."""
    out = {}
    for path in glob.glob(os.path.join(junit_dir, "**", "*.xml"), recursive=True):
        for case in ET.parse(path).iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                node = nodeid(case.get("classname", ""), case.get("name", ""))
                out[node] = float(case.get("time", 0))
    return out


def bisectable(failed: dict[str, float], commits: int) -> bool:
    small = 0 < len(failed) <= MAX_NODES and sum(failed.values()) <= MAX_SECONDS
    return small and 2 <= commits <= MAX_COMMITS


def body(failed: dict[str, float], args: argparse.Namespace) -> str:
    lines = [f"The slow tier is red at {args.sha} ([run]({args.run_url})).", ""]
    if args.last_green:
        lines.append(
            f"Last green run: {args.last_green}; range {args.last_green}..{args.sha}."
        )
    if failed:
        lines += ["", "Failing tests:", ""]
        lines += [
            f"- `{node}` ({seconds:.0f} s)" for node, seconds in sorted(failed.items())
        ]
    else:
        lines += [
            "",
            "No test failed in the reports: the run failed outside the tests.",
        ]
    return "\n".join(lines)


def gh(args: argparse.Namespace, *cmd: str) -> str:
    if args.dry_run:
        print("gh", *cmd)
        return ""
    return subprocess.run(
        ["gh", *cmd], check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("junit_dir")
    for flag in ("--result", "--sha", "--last-green", "--run-url"):
        p.add_argument(flag, default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    failed = failures(args.junit_dir)
    open_issue = gh(args, "issue", "list", "--label", LABEL, "--state", "open",
                    "--json", "number", "--jq", ".[0].number").strip()  # fmt: skip
    output = {"issue": open_issue, "bisect": "false", "nodes": " ".join(sorted(failed))}
    if args.result == "success":
        if open_issue:
            gh(args, "issue", "close", open_issue, "--comment",
               f"Green again at {args.sha} ([run]({args.run_url})).")  # fmt: skip
    else:
        if open_issue:
            gh(args, "issue", "comment", open_issue, "--body", body(failed, args))
        else:
            gh(args, "label", "create", LABEL, "--force", "--color", "B60205")
            url = gh(args, "issue", "create", "--label", LABEL, "--title",
                     f"Slow tier red at {args.sha[:7]}", "--body", body(failed, args))  # fmt: skip
            output["issue"] = url.strip().rsplit("/", 1)[-1]
        commits = 0
        if args.last_green:
            count = ["git", "rev-list", "--first-parent", "--count"]
            run = subprocess.run(
                [*count, f"{args.last_green}..{args.sha}"],
                capture_output=True,
                text=True,
            )
            commits = int(run.stdout.strip() or 0)
        output["bisect"] = str(bisectable(failed, commits)).lower()
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as fh:
        fh.writelines(f"{key}={value}\n" for key, value in output.items())
    print(output)


if __name__ == "__main__":
    main(sys.argv[1:])
