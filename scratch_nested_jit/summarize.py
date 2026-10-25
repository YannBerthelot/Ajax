"""Scratch: per-agent, per-mode medians over rounds of the trial medians, and
the min..max over every trial. Usage: summarize.py RESULTDIR..."""

import collections
import glob
import json
import re
import statistics as st
import sys

rows = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*.jsonl")):
        agent, mode, rnd = re.match(
            r".*/(.+)-(off|step|family)-(\d+)\.jsonl", f
        ).groups()
        vals = [json.loads(l)["exec_steps_per_s"] for l in open(f) if l.strip()]
        rows[agent][mode].append((int(rnd), vals))
for agent, modes in rows.items():
    base = [st.median(v) for _, v in sorted(modes["off"])]
    line = (
        f"{agent:5s} off: rounds {[round(x) for x in base]} med {st.median(base):.0f}"
    )
    print(line)
    for mode in ("step", "family"):
        if mode not in modes:
            continue
        m = [st.median(v) for _, v in sorted(modes[mode])]
        allv = [x for _, v in modes[mode] for x in v]
        alloff = [x for _, v in modes["off"] for x in v]
        delta = st.median(m) / st.median(base) - 1
        per_round = [round((a / b - 1) * 100, 1) for a, b in zip(m, base, strict=False)]
        print(
            f"      {mode:6s}: rounds {[round(x) for x in m]} med {st.median(m):.0f} delta {delta:+.1%} per-round {per_round}%  trials {min(allv):.0f}..{max(allv):.0f} (off {min(alloff):.0f}..{max(alloff):.0f})"
        )
