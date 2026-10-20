"""The Compile preview workflow's table: one row per run, from the files
.github/scripts/compile-preview.sh leaves under stats/ (xdist workers'
statistics summed). Prints Markdown; the workflow appends it to the job
summary."""

import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "stats"
rows = []
for meta_path in sorted(
    glob.glob(os.path.join(root, "**", "*.meta.json"), recursive=True)
):
    meta = json.load(open(meta_path))
    stats = {
        "hits": 0,
        "misses": 0,
        "requests": 0,
        "compile_s": 0.0,
        "retrieval_s": 0.0,
    }
    for path in glob.glob(meta_path.replace(".meta.json", ".json*")):
        for key, value in json.load(open(path)).items():
            if key in stats:
                stats[key] += value
    rows.append({**meta, **stats})

print(
    "| run | CPU | cache key | wall s | exit | ISA warnings | hits | misses | compile requests | compile s | cache read s |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(
        f"| {r['run']} | {r['cpu']} | {r.get('key', '')} | {r['wall_s']} | {r['status']} | {r['isa_warnings']} |"
        f" {r['hits']} | {r['misses']} | {r['requests']} | {r['compile_s']:.0f} | {r['retrieval_s']:.0f} |"
    )
