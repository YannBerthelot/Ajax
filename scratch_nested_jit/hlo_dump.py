"""Scratch: the optimised HLO of an agent's whole train program (the vmapped,
jitted function agent.train runs), plus opcode statistics.
Usage: NESTED_JIT=off|step|family python hlo_dump.py AGENT OUTPREFIX"""

import collections
import json
import os
import re
import sys

sys.path.insert(0, "benchmarks")
import agent_bench as ab
import jax
import jax.numpy as jnp

name, prefix = sys.argv[1], sys.argv[2]
env_id, kw = ab.AGENTS[name]
agent = ab._build(name, env_id, kw, ab.N_ENVS)
train_jit = agent.get_make_train()(
    env_args=agent.env_args,
    actor_optimizer_args=agent.actor_optimizer_args,
    critic_optimizer_args=agent.critic_optimizer_args,
    network_args=agent.network_args,
    agent_config=agent.agent_config,
    total_timesteps=ab.TIMESTEPS,
    num_episode_test=10,
    run_ids=[],
    logging_config=None,
)
keys = jax.vmap(jax.random.PRNGKey)(jnp.array([1]))
index = jnp.arange(1)
lowered = jax.jit(jax.vmap(train_jit)).lower(keys, index)
open(prefix + ".stablehlo.txt", "w").write(lowered.as_text())
compiled = lowered.compile()
text = compiled.as_text()
open(prefix + ".opt.txt", "w").write(text)

OP = re.compile(r"= .*? ([a-z][a-z0-9\-]*)\(")
ops, per_comp, comp = collections.Counter(), collections.Counter(), None
for line in text.splitlines():
    if line and not line.startswith(" ") and line.rstrip().endswith("{"):
        comp = line.split()[1] if line.startswith("ENTRY") else line.split()[0]
        continue
    m = OP.search(line) if line.startswith("  ") else None
    if m:
        ops[m.group(1)] += 1
        per_comp[comp] += 1
stats = {
    "agent": name,
    "mode": os.environ.get("NESTED_JIT", "off"),
    "computations": len(per_comp),
    "instructions": sum(ops.values()),
    "ops": {
        k: ops[k]
        for k in (
            "fusion",
            "call",
            "while",
            "conditional",
            "copy",
            "dot",
            "custom-call",
            "all-reduce",
        )
        if ops[k]
    },
    "top_ops": ops.most_common(12),
}
try:
    cost = compiled.cost_analysis()
    stats["flops"] = cost.get("flops") if isinstance(cost, dict) else None
    stats["bytes_accessed"] = (
        cost.get("bytes accessed") if isinstance(cost, dict) else None
    )
except Exception as exc:  # cost analysis is optional
    stats["cost_error"] = str(exc)[:100]
json.dump(stats, open(prefix + ".stats.json", "w"), indent=1)
print(json.dumps(stats))
