"""Trace + lower time of a tiny TD-MPC2 train (bench preset), with or
without the row collector's random_phase (argv[1] in {with, without})."""
import sys, time
from functools import partial

import jax
import jax.numpy as jnp

variant = sys.argv[1]
episodic = len(sys.argv) > 2 and sys.argv[2] == "episodic"
import ajax
from ajax.agents.TDMPC2 import train_TDMPC2 as tt

preset = dict(model_size=1, enc_dim=64, mlp_dim=64, latent_dim=32, batch_size=32,
              num_samples=64, num_elites=8, num_pi_trajs=4, iterations=2)
if episodic:
    preset["episodic"] = True
if variant == "without":
    tt.collect_row = partial(tt.collect_row, random_phase=None)
    _orig = tt.collect_row.func
    def no_rp(*a, **k):
        k["random_phase"] = None
        return _orig(*a, **k)
    tt.collect_row = no_rp

es_time = {"t": 0.0, "n": 0}
_orig_es = jax.eval_shape
def timed_es(fun, *a, **k):
    t = time.perf_counter()
    out = _orig_es(fun, *a, **k)
    if getattr(fun, "__name__", "") == "act_with_policy":
        es_time["t"] += time.perf_counter() - t
        es_time["n"] += 1
    return out
jax.eval_shape = timed_es

agent = ajax.TDMPC2(env_id="Pendulum-v1", n_envs=4, **preset)
n_timesteps = 8000
sched = agent.schedule
end_tick = sched.num_ticks(n_timesteps, 0)
per_env = end_tick if isinstance(sched, tt.EpisodicSchedule) else sched.steps_before(end_tick)
buffer = agent._replay_buffer(per_env * agent.env_args.n_envs)
train_jit = agent.get_make_train()(
    env_args=agent.env_args,
    actor_optimizer_args=agent.actor_optimizer_args,
    critic_optimizer_args=agent.critic_optimizer_args,
    network_args=agent.network_args,
    agent_config=agent.agent_config,
    total_timesteps=n_timesteps,
    num_episode_test=10,
    run_ids=[],
    logging_config=None,
    buffer=buffer,
    start_tick=0,
)
keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(1))
index = jnp.arange(1)
f = jax.jit(jax.vmap(train_jit))
t0 = time.perf_counter()
traced = f.trace(keys, index)
t1 = time.perf_counter()
lowered = traced.lower()
t2 = time.perf_counter()
print(f"{variant} episodic={episodic}: trace {t1-t0:.2f}s lower {t2-t1:.2f}s total {t2-t0:.2f}s;"
      f" eval_shape(act_with_policy) {es_time['t']:.2f}s x{es_time['n']};"
      f" jaxpr eqns {len(traced.jaxpr.eqns)}; hlo chars {len(lowered.as_text())}")
