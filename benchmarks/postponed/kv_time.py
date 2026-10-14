"""Execution time per APG contextual-controller update (input cache vs K/V
cache): wall time at 2 and 22 updates, no compile cache; the difference
divided by 20 removes compile and setup."""
import sys, time
import jax, jax.numpy as jnp
from gymnax import make as make_gymnax_env
from ajax.agents.APG.APG import APG
from ajax.environments.model_reference import LinearReferenceModel, ModelReferenceWrapper, StepReference
from ajax.environments.system_class import UniformPerturbation

def build(seeds):
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(g=0.0)
    ref = StepReference(horizon=100, min_value=-0.5, max_value=0.5, min_duration=4, max_duration=8)
    task = ModelReferenceWrapper(env, ref, LinearReferenceModel.first_order(), output_fn=lambda o: jnp.arctan2(o[1], o[0]))
    sc = UniformPerturbation(params, fields=("m", "l"), scale=0.1)
    return APG.contextual_controller(task, sc, n_envs=4, horizon=100, env_params=sc.nominal)  # paper-size defaults: 8 layers, d_model 128, context 100

seeds = list(range(int(sys.argv[1]))) if len(sys.argv) > 1 else 0
def run(updates):
    agent = build(seeds)
    t = time.perf_counter()
    out = agent.train(seed=seeds, n_timesteps=4 * 100 * updates)
    jax.block_until_ready(out)
    return time.perf_counter() - t
t1, t2 = run(2), run(22)
print(f"seeds={sys.argv[1] if len(sys.argv)>1 else 1} total_2={t1:.2f}s total_22={t2:.2f}s exec_per_update={(t2-t1)/20:.3f}s")
