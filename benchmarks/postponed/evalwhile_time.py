"""evaluate() vmapped over 8 keys (as training vmaps it over seeds):
CartPole with an untrained actor (episodes end early) and Pendulum (every
episode lasts 200 steps). Median of 20 calls after 3 warm-ups."""
import statistics, time
import jax, jax.numpy as jnp
from ajax.evaluate import evaluate
from tests.test_evaluate import _cartpole_eval_kwargs, _eval_pendulum

def timeit(f):
    for _ in range(3): jax.block_until_ready(f())
    ts = []
    for _ in range(20):
        t = time.perf_counter(); jax.block_until_ready(f()); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

keys = jax.random.split(jax.random.PRNGKey(0), 8)
kw = _cartpole_eval_kwargs(); kw.pop("rng")
cp = jax.jit(jax.vmap(lambda k: evaluate(rng=k, **kw)))
env, env_params, state = _eval_pendulum()
pe = jax.jit(jax.vmap(lambda k: evaluate(env=env, actor_state=state.actor_state, num_episodes=5, rng=k, env_params=env_params, recurrent=False)))
print(f"cartpole_vmap8={timeit(lambda: cp(keys))*1e3:.1f}ms pendulum_vmap8={timeit(lambda: pe(keys))*1e3:.1f}ms")
