# One measurement per family (not a benchmark): with no persistent compile
# cache, after a warm-up trace, the compiled 0-step skeleton vs its eval_shape.
import sys, time
import jax
from ajax import SAC, DreamerV3, TDMPC2
from ajax.checkpoint import skeleton_of

SEEDS = [0, 1]
net = ("8", "relu")
TINY_TD = {"enc_dim": 32, "mlp_dim": 32, "latent_dim": 16, "num_q": 2, "batch_size": 16,
           "num_samples": 32, "num_elites": 8, "num_pi_trajs": 4, "iterations": 2}
TINY_DV = {"model_size": "1m", "units": 16, "hidden": 16, "deter": 32, "classes": 4, "stoch": 4,
           "blocks": 4, "imag_horizon": 3, "batch_size": 4, "batch_length": 8, "warmup": 10}
builds = {
    "SAC": lambda: SAC(env_id="Pendulum-v1", n_envs=2, actor_architecture=net, critic_architecture=net,
                       learning_starts=10, batch_size=4, buffer_size=40),
    "DreamerV3": lambda: DreamerV3("CartPole-v1", n_envs=2, replay_capacity=200, **TINY_DV),
    "TDMPC2": lambda: TDMPC2("Pendulum-v1", n_envs=2, seed_steps=400, **TINY_TD),
}
name = sys.argv[1]
skeleton_of(builds[name](), SEEDS)  # warm-up: imports, module-level traces
t = time.perf_counter(); jax.block_until_ready(builds[name]().train(seed=SEEDS, n_timesteps=0)); tc = time.perf_counter() - t
t = time.perf_counter(); skeleton_of(builds[name](), SEEDS); ta = time.perf_counter() - t
print(f"{name}: compiled skeleton {tc:.2f} s, eval_shape skeleton {ta:.2f} s")
