"""The SBX references (PPO, SAC): Stable-Baselines3's interface with JAX
networks and updates.

usage: python refcmp_sbx.py NAME reference FIRST_SEED N_SEEDS (refcmp_worker)

Imports nothing from Ajax. SBX keeps SB3's rollout and replay buffers,
callbacks and evaluation, so it runs through refcmp_sb3's runner on the same
gymnax environment, stepped by the same adapter.
"""

if __name__ == "__main__":
    import sbx
    from refcmp_sb3 import config, run
    from refcmp_worker import main

    main(
        lambda spec, _, seeds: run(spec, seeds, sbx),
        lambda spec, _: config(spec, sbx),
        (
            "sbx-rl",
            "stable-baselines3",
            "torch",
            "jax",
            "jaxlib",
            "flax",
            "optax",
            "gymnax",
        ),
    )
