from plane_env.env_jax import Airplane2D, EnvParams

from ajax.agents.PPO.PPO import PPO, LoggingConfig, upload_tensorboard_to_wandb

if __name__ == "__main__":
    n_seeds = 1
    log_frequency = 5000
    use_wandb = False
    logging_config = LoggingConfig(
        project_name="PPO_tests_rlzoo",
        run_name="PPO",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=True,
        use_wandb=use_wandb,
    )

    def process_brax_env_id(env_id: str) -> str:
        """Remove version from env_id for brax compatibility."""
        short_env_id = env_id.split("-")[0].lower()
        brax_envs = [
            "ant",
            "halfcheetah",
            "hopper",
            "walker2d",
            "humanoid",
            "reacher",
            "swimmer",
        ]
        if short_env_id in brax_envs:
            return short_env_id
        return env_id

    env = Airplane2D()
    env_params = EnvParams()
    # wrapped_env = ClipAction(env, low=(0,-1), high=(1,1))
    PPO_agent = PPO(
        normalize_advantage=True,
        env_id=env,
        env_params=env_params,  # **init_hyperparams
        normalize_observations=True,
        normalize_rewards=True,
    )  # Remove version from env_id for brax compatibility
    PPO_agent.train(
        seed=list(range(n_seeds)),
        logging_config=logging_config,
        n_timesteps=int(1e6),
    )
    upload_tensorboard_to_wandb(PPO_agent.run_ids, logging_config)
