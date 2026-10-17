"""What each Ajax-vs-reference comparison runs: one source for every runner.

Pure data, no Ajax or JAX import: the SB3 runner imports it from another
Python environment.

Each comparison has two Ajax arms against one recorded reference:

* ``matched``: Ajax set to the reference's own conventions. A gap here
  points at Ajax's maths, environment or evaluation. ``unmatched`` lists
  what no Ajax option can match, so a gap may also come from there.
* ``defaults``: Ajax as shipped, with only the task budget and the eval
  schedule set (and PPO's rollout size, which its default makes longer than
  the eval interval). A gap here is a difference of convention, for review.

``solved`` is the return a seed must reach. It was fixed before any run:
CartPole-v1's 475 is gymnasium's reward_threshold; Pendulum-v1 has none,
and -200 is the usual "swung up and held" level.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Comparison:
    agent: str
    env_id: str
    n_timesteps: int
    eval_every: int  # 0: one evaluation, of the final policy
    solved: float
    reference: str  # "sb3" | "purejaxql" | "brax"
    reference_config: dict
    matched: dict
    unmatched: tuple = ()
    defaults: dict = field(default_factory=dict)
    # Reference setting -> the Ajax keyword(s) holding the same value, where
    # the names differ; tests check every pair the matched arm copies.
    renames: dict = field(default_factory=dict)
    # Reference setting -> the factor its Ajax value carries for the same
    # behaviour, where the two losses differ by a constant.
    scale: dict = field(default_factory=dict)


# Every evaluation: 10 deterministic (greedy) episodes per seed.
N_EVAL_EPISODES = 10

COMPARISONS = {
    "PPO": Comparison(
        renames={"learning_rate": ("actor_learning_rate", "critic_learning_rate")},
        # Ajax's value loss is vf_coef * 0.5 * MSE, SB3's vf_coef * MSE.
        scale={"vf_coef": 2.0},
        agent="PPO",
        env_id="CartPole-v1",
        n_timesteps=98 * 2048,  # 98 rollouts of 8 envs x 256 steps
        eval_every=7 * 2048,
        solved=475.0,
        reference="sb3",
        reference_config={
            "n_envs": 8,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation": "tanh",
        },
        # SB3's Adam eps (1e-5) and orthogonal sqrt(2) hidden layers too.
        matched={
            "n_envs": 8,
            "n_steps": 256,
            "num_minibatches": 8,
            "n_epochs": 10,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "actor_architecture": ("64", "tanh", "64", "tanh"),
            "critic_architecture": ("64", "tanh", "64", "tanh"),
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "adam_eps": 1e-5,
            "encoder_kernel_init": "orthogonal(1.4142135623730951)",
        },
        unmatched=(
            "gradient clipping: SB3 clips actor and critic together, Ajax each",
            "output-layer initialisation gains (SB3: 0.01 policy, 1 value)",
        ),
        defaults={"n_envs": 8, "n_steps": 256},
    ),
    "DQN": Comparison(
        renames={
            "gradient_steps": "n_gradient_steps",
            "exploration_initial_eps": "epsilon_start",
            "exploration_final_eps": "epsilon_end",
            "exploration_fraction": "epsilon_decay_frac",
        },
        agent="DQN",
        env_id="CartPole-v1",
        n_timesteps=100_000,
        eval_every=5_000,
        solved=475.0,
        reference="sb3",
        # One gradient step per env step, the only cadence Ajax's DQN has
        # (SB3's train_freq=1, gradient_steps=1); targets are copied every
        # 500 steps, which SB3 counts in env steps and Ajax in gradient
        # steps: the same here after warm-up.
        reference_config={
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 500,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "max_grad_norm": 10.0,
            "net_arch": [128, 128],
        },
        matched={
            "n_envs": 1,
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 0.99,
            "n_gradient_steps": 1,
            "target_update_interval": 500,
            "tau": 1.0,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_frac": 0.2,
            "max_grad_norm": 10.0,
            "architecture": ("128", "relu", "128", "relu"),
            "td_loss": "huber",  # SB3's smooth_l1_loss; Ajax's default is squared
        },
        unmatched=(
            "weight initialisation (torch default vs Ajax's)",
            "Adam eps: 1e-5 in Ajax, 1e-8 in the reference",
        ),
    ),
    "SAC": Comparison(
        renames={
            "learning_rate": (
                "actor_learning_rate",
                "critic_learning_rate",
                "alpha_learning_rate",
            )
        },
        agent="SAC",
        env_id="Pendulum-v1",
        n_timesteps=20_000,
        eval_every=1_000,
        solved=-200.0,
        reference="sb3",
        # RL Zoo's Pendulum-v1 SAC: SB3 defaults with a 1e-3 learning rate.
        reference_config={
            "learning_rate": 1e-3,
            "buffer_size": 20_000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "net_arch": [256, 256],
        },
        matched={
            "n_envs": 1,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "alpha_learning_rate": 1e-3,
            "buffer_size": 20_000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "alpha_init": 1.0,
            "target_entropy_per_dim": -1.0,
            "policy_update_start": 0,
            "alpha_update_start": 0,
            "actor_architecture": ("256", "relu", "256", "relu"),
            "critic_architecture": ("256", "relu", "256", "relu"),
        },
        unmatched=(
            "weight initialisation",
            "initial policy std: exp(-1) in Ajax, about 1 in SB3",
            "Adam eps: 1e-5 in Ajax, 1e-8 in the reference",
        ),
    ),
    "TD3": Comparison(
        renames={
            "learning_rate": ("actor_learning_rate", "critic_learning_rate"),
            "action_noise_std": "exploration_noise",
        },
        agent="TD3",
        env_id="Pendulum-v1",
        n_timesteps=20_000,
        eval_every=1_000,
        solved=-200.0,
        reference="sb3",
        # SB3 defaults with RL Zoo's Pendulum-v1 learning rate and Gaussian
        # behaviour noise of 0.1 (in [-1, 1] units, as Ajax's).
        reference_config={
            "learning_rate": 1e-3,
            "buffer_size": 20_000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "action_noise_std": 0.1,
            "net_arch": [400, 300],
        },
        matched={
            "n_envs": 1,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "buffer_size": 20_000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "exploration_noise": 0.1,
            "actor_architecture": ("400", "relu", "300", "relu"),
            "critic_architecture": ("400", "relu", "300", "relu"),
        },
        unmatched=(
            "weight initialisation",
            "Adam eps: 1e-5 in Ajax, 1e-8 in the reference",
        ),
    ),
    "PQN": Comparison(
        renames={
            "NUM_ENVS": "n_envs",
            "NUM_STEPS": "n_steps",
            "NUM_MINIBATCHES": "num_minibatches",
            "NUM_EPOCHS": "n_epochs",
            "LR": "learning_rate",
            "GAMMA": "gamma",
            "LAMBDA": "q_lambda",
            "EPS_START": "epsilon_start",
            "EPS_FINISH": "epsilon_end",
            "EPS_DECAY": "epsilon_decay_frac",
            "REW_SCALE": "reward_scale",
            "MAX_GRAD_NORM": "max_grad_norm",
        },
        agent="PQN",
        env_id="CartPole-v1",
        # 240 updates of 32 x 64 steps, tested every 12 (TEST_INTERVAL 0.05):
        # the paper config's 5e5 steps would make 244, whose last 4 are never
        # tested, so both sides run 491,520 and decay over that.
        n_timesteps=240 * 2048,
        eval_every=12 * 2048,
        solved=475.0,
        reference="purejaxql",
        # purejaxql/config/alg/pqn_cartpole.yaml at 47af6d7, but the length.
        reference_config={
            "TOTAL_TIMESTEPS": 240 * 2048,
            "TOTAL_TIMESTEPS_DECAY": 240 * 2048,
            "NUM_ENVS": 32,
            "NUM_STEPS": 64,
            "EPS_START": 1.0,
            "EPS_FINISH": 0.2,
            "EPS_DECAY": 0.2,
            "NUM_MINIBATCHES": 16,
            "NUM_EPOCHS": 4,
            "NORM_INPUT": False,
            "HIDDEN_SIZE": 256,
            "NUM_LAYERS": 2,
            "NORM_TYPE": "layer_norm",
            "LR": 1e-4,
            "MAX_GRAD_NORM": 10,
            "LR_LINEAR_DECAY": True,
            "REW_SCALE": 0.1,
            "GAMMA": 0.99,
            "LAMBDA": 0.95,
            "ENV_NAME": "CartPole-v1",
            "ENV_KWARGS": {},
            "TEST_DURING_TRAINING": True,
            "TEST_INTERVAL": 0.05,
            "TEST_NUM_ENVS": 128,
            "EPS_TEST": 0.0,
            "WANDB_MODE": "disabled",
        },
        matched={
            "n_envs": 32,
            "n_steps": 64,
            "num_minibatches": 16,
            "n_epochs": 4,
            "learning_rate": 1e-4,
            "architecture": ("256", "relu", "256", "relu"),
            "gamma": 0.99,
            "q_lambda": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.2,
            "epsilon_decay_frac": 0.2,
            "reward_scale": 0.1,
            "max_grad_norm": 10.0,
            # purejaxql decays the rate linearly to ~0 over every gradient
            # step (240 updates x 16 minibatches x 4 epochs), and halves the
            # squared TD error, which moves where the clip binds.
            "lr_decay": ("linear", 1e-20, 240 * 16 * 4),
            "td_loss": "half_squared",
        },
        unmatched=(
            "optimiser: RAdam (eps 1e-8) in purejaxql, Adam (eps 1e-5) in Ajax",
            "weight initialisation (lecun_normal vs Ajax's)",
            "time limits: the reference treats them as terminations",
            "evaluation: 128 greedy envs over one 500-step window, vs 10 episodes",
        ),
    ),
    "APG": Comparison(
        renames={"num_envs": "n_envs", "horizon_length": "horizon"},
        agent="APG",
        env_id="Pendulum-v1",
        n_timesteps=1_000 * 64 * 32,  # 1,000 updates of 64 envs x 32 steps
        eval_every=0,
        solved=-200.0,
        reference="brax",
        # Brax's APG trainer on gymnax Pendulum (refcmp_jax.GymnaxAsBrax),
        # its defaults apart from the run's size.
        reference_config={
            "episode_length": 200,
            "policy_updates": 1_000,
            "horizon_length": 32,
            "num_envs": 64,
            "learning_rate": 1e-4,
        },
        matched={
            "n_envs": 64,
            "horizon": 32,
            "learning_rate": 1e-4,
            "beta_1": 0.7,
            "beta_2": 0.95,
            "max_grad_norm": None,
            "lr_decay": ("exponential", 0.997),  # Brax's use_schedule default
            "actor_architecture": (
                "32",
                "relu",
                "32",
                "relu",
                "32",
                "relu",
                "32",
                "relu",
            ),
        },
        unmatched=(
            "episodes: Brax continues them across updates from scrambled start"
            " times and resets each env to its own first state (64 starts);"
            " Ajax restarts them from fresh states at every update",
            "Brax trains on sampled actions (tanh-Gaussian, var_scale 0.1);"
            " Ajax's controller is deterministic",
            "Brax clips each gradient entry to [-1, 1]; its loss is the mean"
            " reward over the horizon, Ajax's the summed return",
            "network: ELU with LayerNorm and orthogonal(0.01) in Brax, ReLU in Ajax",
            "Adam eps: 1e-5 in Ajax, 1e-8 in the reference",
        ),
    ),
}
