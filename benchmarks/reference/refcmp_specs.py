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
and -200 is the usual "swung up and held" level; 950 is Gymnasium's
InvertedPendulum threshold.

Both sides train on the same JAX environment, ``env_id`` (``environment``
says how it differs from the task the reference was written for), and
``envs/<reference>/`` pins the reference's libraries. Every
``eval_every`` steps (once, at the end, when 0) the deterministic (greedy)
policy plays ``n_eval_episodes`` episodes; a seed's score is the mean return
of its last evaluation. Where a reference evaluates otherwise, ``unmatched``
says how. ``adastop`` is AdaStop's ``(n, K, alpha)``: n new seeds per side
and interim, at most K interims, the family-wise error alpha.

``margin`` is how far apart the two sides' mean scores may be and still
count as similar: 5% of the score's range, or, for a score without a fixed
range, ``relative_margin`` times the reference's mean score. A comparison
without either gets no verdict.

A spec is written before its runs and not edited once results exist: a
change becomes a new ``version``. Every record keeps the version and the
hash (:func:`spec_sha256`) of the spec it ran, and the tests hold the
recorded results of the current version to the current spec.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Deviation:
    """A difference between Ajax's matched arm and its original, found and pinned by a
    conformance check (tests/conformance): ``kind`` is "ajax" (an Ajax choice the matched arm
    cannot switch off), "convention" (both sides defensible) or "reference defect" (a bug in
    the original, which the check asserts rather than reproduces); ``submap`` is the sub-step
    where it acts, ``where`` the code that makes it, ``check`` the test that pins it."""

    name: str
    kind: str
    submap: str
    description: str
    where: str
    check: str


@dataclass(frozen=True)
class Comparison:
    agent: str
    env_id: str
    n_timesteps: int
    eval_every: int  # 0: one evaluation, of the final policy
    solved: float | None
    reference: str  # "sb3" | "purejaxql" | "brax": envs/<reference> pins it
    reference_config: dict
    matched: dict
    source: str  # the reference's code
    environment: str  # the JAX environment, against the original task
    unmatched: tuple = ()
    defaults: dict = field(default_factory=dict)
    # Reference setting -> the Ajax keyword(s) holding the same value, where
    # the names differ; tests check every pair the matched arm copies.
    renames: dict = field(default_factory=dict)
    # Reference setting -> the factor its Ajax value carries for the same
    # behaviour, where the two losses differ by a constant.
    scale: dict = field(default_factory=dict)
    n_eval_episodes: int = 10
    adastop: tuple = (4, 24, 0.05)
    margin: float | None = None
    relative_margin: float | None = None
    version: int = 1
    # What the conformance checks found: documentation of the two codes, not a setting of the
    # runs, so it is left out of the spec's hash and adding one does not stale any record.
    deviations: tuple = ()


def spec_sha256(spec: Comparison) -> str:
    settings = {k: v for k, v in asdict(spec).items() if k != "deviations"}
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


CARTPOLE = (
    "gymnax CartPole-v1, a port of Gymnasium's CartPole-v1 (dynamics,"
    " termination, 500-step limit); SB3 steps it through"
    " refcmp_sb3.gymnax_vec_env"
)
PENDULUM = (
    "gymnax Pendulum-v1, a port of Gymnasium's Pendulum-v1 (200-step"
    " episodes); SB3 steps it through refcmp_sb3.gymnax_vec_env"
)

# purejaxql's PQN against Ajax's: its settings' Ajax names, and what no Ajax
# option matches.
PUREJAXQL_RENAMES = {
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
}
PUREJAXQL_UNMATCHED = (
    "optimiser: RAdam (eps 1e-8) in purejaxql, Adam (eps 1e-5) in Ajax",
    "weight initialisation (lecun_normal vs Ajax's)",
    "time limits: the reference treats them as terminations",
)

_PQN_CHECKS = "tests/conformance/test_pqn_purejaxql.py::"
PUREJAXQL_DEVIATIONS = (
    Deviation(
        name="per-step epsilon",
        kind="ajax",
        submap="collection, epsilon-greedy actions",
        description="Ajax reads epsilon at the collector's timestep, so it moves at every"
        " env step; purejaxql reads it once per update. While epsilon decays, Ajax explores"
        " slightly less within a rollout.",
        where="src/ajax/agents/DQN/train_DQN.py linear_epsilon;"
        " vendor/purejaxql/pqn_gymnax.py:189",
        check=_PQN_CHECKS + "test_exploration_differs_only_by_the_per_step_epsilon",
    ),
    Deviation(
        name="time limits",
        kind="convention",
        submap="Q(lambda) targets",
        description="purejaxql treats a time-limit cut as a termination (the target is the"
        " reward); Ajax bootstraps it. Only the targets of episodes cut by a time limit"
        " differ.",
        where="refcmp_jax._FiveValueStep; src/ajax/agents/PQN/utils.py"
        " compute_q_lambda_targets",
        check=_PQN_CHECKS + "test_time_limit_cuts_change_only_their_episodes_targets",
    ),
    Deviation(
        name="Q(lambda) off-by-one",
        kind="reference defect",
        submap="Q(lambda) targets",
        description="purejaxql's backward pass starts with the last step's bootstrap,"
        " V(next_obs) of the last transition, as the next value of the second-to-last one,"
        " whose next value is V(obs) of the last; that target, and through lambda those"
        " before it in its episode, differ from Q(lambda). Ajax computes Q(lambda)."
        " Reported upstream as purejaxql issue #12.",
        where="vendor/purejaxql/pqn_gymnax.py:246-254",
        check=_PQN_CHECKS + "test_targets_differ_only_by_purejaxqls_off_by_one",
    ),
)

_REDQ_CHECKS = "tests/conformance/test_redq_original.py::"
#: watchernyu/REDQ (7b5d1bf) against Ajax's REDQ, from the conformance check (no outcome comparison).
REDQ_DEVIATIONS = (
    Deviation(
        name="first step's temperature",
        kind="reference defect",
        submap="targets and actor loss of the first training step",
        description="REDQ's __init__ derives its temperature from log_alpha (1.0), then sets"
        " self.alpha = alpha, the unused default 0.2: its first training step's targets and"
        " actor loss read 0.2, the later ones exp(log_alpha). Ajax reads exp(log_alpha)"
        " throughout.",
        where="vendor/watchernyu_redq/redq/algos/redq_sac.py:61,80",
        check=_REDQ_CHECKS + "test_redqs_first_step_reads_its_overwritten_temperature",
    ),
    Deviation(
        name="tanh log-det",
        kind="convention",
        submap="log-probabilities: targets, actor loss, temperature",
        description="REDQ subtracts log(1 - tanh(u)^2 + 1e-6) from the Gaussian's"
        " log-probability; Ajax the exact log-det of tanh. They part as actions saturate; the"
        " check runs Ajax with REDQ's form.",
        where="vendor/watchernyu_redq/redq/algos/core.py:232;"
        " src/ajax/networks/distributions.py Squash",
        check=_REDQ_CHECKS + "test_one_step_matches_redqs",
    ),
)

COMPARISONS = {
    "PPO": Comparison(
        # Version 1 (the pilot, at c4aeac6) set fused_grad_clip, which clipped
        # both networks at 1.0 on top of each one's own clip at 0.5, so never
        # bound: each network was clipped on its own. Version 2 runs PPO's
        # one clip over both networks at 0.5, as SB3 clips its optimiser.
        version=2,
        renames={"learning_rate": ("actor_learning_rate", "critic_learning_rate")},
        # Ajax's value loss is vf_coef * 0.5 * MSE, SB3's vf_coef * MSE.
        scale={"vf_coef": 2.0},
        agent="PPO",
        env_id="CartPole-v1",
        n_timesteps=98 * 2048,  # 98 rollouts of 8 envs x 256 steps
        eval_every=7 * 2048,
        solved=475.0,
        margin=25.0,  # 5% of 0-500
        reference="sb3",
        source="stable-baselines3's PPO, MlpPolicy",
        environment=CARTPOLE,
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
        # SB3's Adam eps (1e-5) and orthogonal sqrt(2) hidden layers too; its
        # output layers (0.01 policy, 1 value) are Ajax's PPO defaults.
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
        defaults={"n_envs": 8, "n_steps": 256},
    ),
    "PPO-inverted_pendulum": Comparison(
        # Version 2 writes out Adam's eps, 1e-8 as optax's default in Brax,
        # since Ajax's PPO default became 1e-5; version 1 ran it implicitly.
        version=2,
        renames={
            "num_envs": "n_envs",
            "num_updates_per_batch": "n_epochs",
            "learning_rate": ("actor_learning_rate", "critic_learning_rate"),
            "entropy_cost": "ent_coef",
            "discounting": "gamma",
            "clipping_epsilon": "clip_range",
            "vf_loss_coefficient": "vf_coef",
        },
        agent="PPO",
        env_id="inverted_pendulum",
        n_timesteps=5 * 163_840,  # 5 updates of 2048 envs x 80 steps
        eval_every=163_840,
        solved=950.0,
        margin=50.0,  # 5% of 0-1000
        reference="brax",
        source="brax.training.agents.ppo, with Brax's settings for"
        " inverted_pendulum (notebooks/training.ipynb) but the length",
        environment="Brax's inverted_pendulum (generalized pipeline, 1000-step"
        " episodes), a MuJoCo InvertedPendulum simulated by Brax: the same Env"
        " on both sides",
        n_eval_episodes=128,  # Brax's num_eval_envs
        # Brax's defaults written out where the matched arm copies them.
        reference_config={
            "num_envs": 2048,
            "batch_size": 1024,
            "num_minibatches": 32,
            "unroll_length": 5,
            "num_updates_per_batch": 4,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "discounting": 0.97,
            "gae_lambda": 0.95,
            "clipping_epsilon": 0.3,
            "vf_loss_coefficient": 0.5,
            "max_grad_norm": None,
            "normalize_advantage": True,
            "normalize_observations": True,
            "reward_scaling": 10,
            "episode_length": 1000,
            "action_repeat": 1,
        },
        # Brax's value loss is vf_loss_coefficient * 0.5 * MSE, as Ajax's.
        matched={
            "n_envs": 2048,
            # Each env runs batch_size * num_minibatches / num_envs = 16
            # unrolls of 5 steps per update; GAE runs within each unroll.
            "n_steps": 80,
            "unroll_length": 5,
            "num_minibatches": 32,
            "n_epochs": 4,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "ent_coef": 1e-2,
            "gamma": 0.97,
            "gae_lambda": 0.95,
            "clip_range": 0.3,
            "vf_coef": 0.5,
            "adam_eps": 1e-8,
            "max_grad_norm": None,
            "normalize_advantage": True,
            "normalize_observations": True,
            "episode_length": 1000,
            "action_repeat": 1,
            "actor_architecture": ("32", "swish") * 4,
            "critic_architecture": ("256", "swish") * 5,
            "encoder_kernel_init": "lecun_uniform",
            "mean_kernel_init": "lecun_uniform",
            "squash": True,  # Brax's tanh-Gaussian
            # A tanh-Gaussian's std comes out of Brax's policy network, beside
            # the mean (networks.make_policy_network), and starts near
            # softplus(0) + 0.001.
            "log_std_state_independent": False,
            "log_std_init": -0.3651,
        },
        unmatched=(
            "rewards: Brax scales them by 10 (reward_scaling); Ajax's PPO cannot",
            "time limits: Brax leaves a truncated step out of the advantage"
            " (bootstrap_on_timeout off: this env marks every fall as a time"
            " out, so it would bootstrap falls); Ajax bootstraps it",
            "auto-reset: Brax restarts each env from its own first state;"
            " Ajax draws a new one",
            "std, computed from the observation on both sides: softplus(x) +"
            " 0.001 in Brax, exp(x) clipped to [-20, 2] in Ajax; its weights"
            " start lecun_uniform in Brax, zero in Ajax",
            "the value head's initialisation (lecun_uniform in Brax)",
        ),
        defaults={"n_envs": 2048, "n_steps": 80, "episode_length": 1000},
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
        source="stable-baselines3's DQN, MlpPolicy",
        environment=CARTPOLE,
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
        source="stable-baselines3's SAC, MlpPolicy",
        environment=PENDULUM,
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
        source="stable-baselines3's TD3, MlpPolicy",
        environment=PENDULUM,
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
        renames=PUREJAXQL_RENAMES,
        agent="PQN",
        env_id="CartPole-v1",
        # 240 updates of 32 x 64 steps, tested every 12 (TEST_INTERVAL 0.05):
        # the paper config's 5e5 steps would make 244, whose last 4 are never
        # tested, so both sides run 491,520 and decay over that.
        n_timesteps=240 * 2048,
        eval_every=12 * 2048,
        solved=475.0,
        margin=25.0,  # 5% of 0-500
        reference="purejaxql",
        source="purejaxql 47af6d7: vendor/purejaxql/pqn_gymnax.py",
        environment="gymnax CartPole-v1, a port of Gymnasium's CartPole-v1",
        n_eval_episodes=128,  # purejaxql tests 128 envs
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
            *PUREJAXQL_UNMATCHED,
            "evaluation: purejaxql scores the last episode each of its 128 test"
            " envs ends within one 500-step window, Ajax one episode per env",
        ),
        deviations=PUREJAXQL_DEVIATIONS,
    ),
    "PQN-Asterix": Comparison(
        renames=PUREJAXQL_RENAMES,
        agent="PQN",
        env_id="Asterix-MinAtar",
        # 300 updates of 128 x 32 steps, tested every 15 (TEST_INTERVAL 0.05).
        n_timesteps=300 * 4096,
        eval_every=15 * 4096,
        solved=None,  # no threshold: scored by its return only
        relative_margin=0.05,
        reference="purejaxql",
        source="purejaxql 47af6d7: vendor/purejaxql/pqn_gymnax.py with"
        " config/alg/pqn_minatar.yaml but the length. That config is for"
        " pqn_minatar.py, whose CNN Ajax's CNN encoder cannot build (VALID"
        " padding, LayerNorm after the convolution); both sides use"
        " pqn_gymnax.py's MLP on the flattened image instead",
        environment="gymnax Asterix-MinAtar, a port of MinAtar's Asterix with"
        " a 1000-step limit and no sticky actions",
        n_eval_episodes=128,
        reference_config={
            "TOTAL_TIMESTEPS": 300 * 4096,
            "TOTAL_TIMESTEPS_DECAY": 300 * 4096,
            "NUM_ENVS": 128,
            "NUM_STEPS": 32,
            "EPS_START": 1.0,
            "EPS_FINISH": 0.05,
            "EPS_DECAY": 0.1,
            "NUM_MINIBATCHES": 32,
            "NUM_EPOCHS": 2,
            "NORM_TYPE": "layer_norm",
            "LR": 5e-4,
            "MAX_GRAD_NORM": 10,
            "LR_LINEAR_DECAY": True,
            "REW_SCALE": 1.0,  # pqn_gymnax.py's default, written out
            "GAMMA": 0.99,
            "LAMBDA": 0.65,
            "ENV_NAME": "Asterix-MinAtar",
            "ENV_KWARGS": {},
            "TEST_DURING_TRAINING": True,
            "TEST_INTERVAL": 0.05,
            "TEST_NUM_ENVS": 128,
            "EPS_TEST": 0.0,
            "WANDB_MODE": "disabled",
        },
        matched={
            "n_envs": 128,
            "n_steps": 32,
            "num_minibatches": 32,
            "n_epochs": 2,
            "learning_rate": 5e-4,
            # pqn_gymnax.py's defaults: two 128-wide LayerNorm ReLU layers.
            "architecture": ("128", "relu", "128", "relu"),
            "gamma": 0.99,
            "q_lambda": 0.65,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "reward_scale": 1.0,
            "epsilon_decay_frac": 0.1,
            "max_grad_norm": 10.0,
            "lr_decay": ("linear", 1e-20, 300 * 32 * 2),
            "td_loss": "half_squared",
        },
        unmatched=(
            *PUREJAXQL_UNMATCHED,
            "evaluation: purejaxql scores the last episode each of its 128 test"
            " envs ends within one 1000-step window, Ajax one episode per env",
        ),
        deviations=PUREJAXQL_DEVIATIONS,
    ),
    "APG": Comparison(
        renames={"num_envs": "n_envs", "horizon_length": "horizon"},
        agent="APG",
        env_id="Pendulum-v1",
        n_timesteps=1_000 * 64 * 32,  # 1,000 updates of 64 envs x 32 steps
        eval_every=0,
        solved=-200.0,
        reference="brax",
        source="brax.training.agents.apg, on refcmp_brax.GymnaxAsBrax",
        environment="gymnax Pendulum-v1 with transition gradients, a port of"
        " Gymnasium's Pendulum-v1",
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
