"""Per-task presets (ajax.agents.presets, DeclaredAgent.preset).

The contract, from the API's documentation:

- ``Agent.preset(name, env_id, **overrides)`` builds the agent with the
  values of the task ``name`` in the PRESETS table of the agent's module;
- the overrides given win over the preset's values; the agent's other
  fields keep their defaults;
- an unknown name raises ValueError naming the presets that exist;
- the run config records the preset's name under "preset";
- every entry of every table the library ships sets the agent's own fields,
  and builds; the agent's algorithm page names it under "## Presets".
"""

import dataclasses
import re
import sys
import types

import pytest

from ajax import SAC
from ajax.agents.families import DeclaredAgent
from ajax.agents.presets import Preset, presets_of
from ajax.extensions.expert import ImitationLoss
from tests.doc_examples import ROOT
from tests.test_exports import library_agents

#: The table of TunedSAC, the module's agent: a task setting two fields.
PRESETS = {"Swing": Preset({"tau": 0.01, "gamma": 0.9}, "this test")}


@dataclasses.dataclass(kw_only=True, eq=False, repr=False)
class TunedSAC(SAC):
    """SAC defined in this module, so that this module's PRESETS is its table."""


def test_a_preset_sets_its_values_and_leaves_the_rest_at_the_defaults():
    """Catches a preset whose values do not reach the agent, or one that
    resets fields it does not name."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1")

    assert (agent.tau, agent.gamma) == (0.01, 0.9)
    assert agent.batch_size == TunedSAC("Pendulum-v1").batch_size


def test_the_overrides_given_win_over_the_preset():
    """Catches a preset applied over the caller's own arguments."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1", tau=0.5, batch_size=32)

    assert (agent.tau, agent.gamma, agent.batch_size) == (0.5, 0.9, 32)


def test_the_run_config_records_the_preset_name():
    """Catches a logged run that does not say which preset built it."""
    agent = TunedSAC.preset("Swing", "Pendulum-v1")

    assert agent.config["preset"] == "Swing"
    assert agent.config["tau"] == 0.01


def test_an_unknown_name_raises_and_names_the_presets():
    """Catches a misspelt task silently building the defaults."""
    with pytest.raises(ValueError, match="'Swnig'.*Swing"):
        TunedSAC.preset("Swnig", "Pendulum-v1")


def test_a_subclass_defined_elsewhere_does_not_take_its_parents_table(
    monkeypatch,
):
    """Catches a subclass in another module (a variant of the algorithm)
    built with values tuned for its parent."""
    module = types.ModuleType("a_module_without_presets")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Elsewhere(TunedSAC):
        __module__ = module.__name__

    with pytest.raises(ValueError, match="its presets: none"):
        Elsewhere.preset("Swing", "Pendulum-v1")


SHIPPED = sorted(
    (
        (cls, name)
        for cls in library_agents()
        if issubclass(cls, DeclaredAgent)
        for name in presets_of(cls)
    ),
    key=lambda case: (case[0].__name__, case[1]),
)
SHIPPED_IDS = [f"{cls.__name__}-{name}" for cls, name in SHIPPED]


def test_the_library_ships_presets():
    """Catches the tables dropping out of their modules (a renamed PRESETS)."""
    assert {cls.__name__ for cls, _ in SHIPPED} >= {"ASAC", "PPO", "PQN", "TQC"}


@pytest.mark.parametrize(("cls", "name"), SHIPPED, ids=SHIPPED_IDS)
def test_every_shipped_preset_sets_the_agents_fields_and_builds(cls, name):
    """Catches an entry naming a field the agent does not have (a typo, a
    renamed field) or a value the agent refuses. The preset builds on a
    stand-in env the agent acts in, a field's check needing no simulator of
    the task: brax's fast, which also takes an action repeat, unless the
    agent only takes discrete actions."""
    preset = presets_of(cls)[name]
    init_fields = {field.name for field in dataclasses.fields(cls) if field.init}
    assert set(preset.overrides) <= init_fields
    assert preset.citation

    env_id = "CartPole-v1" if cls.action_space == "discrete" else "fast"
    agent = cls.preset(name, env_id)

    for field, value in preset.overrides.items():
        assert getattr(agent, field) == value, field


@pytest.mark.parametrize(("cls", "name"), SHIPPED, ids=SHIPPED_IDS)
def test_every_shipped_preset_is_named_on_its_algorithm_page(cls, name):
    """Catches a preset added or renamed without its page saying so."""
    page = ROOT / "docs" / "algorithms" / f"{cls.__name__.lower()}.md"
    section = page.read_text().split("\n## Presets\n", 1)[1].split("\n## ", 1)[0]

    assert re.search(f'[`"]{name}[`"]', section)


def asac_task(beta, tau_theta, learning_starts=10_000, buffer_size=1_000_000):
    """One hparams/<task>-v5/asac.yaml: beta (:3), buffer_size (:5),
    learning_starts (:9) and tau_theta (:12). Ajax's alpha_init is the
    temperature 1 / beta."""
    return {
        "alpha_init": 1 / beta,
        "tau_theta": tau_theta,
        "learning_starts": learning_starts,
        "buffer_size": buffer_size,
    }


#: Each shipped preset as its reference states it, read from the reference's
#: own files (local clones at the cited commits), not from Ajax's tables.
#: Where Ajax's field has another unit than the reference's, the comment
#: works the conversion.
REFERENCE = {
    # BobMak/asac@4b68862, hparams/<task>-v5/asac.yaml.
    ("ASAC", "Ant"): asac_task(5, 0.005),
    ("ASAC", "HalfCheetah"): asac_task(5, 0.005),
    ("ASAC", "Hopper"): asac_task(5, 0.001),
    ("ASAC", "Humanoid"): asac_task(20, 0.001),
    ("ASAC", "InvertedDoublePendulum"): asac_task(5, 0.001, learning_starts=0),
    ("ASAC", "InvertedPendulum"): asac_task(5, 0.005, learning_starts=0),
    ("ASAC", "Pusher"): asac_task(5, 0.005),
    ("ASAC", "Reacher"): asac_task(5, 0.005, buffer_size=100_000),
    ("ASAC", "Swimmer"): asac_task(20, 0.005),
    ("ASAC", "Walker2d"): asac_task(5, 0.0005),
    # adityab/CrossQ@a318a26 train.py:200: 0.99 unless the env is Swimmer-v4.
    ("CrossQ", "Swimmer"): {"gamma": 0.9999},
    # google-deepmind/dqn@9d9b1d1 run_gpu:12-32: lr 0.00025, discount 0.99,
    # replay_memory 1000000, minibatch_size 32, learn_start 50000,
    # update_freq 4, ep 1 to eps_end 0.1 over eps_endt = replay_memory steps
    # counted from learn_start (NeuralQLearner.lua:368-370), target_q 10000,
    # steps 50000000. Ajax counts the target's interval in updates,
    # 10000 / 4, and the decay as a fraction of the run, 1e6 / 5e7.
    ("DQN", "Atari"): {
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "buffer_size": 1_000_000,
        "batch_size": 32,
        "learning_starts": 50_000,
        "train_frequency": 4,
        "target_update_interval": 2_500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_start": 50_000,
        "epsilon_decay_frac": 0.02,
    },
    # danijar/dreamerv3@29eb964 configs.yaml: run.num_envs 16 (:44), dmc
    # repeat 2 (:84), dmc_proprio takes size12m and train_ratio 512 (:232-234).
    ("DreamerV3", "DMC"): {
        "model_size": "12m",
        "train_ratio": 512,
        "n_envs": 16,
        "action_repeat": 2,
    },
    # Kostrikov et al. (2021), Appendix B: expectile 0.9 and inverse
    # temperature 10 on AntMaze (the cited config file was not at hand).
    ("IQL", "AntMaze"): {"expectile": 0.9, "temperature": 10.0},
    # openai/baselines@ea25b9e ppo2/defaults.py:15-22: nsteps 128,
    # nminibatches 4, noptepochs 4, ent_coef .01, lr 2.5e-4 (times the
    # remaining fraction), cliprange 0.1, and no value_network='copy' (the
    # MuJoCo block's), so one network. 8 actors: Schulman et al. (2017),
    # Table 5.
    ("PPO", "Atari"): {
        "n_envs": 8,
        "n_steps": 128,
        "num_minibatches": 4,
        "n_epochs": 4,
        "actor_learning_rate": 2.5e-4,
        "critic_learning_rate": 2.5e-4,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "share_encoder": True,
    },
    # DLR-RM/stable-baselines3 2.9.0, PPO.__init__ (ppo/ppo.py:84-95):
    # learning_rate 3e-4 (one optimiser over both networks), n_steps 2048,
    # batch_size 64, n_epochs 10, gamma 0.99, gae_lambda 0.95, clip_range
    # 0.2, clip_range_vf None (no value clip), normalize_advantage True,
    # ent_coef 0.0, vf_coef 0.5, max_grad_norm 0.5; Adam's eps 1e-5
    # (common/policies.py:472). Its value loss is vf_coef * MSE (:244) and
    # Ajax's vf_coef * 0.5 * MSE, so 0.5 there is 2 * 0.5 here. It
    # normalises observations and rewards only through a VecNormalize
    # wrapper, which its PPO does not add.
    # openai/spinningup @038665d, spinup/examples/pytorch/pg_math/2_rtg_pg.py:
    # train(env_name, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000)
    # (:24-25), a tanh MLP (:9), one Adam (:55) on the logits net alone (no
    # baseline), weights reward_to_go(ep_rews) (:17-22, undiscounted), the
    # raw weights in the loss (:50-52, no normalisation).
    ("REINFORCE", "rtg_pg"): {
        "baseline": False,
        "normalize_advantage": False,
        "gamma": 1.0,
        "n_steps": 5000,
        "actor_architecture": ("32", "tanh"),
        "actor_learning_rate": 1e-2,
    },
    ("PPO", "SB3"): {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_value_loss": False,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 2 * 0.5,
        "max_grad_norm": 0.5,
        "adam_eps": 1e-5,
        "normalize_observations": False,
        "normalize_rewards": False,
    },
    # mttga/purejaxql@47af6d7 config/alg/pqn_cartpole.yaml: NUM_ENVS 32,
    # NUM_STEPS 64, EPS_FINISH 0.2, EPS_DECAY 0.2, NUM_MINIBATCHES 16,
    # NUM_EPOCHS 4, HIDDEN_SIZE 256 x NUM_LAYERS 2, LR 0.0001,
    # LR_LINEAR_DECAY True, REW_SCALE 0.1, LAMBDA 0.95.
    ("PQN", "CartPole"): {
        "n_envs": 32,
        "n_steps": 64,
        "epsilon_end": 0.2,
        "epsilon_decay_frac": 0.2,
        "num_minibatches": 16,
        "n_epochs": 4,
        "architecture": ("256", "relu", "256", "relu"),
        "learning_rate": 0.0001,
        "learning_rate_decay": "linear",
        "reward_scale": 0.1,
        "q_lambda": 0.95,
    },
    # The same repository's pqn_atari.yaml: NUM_ENVS 128, NUM_STEPS 32,
    # EPS_FINISH 0.001, EPS_DECAY 0.1, NUM_MINIBATCHES 32, NUM_EPOCHS 2,
    # LR 0.00025, LR_LINEAR_DECAY False, no REW_SCALE (1), LAMBDA 0.65.
    ("PQN", "Atari"): {
        "n_envs": 128,
        "n_steps": 32,
        "epsilon_end": 0.001,
        "epsilon_decay_frac": 0.1,
        "num_minibatches": 32,
        "n_epochs": 2,
        "learning_rate": 0.00025,
        "learning_rate_decay": None,
        "reward_scale": 1.0,
        "q_lambda": 0.65,
    },
    # watchernyu/REDQ@7b5d1bf redq/algos/core.py:11: MBPO's target entropy
    # of the whole action, Hopper -1, HalfCheetah -3, Walker2d -3, Ant -4,
    # Humanoid -2, over the gym task's action dimensions (3, 6, 6, 8, 17).
    ("REDQ", "Hopper"): {"target_entropy_per_dim": -1 / 3},
    ("REDQ", "HalfCheetah"): {"target_entropy_per_dim": -3 / 6},
    ("REDQ", "Walker2d"): {"target_entropy_per_dim": -3 / 6},
    ("REDQ", "Ant"): {"target_entropy_per_dim": -4 / 8},
    ("REDQ", "Humanoid"): {"target_entropy_per_dim": -2 / 17},
    # sfujim/TD3_BC@8791ad7: alpha 2.5 (TD3_BC.py:76, main.py:56), states
    # normalised (main.py:57), the cloned actions the dataset's.
    ("TD3", "D4RL"): {
        "extensions": (ImitationLoss(None, target="replay", alpha=2.5),),
        "normalize_observations": True,
    },
    # nicklashansen/tdmpc2@eece801 tdmpc2/envs/mujoco.py:57-58, in make_env,
    # for every task the module builds: cfg.discount_max = 0.99, cfg.rho = 0.7.
    ("TDMPC2", "MuJoCo"): {"discount_max": 0.99, "rho": 0.7},
    # bayesgroup/tqc_pytorch@65114eb README.md:42-48.
    ("TQC", "Hopper"): {"top_quantiles_to_drop_per_net": 5},
    ("TQC", "HalfCheetah"): {"top_quantiles_to_drop_per_net": 0},
    ("TQC", "Walker2d"): {"top_quantiles_to_drop_per_net": 2},
    ("TQC", "Ant"): {"top_quantiles_to_drop_per_net": 2},
    ("TQC", "Humanoid"): {"top_quantiles_to_drop_per_net": 2},
}


def test_every_shipped_preset_has_its_reference_statement():
    """Catches a preset added without the reference's values beside it."""
    assert {(cls.__name__, name) for cls, name in SHIPPED} == set(REFERENCE)


@pytest.mark.parametrize(("cls", "name"), SHIPPED, ids=SHIPPED_IDS)
def test_every_shipped_preset_holds_its_references_values(cls, name):
    """Catches a value mistyped in a table, or converted to Ajax's unit
    wrongly: the other tests here read the table on both sides."""
    stated = REFERENCE[(cls.__name__, name)]
    overrides = presets_of(cls)[name].overrides

    assert set(overrides) == set(stated)
    for field, value in stated.items():
        if isinstance(value, float):
            # Fractions written two ways (1 / 5 and 0.2) differ by an ulp.
            assert overrides[field] == pytest.approx(value, rel=1e-12), field
        else:
            assert overrides[field] == value, field
