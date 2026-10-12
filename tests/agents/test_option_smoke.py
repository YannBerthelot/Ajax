"""Options where they meet: every pair of an agent's options, and every src
extension stacked into an agent that folds it, runs. A row is one tiny
logged run, which must update the policy and keep every parameter finite;
what each option computes is tested where it lives. The rows come from the
code: an agent's options are the OPTIONS its signature (and env ids) take,
and a row grows while compatible, so a few rows pair every option with
every other. A switch of a signature (a default of None, a bool or a
string) in neither OPTIONS nor ELSEWHERE, or a src extension without a
builder, fails."""

import importlib
import inspect
import itertools
import pkgutil
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
import ajax.extensions
from ajax.agents.APG.APG import APG
from ajax.agents.APG.networks import PIDHeadConfig
from ajax.agents.TD3.train_TD3 import make_default_action_pipeline
from ajax.extensions import ensemble, expert, exploration, instrumentation
from ajax.extensions import pretrain as pre
from ajax.extensions import target_mods as tm
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.memory import MemoryConfig
from tests.agents.test_agent_contract import CASES, Case
from tests.environments import paid_action


class Expert:
    """A gain expert (env units), called with its state or without."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args: jax.Array) -> Any:
        out = self.step_with_gains(*args[-2:], self.anchor_gains)
        return out if len(args) == 2 else out[0]

    def step_with_gains(self, state: Any, obs: jax.Array, gains: Any) -> Any:
        return jnp.clip(-gains[0] * obs[..., :1], -2.0, 2.0), state + obs[..., :1]


def stateless_expert(obs: jax.Array) -> jax.Array:
    return jnp.clip(-obs[..., :1], -2.0, 2.0)


EXPERT, GRU = Expert(), MemoryConfig(kind="gru", hidden_size=8)
#: Every row's keywords an agent takes: one replay window per env, then updates.
BASE = {"learning_starts": 16, "burn_in": 2, "sequence_length": 4}
BASE |= {"expert_buffer_n_steps": 0, "policy_update_start": 8, "alpha_update_start": 8}
FLAGS = "normalize_observations normalize_rewards normalize_obs_running use_train_frac"
FLAGS += " fixed_alpha store_policy_action fused_grad_clip"
FLAGS += " expose_recent_rollout log_std_state_independent encoder_layer_norm"
FLAGS += " augment_obs_with_expert_action augment_obs_with_expert_state use_pid_policy"
ENVS = ("Pendulum-v1", "fast", "CartPole-v1")
MEMORY = {"memory", "stored_state", "bptt_length"}
#: Each option's keywords, taken by an agent whose signature (env ids) takes them;
#: "squash" negates the default, the CNN fits the env.
OPTIONS: dict[str, dict] = {k: {k: True} for k in FLAGS.split()}
OPTIONS |= {env: {"env_id": env} for env in ENVS}
OPTIONS |= {
    "reward_normalization_gamma": {
        "normalize_rewards": True,
        "reward_normalization_gamma": 0.9,
    },
    "memory": {"memory": GRU},
    "stored_state": {"memory": GRU, "stored_state": True},
    "bptt_length": {"memory": GRU, "bptt_length": 8},
    "n_envs": {"n_envs": 2},
    "max_grad_norm": {"max_grad_norm": 1.0},
    "unroll_length": {"unroll_length": 8},
    "normalize_advantage": {"normalize_advantage": False},
    "squash": {"squash": None},
    "expert_policy": {"expert_policy": EXPERT},
    "eval_expert_policy": {"eval_expert_policy": EXPERT},
    "expert_buffer_n_steps": {"expert_buffer_n_steps": 16},  # the expert prefill
    "target_entropy_initial_per_dim": {"target_entropy_initial_per_dim": 0.0},
    "action_pipeline": {"action_pipeline": None},  # TD3's, for the row's actor
    "n_gradient_steps": {"n_gradient_steps": 2},
    "n_epochs": {"n_epochs": 2},
    "cnn_image_shape": {"cnn_image_shape": None},
    "mlp": {"cnn_image_shape": None},  # the CNN's alternative
    "lr_schedule": {"lr_schedule": "warmup_cosine", "warmup_steps": 2},
    "pid": {"pid": PIDHeadConfig()},
    "actor_bias_init": {"actor_bias_init": "0.1"},
    "critic_bias_init": {"critic_bias_init": "0.1"},
    "encoder_kernel_init": {"encoder_kernel_init": "lecun_uniform"},
    "mean_kernel_init": {"mean_kernel_init": "lecun_uniform"},
}
CNN = {"Pendulum-v1": (1, 3, 1), "fast": (1, 2, 1), "CartPole-v1": (2, 2, 1)}
#: The switches other tests run (tests/...).
ELSEWHERE = dict.fromkeys(
    "actor_cloning_epochs actor_cloning_lr actor_cloning_batch_size pre_train_n_steps"
    " skip_actor_pretrain reset_log_std_after_bc reset_actor_head_after_bc".split(),
    "agents/test_cloning.py",  # deprecated for CloningPretrain, run here
) | {
    "env_params": "agents/test_agent_contract.py",
    "skip_critic_pretrain": "agents/SAC/test_sac.py",
    "jsrl_curriculum": "agents/SAC/test_sac.py",  # deprecated
    "td_target_fn": "agents/DQN/test_DQN.py",
    "td_loss_fn": "agents/PQN/test_PQN.py",
    "q_network_cls": "agents/DQN/test_DQN.py",
    "cnn_spec": "networks/test_cnn.py",
    "episode_length": "environments/test_create.py",
    "system_class": "agents/APG/test_APG.py",
    "controller_factory": "agents/APG/test_APG.py",
    # AVG's inert temperature keywords, warned until removed.
    "alpha_learning_rate": "agents/AVG/test_AVG.py",
    "target_entropy_per_dim": "agents/AVG/test_AVG.py",
}
#: Each src extension's builder and the agents whose rows take it, in stack
#: order (cloning seeds the statistics the other pretraining reads).
EXTENSIONS: dict[str, tuple[Callable[[], Extension], str]] = {
    "CloningPretrain": (lambda: pre.CloningPretrain(EXPERT, 64, 1, 16), "SAC TD3"),
    "MCPretrain": (
        lambda: pre.MCPretrain(EXPERT, 200, 4, 20, online_light_steps=5),
        "SAC",
    ),
    "OnlineBC": (expert.OnlineBC, "SAC"),
    "ValueBox": (tm.ValueBox, "SAC"),
    "CriticBlend": (lambda: tm.CriticBlend(EXPERT), "SAC"),
    "MCVarianceCorrection": (lambda: tm.MCVarianceCorrection(1.0), "SAC"),
    "PhiRefresh": (lambda: pre.PhiRefresh(EXPERT, interval=16, steps=2), "SAC"),
    "IBRL": (lambda: tm.IBRL(EXPERT), "SAC"),
    "LCBGatedBootstrap": (lambda: tm.LCBGatedBootstrap(EXPERT), "SAC"),
    "EDGEExploration": (exploration.EDGEExploration, "SAC"),
    "JSRLCurriculum": (lambda: expert.JSRLCurriculum(episode_length=8), "SAC"),
    "ResidualPolicy": (expert.ResidualPolicy, "SAC"),
    "ExpertValueMetrics": (expert.ExpertValueMetrics, "SAC"),
    "BellmanPretrain": (lambda: pre.BellmanPretrain(n_steps=4), "SAC"),
    "ImitationLoss": (lambda: expert.ImitationLoss(stateless_expert), "APO"),
    "KernelRepulsion": (ensemble.KernelRepulsion, "REDQ"),
    "BiasVorePenalty": (instrumentation.BiasVorePenalty, "AVG"),
    "ConditioningMetrics": (instrumentation.ConditioningMetrics, "PQN"),
    "BiasVoreDecomposition": (instrumentation.BiasVoreDecomposition, "PPO"),
}
ACTING = {"ValueBox", "EDGEExploration", "JSRLCurriculum", "ResidualPolicy"}
PHI_STAR = (
    "OnlineBC ValueBox CriticBlend MCVarianceCorrection PhiRefresh EDGEExploration"
)
NEEDS_EXPERT = "augment_obs_with_expert_action augment_obs_with_expert_state"
NEEDS_EXPERT += " use_pid_policy expert_buffer_n_steps IBRL LCBGatedBootstrap"
NEEDS_EXPERT += " ExpertValueMetrics CloningPretrain ResidualPolicy JSRLCurriculum"
#: What an option or extension needs on in its row: phi* is MCPretrain's,
#: BellmanPretrain reads its prefilled replay.
REQUIRES = {o: {"MCPretrain"} for o in (*PHI_STAR.split(), "BellmanPretrain")}
REQUIRES |= {o: {"expert_policy"} for o in NEEDS_EXPERT.split()}
REQUIRES |= {"MCPretrain": {"expert_policy", "expert_buffer_n_steps"}}
REQUIRES |= {
    o: {"expose_recent_rollout"}
    for o in ("ConditioningMetrics", "BiasVoreDecomposition")
}
EXPERTS = {"expert_policy", "eval_expert_policy"}
#: Options no row takes together: one of each group, and no pair across these.
ONE_OF = [{"normalize_rewards", "reward_normalization_gamma"}, MEMORY, set(ENVS)]
ONE_OF += [{"cnn_image_shape", "mlp"}]  # else the CNN takes every row
EXCLUSIVE = [
    # Recurrent agents refuse experts and extensions (recurrent.py), PPO
    # its feedforward fragments.
    (MEMORY, EXPERTS | set(EXTENSIONS) | {"unroll_length"}),
    # The gain policy bypasses SAC's action slots, and the expert acts on its
    # output (SAC refuses its action in the observation).
    ({"use_pid_policy"}, ACTING | {"augment_obs_with_expert_action"}),
    # Experts act in continuous spaces; ImitationLoss reads a mean action.
    ({"CartPole-v1"}, EXPERTS | {"ImitationLoss"}),
]
TABLE = CASES | {"APG": Case(APG, {"horizon": 1, "pid": None}, ())}
#: The exported agents off TABLE, and the tests that run their options.
OWN_TESTS = {"DreamerV3": "DreamerV3", "TDMPC2": "TDMPC2", "TDMPC2MultiTask": "TDMPC2"}


def _options(name: str) -> list[str]:
    """``name``'s options and extensions."""
    case, takes = TABLE[name], set(inspect.signature(TABLE[name].cls).parameters)
    out = [o for o, kw in OPTIONS.items() if set(kw) <= takes and o not in ENVS]
    out += [env for env in ENVS if env in case.envs]
    return out + [e for e, (_, at) in EXTENSIONS.items() if name in at.split()]


def rows(name: str) -> list[frozenset[str]]:
    """Rows covering every compatible pair of ``name``'s options."""
    options, out = _options(name), []

    def closed(row: set) -> set | None:
        """``row`` with what it requires; None if incompatible."""
        while need := set().union(*(REQUIRES.get(o, set()) for o in row)) - row:
            row = row | need
        clash = any(row & a and row & b for a, b in EXCLUSIVE)
        clash |= any(len(row & group) > 1 for group in ONE_OF)
        return None if clash or not row <= set(options) else row

    pairs = {frozenset(p) for p in itertools.combinations(options, 2) if closed(set(p))}
    while pairs:
        row = closed(set(min(pairs, key=sorted)))
        assert row is not None  # every pair left is compatible
        for option in sorted(options, key=lambda o: -sum(o in p for p in pairs)):
            row = closed(row | {option}) or row  # grow while compatible
        pairs = {p for p in pairs if not p <= row}
        out.append(frozenset(row))
    return out


ROWS = {name: rows(name) for name in sorted(TABLE)}
PARAMS = [
    pytest.param(n, r, id=f"{n}-{i}") for n in ROWS for i, r in enumerate(ROWS[n])
]


def _build(name: str, row: frozenset[str]) -> Any:
    case, params = TABLE[name], inspect.signature(TABLE[name].cls).parameters
    kw = case.kwargs | {k: v for k, v in BASE.items() if k in params}
    for option in sorted(row & set(OPTIONS)):
        kw |= OPTIONS[option]
    env = kw.pop("env_id", None)
    if "squash" in kw:
        kw["squash"] = not params["squash"].default
    if "cnn_image_shape" in row:
        kw["cnn_image_shape"] = CNN[env]
    if "action_pipeline" in kw:
        kw["action_pipeline"] = make_default_action_pipeline("memory" in kw, 0.1)
    exts = [build() for e, (build, _) in EXTENSIONS.items() if e in row]
    if case.cls is APG:
        env, kw["env_params"] = paid_action.make()
    return case.cls(env, extensions=exts, **kw)


@pytest.mark.slow
@pytest.mark.parametrize("name, row", PARAMS)
def test_a_row_of_options_trains(name: str, row: frozenset[str]) -> None:
    """A logged run updates the policy and keeps every parameter finite."""
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    state, _ = _build(name, row).train(
        seed=0, n_timesteps=TABLE[name].n_timesteps, logging_config=logging
    )
    assert int(np.asarray(state.actor_state.step).max()) > 0, "no update"
    for part in ("actor_state", "critic_state"):
        params = getattr(getattr(state, part, None), "params", None)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))


@pytest.mark.parametrize("name", sorted(TABLE))
def test_every_switch_is_in_a_row_or_named_elsewhere(name: str) -> None:
    """A switch joins OPTIONS or names a test that runs it; every option
    takes a row."""
    params = inspect.signature(TABLE[name].cls).parameters.values()
    switches = {
        p.name for p in params if isinstance(p.default, (bool, str, type(None)))
    }
    keys = set().union(*OPTIONS.values()) | {"extensions"}
    missing = switches - keys - set(ELSEWHERE)
    assert not missing, f"give {sorted(missing)} an OPTIONS entry or an ELSEWHERE test"
    for switch in switches & set(ELSEWHERE):
        assert switch in (Path(__file__).parents[1] / ELSEWHERE[switch]).read_text()
    assert set(_options(name)) <= set().union(*ROWS[name])


def test_every_src_extension_has_a_builder() -> None:
    for module in pkgutil.iter_modules(ajax.extensions.__path__):
        importlib.import_module(f"ajax.extensions.{module.name}")
    todo, found = [Extension], set()
    while todo:
        for sub in todo.pop().__subclasses__():
            todo.append(sub)
            found |= {sub.__name__} if sub.__module__.startswith("ajax.") else set()
    assert found == set(EXTENSIONS)


def test_every_agent_has_rows_or_its_own_option_tests() -> None:
    """A new agent joins TABLE (test_agent_contract.CASES) or names its tests."""
    assert set(TABLE) | set(OWN_TESTS) == set(ajax.__all__)
    assert all((Path(__file__).parent / d).is_dir() for d in OWN_TESTS.values())
