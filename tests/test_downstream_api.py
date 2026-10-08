"""Pin the public API that downstream projects depend on.

Research code in sibling repositories imports the symbols below and calls
them with the listed keyword arguments (inventory taken from an AST scan of
those repositories on 2026-10-08). Removing or renaming one breaks that
code without any Ajax test noticing, so this test fails instead. To change
one deliberately, migrate the downstream callers first, then update the
inventory here.
"""

import importlib
import inspect

import pytest

# "module:qualname" -> keyword arguments downstream passes when calling it.
DOWNSTREAM_API: dict[str, tuple[str, ...]] = {
    "ajax:APG": (
        "controller_factory",
        "env_params",
        "extensions",
        "horizon",
        "learning_rate",
        "n_envs",
        "system_class",
    ),
    "ajax:APG.contextual_controller": (
        "context",
        "controller_factory",
        "d_model",
        "env_params",
        "extensions",
        "horizon",
        "learning_rate",
        "max_grad_norm",
        "n_envs",
        "n_heads",
        "n_layers",
        "warmup_steps",
        "weight_decay",
    ),
    "ajax:PPO": ("env_id", "env_params", "n_envs"),
    "ajax:SAC": (
        "action_scale",
        "actor_architecture",
        "actor_learning_rate",
        "alpha_init",
        "alpha_learning_rate",
        "batch_size",
        "critic_architecture",
        "critic_learning_rate",
        "env_id",
        "env_params",
        "eval_expert_policy",
        "gamma",
        "max_grad_norm",
        "n_envs",
        "normalize_observations",
        "normalize_rewards",
        "target_entropy_per_dim",
        "tau",
    ),
    "ajax.agents.SAC.SAC:SAC": (),
    "ajax.agents.TD3.TD3:TD3": (
        "actor_architecture",
        "actor_learning_rate",
        "batch_size",
        "critic_architecture",
        "critic_learning_rate",
        "env_id",
        "env_params",
        "exploration_noise",
        "gamma",
        "max_grad_norm",
        "policy_delay",
        "target_noise_clip",
        "target_policy_noise",
        "tau",
    ),
    "ajax.agents.APG:CurriculumStage": ("agent", "n_timesteps", "name"),
    "ajax.agents.APG:train_curriculum": ("logging_config", "num_episode_test", "seed"),
    "ajax.agents.APG.curriculum:CurriculumStage": ("agent", "n_timesteps", "name"),
    "ajax.agents.APG.curriculum:train_curriculum": ("seed",),
    "ajax.agents.APG.networks:Controller": (
        "action_dim",
        "input_architecture",
        "memory",
        "pid",
        "squash",
    ),
    "ajax.agents.APG.networks:PIDHeadConfig": (),
    "ajax.agents.APG.train_APG:build_controller": (),
    "ajax.agents.APG.train_APG:init_APG": ("controller_factory",),
    "ajax.agents.APG.train_APG:make_policy_step": (),
    "ajax.agents.APG.train_APG:rollout_returns": ("stateful",),
    "ajax.checkpoint:checkpoint_exists": (),
    "ajax.checkpoint:restore_into": (),
    "ajax.checkpoint:save_checkpoint": (),
    "ajax.environments.create:build_env_from_id": ("episode_length", "n_envs"),
    "ajax.environments.create:register_brax_builder": (),
    "ajax.environments.differentiable:Rollout": (
        "action",
        "done",
        "info",
        "next_obs",
        "obs",
        "resets",
        "reward",
    ),
    "ajax.environments.differentiable:closed_loop_rollout": ("n_envs",),
    "ajax.environments.differentiable:with_transition_gradients": (),
    "ajax.environments.interaction:reset": (),
    "ajax.environments.interaction:step": (),
    "ajax.environments.model_reference:LinearReferenceModel": (),
    "ajax.environments.model_reference:ModelReferenceWrapper": (
        "action_dim",
        "output_idx",
    ),
    "ajax.environments.model_reference:StepReference": (),
    "ajax.environments.system_class:FixedSystem": (),
    "ajax.environments.system_class:SystemClass": (),
    "ajax.environments.system_class:UniformPerturbation": ("fields", "scale"),
    "ajax.environments.system_class:broadcast_env_params": (),
    "ajax.environments.utils:get_action_dim": (),
    "ajax.evaluate:evaluate": (
        "actor_state",
        "env",
        "env_params",
        "num_episodes",
        "recurrent",
        "rng",
    ),
    "ajax.extensions.base:Extension": (),
    "ajax.extensions.base:ExtensionContext": (),
    "ajax.extensions.expert:ExpertGuidance": (),
    "ajax.extensions.expert:JSRLCurriculum": (
        "decay_frac",
        "episode_length",
        "expert_policy",
    ),
    "ajax.extensions.expert:ResidualPolicy": ("expert_policy", "scale"),
    "ajax.extensions.exploration:EDGEExploration": (
        "decay_frac",
        "epsilon_floor",
        "expert_policy",
        "fixed_prob",
        "gate",
        "lcb_asymmetric",
        "lcb_beta_decay_k",
        "lcb_beta_init",
        "lcb_temperature",
        "tau",
    ),
    "ajax.extensions.target_mods:IBRL": ("expert_policy",),
    "ajax.logging.wandb_logging:LoggingConfig": (
        "config",
        "folder",
        "horizon",
        "log_frequency",
        "project_name",
        "run_name",
        "sweep",
        "use_tensorboard",
        "use_wandb",
    ),
    "ajax.logging.wandb_logging:load_scalars_from_tfevents": (),
    "ajax.modules.pid_head:PIDOutputHead": (
        "kd_init",
        "ki_init",
        "kp_init",
        "n_outputs",
    ),
    "ajax.modules.pid_head:init_pid_carry": (),
    "ajax.networks.memory:MemoryCell": (),
    "ajax.networks.memory:MemoryConfig": (
        "hidden_size",
        "kind",
        "num_heads",
        "num_layers",
        "window",
    ),
    "ajax.networks.memory:init_carry": (),
    "ajax.networks.memory:zeros_carry_like": (),
    "ajax.plane.plane_exps_utils:get_mode": (),
    "ajax.schedule:warmup_cosine_schedule": (),
    "ajax.wrappers:AutoResetWrapper": (),
    "ajax.wrappers:FinalObsWrapper": (),
    "ajax.wrappers:GymnaxWrapper": (),
}


# Callables whose ``**kwargs`` go to another public callable: the downstream
# keywords are checked against that callable's signature.
KWARGS_FORWARDED_TO = {"ajax:APG.contextual_controller": "ajax:APG"}

# Callables that read downstream keywords out of ``**kwargs`` by name. A
# signature cannot pin those keywords, only that ``**kwargs`` is still taken
# (making them explicit parameters would let this test pin them).
KWARGS_READ_BY_NAME = {
    "ajax.environments.create:build_env_from_id": ("episode_length",)
}


def _resolve(target: str):
    module_name, qualname = target.split(":")
    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _keyword_names(obj) -> set[str]:
    """Keyword parameters ``obj`` accepts, following ``**kwargs`` up the MRO."""
    callables = [obj]
    if isinstance(obj, type):
        callables += [b.__init__ for b in obj.__mro__[1:] if "__init__" in vars(b)]
    names: set[str] = set()
    for fn in callables:
        params = inspect.signature(fn).parameters.values()
        names |= {
            p.name
            for p in params
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        if not any(p.kind is p.VAR_KEYWORD for p in params):
            break
    return names


def _takes_var_keyword(obj) -> bool:
    params = inspect.signature(obj).parameters.values()
    return any(p.kind is p.VAR_KEYWORD for p in params)


@pytest.mark.parametrize("target", sorted(DOWNSTREAM_API))
def test_downstream_symbol_and_keywords_exist(target):
    obj = _resolve(target)
    accepted = _keyword_names(obj)
    if target in KWARGS_FORWARDED_TO:
        assert _takes_var_keyword(obj)
        accepted |= _keyword_names(_resolve(KWARGS_FORWARDED_TO[target]))
    if target in KWARGS_READ_BY_NAME:
        assert _takes_var_keyword(obj)
        accepted |= set(KWARGS_READ_BY_NAME[target])
    missing = set(DOWNSTREAM_API[target]) - accepted
    assert not missing, f"{target} no longer accepts {sorted(missing)}"
