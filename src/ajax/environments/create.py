from typing import Optional, Tuple, Union

import brax
import brax.envs
import gymnax
from gymnax import EnvParams

from ajax.environments.utils import (
    EnvType,
    check_if_environment_has_continuous_actions,
    get_env_type,
)
from ajax.wrappers import AutoResetWrapper, FinalObsWrapper, NoiseWrapper, get_wrappers


# External callers (e.g. SafetyExperiments) can register a custom builder for
# a playground env id here, overriding the default wrapper stack below. The
# builder signature is (n_envs, episode_length) -> env with `_ajax_env_id`
# set. Used when the caller needs extra wrappers (safety termination,
# observation augmentation, narrowed reset distribution) that must persist
# through eval's env rebuild.
_PLAYGROUND_BUILDERS: dict = {}
_BRAX_BUILDERS: dict = {}


def register_brax_builder(env_id: str, builder) -> None:
    """Register a custom brax env builder for `env_id`.

    Mirrors `register_playground_builder` for brax-stack envs. The builder
    takes (n_envs, episode_length) and must return a fully wrapped env
    whose `_ajax_env_id` attribute equals `env_id`. Subsequent calls to
    `_build_brax_env(env_id, ...)` delegate to this builder. Used when a
    safety-experiments-style env wraps a stock brax robot with custom
    termination/observation-augmentation that must survive eval's rebuild.
    """
    _BRAX_BUILDERS[env_id] = builder


def register_playground_builder(env_id: str, builder) -> None:
    """Register a custom playground env builder for `env_id`.

    The builder takes (n_envs, episode_length) and must return a fully
    wrapped env whose `_ajax_env_id` attribute equals `env_id`. Subsequent
    calls to `_build_playground_env(env_id, ...)` delegate to this builder.
    """
    _PLAYGROUND_BUILDERS[env_id] = builder


def _build_playground_env(env_id: str, n_envs: int, episode_length: int):
    """Compose a mujoco_playground env with the same wrapper stack as
    `wrap_for_brax_training`, but inject FinalObsWrapper between the
    episode wrapper and the auto-reset wrapper so the terminal observation
    is preserved in info['final_obs'] for correct truncation bootstrapping.

    BatchRngWrapper sits at the top: playground's BraxAutoResetWrapper expects
    an already-batched rng (calls `jax.vmap(jax.random.split)(rng)`), so we
    split the caller's single key into `n_envs` keys on reset to keep Ajax's
    unbatched-rng convention intact.
    """
    if env_id in _PLAYGROUND_BUILDERS:
        return _PLAYGROUND_BUILDERS[env_id](n_envs, episode_length)

    from brax.envs.wrappers import training as brax_training
    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import BraxAutoResetWrapper

    from ajax.wrappers import BatchRngWrapper

    import jax as _jax
    _overrides = {"impl": "jax"} if _jax.default_backend() == "cpu" else None
    env = registry.load(env_id, config_overrides=_overrides)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=1)
    env = brax_training.VmapWrapper(env)
    env = FinalObsWrapper(env)
    env = BraxAutoResetWrapper(env)
    env = BatchRngWrapper(env, n_envs=n_envs)
    env._ajax_env_id = env_id
    return env


def _build_brax_env(env_id: str, n_envs: int, episode_length: int):
    """Build a brax env with the same stack Ajax uses for playground:
    Ajax owns vectorization via VmapWrapper (not brax's native batch_size
    argument to `brax.envs.create`, which composes a different wrapper order
    and has been implicated in the Ant GPU double-free crash). EpisodeWrapper
    exposes truncation in info, FinalObsWrapper preserves the pre-reset
    observation, and AutoResetWrapper re-samples the reset seed.
    """
    if env_id in _BRAX_BUILDERS:
        return _BRAX_BUILDERS[env_id](n_envs, episode_length)

    from brax.envs.wrappers import training as brax_training

    env = brax.envs._envs[env_id]()
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=1)
    env = brax_training.VmapWrapper(env, batch_size=n_envs)
    env = FinalObsWrapper(env)
    env = AutoResetWrapper(env)
    env._ajax_env_id = env_id
    return env


def build_env_from_id(
    env_id: str,
    n_envs: int = 1,
    **kwargs,
) -> tuple[EnvType, Optional[EnvParams]]:
    if env_id in gymnax.registered_envs:
        env, env_params = gymnax.make(env_id)
        return env, env_params  # TODO : see how to have env_params not mess up the rest

    episode_length = kwargs.get("episode_length", 1000)

    # External callers (e.g. SafetyExperiments) may register a playground
    # builder for an env id that is not part of `mp_registry.ALL_ENVS`
    # (e.g. a custom MJX env composed from our own MJCF). Honour the
    # builder registry before falling through to the upstream registry.
    if env_id in _PLAYGROUND_BUILDERS:
        return _build_playground_env(env_id, n_envs=n_envs, episode_length=episode_length), None

    try:
        from mujoco_playground import registry as mp_registry

        if env_id in mp_registry.ALL_ENVS:
            return _build_playground_env(env_id, n_envs=n_envs, episode_length=episode_length), None
    except ImportError:
        pass

    if env_id in _BRAX_BUILDERS or env_id in list(brax.envs._envs.keys()):
        return _build_brax_env(env_id, n_envs=n_envs, episode_length=episode_length), None
    raise ValueError(f"Environment {env_id} not found in gymnax or brax")


def prepare_env(
    env_id: Union[str, EnvType],
    episode_length: Optional[int] = None,
    env_params: Optional[EnvParams] = None,
    n_envs: int = 1,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    gamma: Optional[float] = None,  # Discount factor for reward normalization
    noise_scale: Optional[float] = None,
) -> Tuple[EnvType, Optional[EnvParams], Union[str, EnvType], bool]:
    if isinstance(env_id, str):
        env, env_params = build_env_from_id(
            env_id,
            episode_length=episode_length or 1000,
            n_envs=n_envs,
        )
    else:
        env = env_id  # Assume prebuilt env
    continuous = check_if_environment_has_continuous_actions(env)

    mode = get_env_type(env)
    if normalize_obs or normalize_reward:
        ClipAction, NormalizeVecObservation = get_wrappers(mode)

    # Apply wrappers based on flags
    if normalize_obs or normalize_reward:
        env = ClipAction(
            NormalizeVecObservation(
                env,
                normalize_reward=normalize_reward,
                normalize_obs=normalize_obs,
                gamma=gamma if normalize_reward else None,
            )
        )
    if noise_scale is not None:
        print("noise wrapper")
        env = NoiseWrapper(env, scale=noise_scale)
    return env, env_params, env_id, continuous
