import inspect
from typing import Callable, Optional, Tuple, Union

import brax
import brax.envs
import gymnax
from gymnax import EnvParams

from ajax.environments.utils import (
    EnvType,
    check_action_repeat,
    check_if_environment_has_continuous_actions,
    env_action_repeat,
    get_env_type,
    wrapper_chain,
)
from ajax.types import EnvNormalizationInfo
from ajax.wrappers import (
    AutoResetWrapper,
    ClipAction,
    ClipActionBrax,
    FinalObsWrapper,
    FlattenObservationWrapper,
    NoiseWrapper,
    NormalizeVecObservationBrax,
    NormalizeVecObservationGymnax,
    get_wrappers,
)

# External callers (e.g. SafetyExperiments) can register a custom builder for
# a playground env id here, overriding the default wrapper stack below. The
# builder signature is (n_envs, episode_length) -> env with `_ajax_env_id`
# set. Used when the caller needs extra wrappers (safety termination,
# observation augmentation, narrowed reset distribution) that must persist
# through eval's env rebuild. A builder that supports action repeat also
# accepts an ``action_repeat`` keyword; it is only passed when > 1, so the
# two-argument builders registered so far keep working unchanged.
_PLAYGROUND_BUILDERS: dict = {}
_BRAX_BUILDERS: dict = {}


def _builder_accepts_action_repeat(builder: Callable) -> bool:
    try:
        params = inspect.signature(builder).parameters
    except (TypeError, ValueError):  # builtins / C callables: no signature
        return False
    return "action_repeat" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _call_registered_builder(
    kind: str,
    builder: Callable,
    env_id: str,
    n_envs: int,
    episode_length: int,
    action_repeat: int,
):
    """Call a registered ``(n_envs, episode_length)`` builder.

    At ``action_repeat == 1`` the builder is called exactly as before (two
    arguments). Above 1, ``action_repeat`` is forwarded as a keyword when the
    builder's signature takes it (by name or ``**kwargs``), and the built
    env -- not the signature -- decides: it must repeat each action
    ``action_repeat`` times (:func:`env_action_repeat`), otherwise this
    raises. A two-argument builder is accepted only when its env already
    repeats that many times (a repeat hardcoded in its EpisodeWrapper).
    """
    if action_repeat == 1:
        return builder(n_envs, episode_length)
    accepts = _builder_accepts_action_repeat(builder)
    env = (
        builder(n_envs, episode_length, action_repeat=action_repeat)
        if accepts
        else builder(n_envs, episode_length)
    )
    built = env_action_repeat(env)
    if built == action_repeat:
        return env
    if not accepts:
        raise ValueError(
            f"The {kind} builder registered for {env_id!r} does not accept an"
            f" `action_repeat` keyword and its env repeats each action {built}"
            f" times, so action_repeat={action_repeat} cannot be applied. Add"
            " `action_repeat` to the builder's signature (and pass it to its"
            " EpisodeWrapper) to use action repeat with this env."
        )
    raise ValueError(
        f"The {kind} builder registered for {env_id!r} accepted"
        f" action_repeat={action_repeat} but returned an env that repeats each"
        f" action {built} times. Pass `action_repeat` to the builder's"
        " EpisodeWrapper."
    )


def register_brax_builder(env_id: str, builder) -> None:
    """Register a custom brax env builder for `env_id`.

    Mirrors `register_playground_builder` for brax-stack envs. The builder
    takes (n_envs, episode_length) and must return a fully wrapped env
    whose `_ajax_env_id` attribute equals `env_id`. Subsequent calls to
    `_build_brax_env(env_id, ...)` delegate to this builder. Used when a
    safety-experiments-style env wraps a stock brax robot with custom
    termination/observation-augmentation that must survive eval's rebuild.
    To support ``action_repeat > 1`` the builder must also accept an
    ``action_repeat`` keyword (passed only when > 1).
    """
    _BRAX_BUILDERS[env_id] = builder


def register_playground_builder(env_id: str, builder) -> None:
    """Register a custom playground env builder for `env_id`.

    The builder takes (n_envs, episode_length) and must return a fully
    wrapped env whose `_ajax_env_id` attribute equals `env_id`. Subsequent
    calls to `_build_playground_env(env_id, ...)` delegate to this builder.
    To support ``action_repeat > 1`` the builder must also accept an
    ``action_repeat`` keyword (passed only when > 1).
    """
    _PLAYGROUND_BUILDERS[env_id] = builder


def _build_playground_env(
    env_id: str,
    n_envs: int,
    episode_length: int,
    action_repeat: int = 1,
    fresh_reset: bool = True,
    differentiable_reset: bool = False,
):
    """Compose a mujoco_playground env with the same wrapper stack as
    `wrap_for_brax_training`, but inject FinalObsWrapper between the
    episode wrapper and the auto-reset wrapper so the terminal observation
    is preserved in info['final_obs'] for correct truncation bootstrapping.

    BatchRngWrapper sits at the top: the auto-reset wrapper expects an
    already-batched rng (calls `jax.vmap(jax.random.split)(rng)`), so we
    split the caller's single key into `n_envs` keys on reset to keep Ajax's
    unbatched-rng convention intact.

    ``action_repeat`` goes to brax's ``EpisodeWrapper``: one agent step runs
    the simulator ``action_repeat`` times and returns the summed reward.
    ``episode_length`` counts *simulator* steps (the episode lasts
    ``episode_length // action_repeat`` agent steps), and the repeat does
    not stop early on termination (brax's behaviour; the DMC tasks never
    terminate -- deviation E19 in docs/world_models/deviations.md).

    `fresh_reset` selects the auto-reset. True (the default) uses
    `FreshAutoResetWrapper`, which draws a new initial state for every
    episode, as dm_control does. False keeps playground's
    `BraxAutoResetWrapper`, which restarts every episode of env i from the
    same cached first state (a run sees only `n_envs` initial conditions);
    that was Ajax's behaviour before fresh resets became the default, so it
    reproduces playground results produced earlier. `differentiable_reset` is
    passed to `FreshAutoResetWrapper` (see `build_env_from_id`); the cached
    auto-reset computes no reset in `step` and needs no such option.
    Registered builders own their whole stack, auto-reset included, and are
    not affected by these flags.
    """
    check_action_repeat(action_repeat)
    if env_id in _PLAYGROUND_BUILDERS:
        return _call_registered_builder(
            "playground",
            _PLAYGROUND_BUILDERS[env_id],
            env_id,
            n_envs,
            episode_length,
            action_repeat,
        )

    import jax as _jax
    from brax.envs.wrappers import training as brax_training
    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import BraxAutoResetWrapper

    from ajax.wrappers import BatchRngWrapper, FreshAutoResetWrapper

    _overrides = {"impl": "jax"} if _jax.default_backend() == "cpu" else None
    env = registry.load(env_id, config_overrides=_overrides)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
    env = brax_training.VmapWrapper(env)
    env = FinalObsWrapper(env)
    if fresh_reset:
        env = FreshAutoResetWrapper(env, differentiable_reset=differentiable_reset)
    else:
        env = BraxAutoResetWrapper(env)
    env = BatchRngWrapper(env, n_envs=n_envs)
    env._ajax_env_id = env_id
    # Read by evaluate.setup_environment so the eval rebuild keeps the same
    # auto-reset semantics as training.
    env._ajax_fresh_reset = fresh_reset
    return env


def _build_brax_env(
    env_id: str,
    n_envs: int,
    episode_length: int,
    action_repeat: int = 1,
    differentiable_reset: bool = False,
):
    """Build a brax env with the same stack Ajax uses for playground:
    Ajax owns vectorization via VmapWrapper (not brax's native batch_size
    argument to `brax.envs.create`, which composes a different wrapper order
    and has been implicated in the Ant GPU double-free crash). EpisodeWrapper
    exposes truncation in info, FinalObsWrapper preserves the pre-reset
    observation, and AutoResetWrapper re-samples the reset seed.

    ``action_repeat`` is handled by ``EpisodeWrapper`` exactly as in
    :func:`_build_playground_env` (episode_length in simulator steps,
    summed reward, no early stop on termination). `differentiable_reset` is
    passed to AutoResetWrapper (see `build_env_from_id`); registered
    builders are not affected by it.
    """
    check_action_repeat(action_repeat)
    if env_id in _BRAX_BUILDERS:
        return _call_registered_builder(
            "brax",
            _BRAX_BUILDERS[env_id],
            env_id,
            n_envs,
            episode_length,
            action_repeat,
        )

    from brax.envs.wrappers import training as brax_training

    env = brax.envs._envs[env_id]()
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
    env = brax_training.VmapWrapper(env, batch_size=n_envs)
    env = FinalObsWrapper(env)
    env = AutoResetWrapper(env, differentiable_reset=differentiable_reset)
    env._ajax_env_id = env_id
    return env


def build_env_from_id(
    env_id: str,
    n_envs: int = 1,
    fresh_reset: bool = True,
    *,
    action_repeat: int = 1,
    differentiable_reset: bool = False,
    **kwargs,
) -> tuple[EnvType, Optional[EnvParams]]:
    """Build a wrapped env from its id (gymnax, mujoco_playground or brax).

    ``fresh_reset`` only concerns mujoco_playground envs (see
    ``_build_playground_env``); gymnax and Ajax's brax stack already draw a
    new initial state for every episode.

    ``action_repeat`` (default 1) repeats each agent action for that many
    simulator steps on brax / playground envs (see
    :func:`_build_playground_env`); ``episode_length`` (a keyword, default
    1000) then counts simulator steps. gymnax envs do not support it.

    ``differentiable_reset`` concerns the auto-resets that compute a fresh
    reset inside ``step``: brax envs, and playground envs with
    ``fresh_reset=True`` (the default). With ``differentiable_reset=False``
    (the default) they evaluate the reset only on steps where some env is
    done, inside a ``lax.while_loop``, because the reset can cost as much as
    many env steps. Gradients through such an env still flow with respect
    to the actions, the policy parameters and the env state, and forward
    mode (``jax.jvp``) works with respect to anything.
    Only reverse mode *through the reset itself* -- ``jax.grad`` with
    respect to something the reset depends on, such as physics parameters
    of the env -- fails, loudly, at trace time: "Reverse-mode
    differentiation does not work for lax.while_loop". Pass
    ``differentiable_reset=True`` for that case: the reset is then evaluated
    on every step and kept only for done envs, which reverse mode can cross,
    at the cost of a reset per step. Transitions are the same either way,
    up to float rounding. Gymnax envs (whose auto-reset is always computed)
    and playground's cached auto-reset ignore the flag, and evaluation
    (``evaluate.setup_environment``), which never differentiates, rebuilds
    the env with the default.
    """
    check_action_repeat(action_repeat)
    if env_id in gymnax.registered_envs:
        if action_repeat > 1:
            raise NotImplementedError(
                f"action_repeat={action_repeat} is not supported on gymnax envs"
                f" ({env_id!r}); it is implemented for brax / mujoco_playground"
                " envs only (no gymnax task in the reproduced papers uses it)."
            )
        env, env_params = gymnax.make(env_id)
        # Ajax's actor/critic heads consume a flat observation vector: a Dense
        # layer applied to an unflattened (H, W, C) observation produces one
        # action distribution *per spatial cell* instead of one per env. That
        # silently yields (n_envs, H, W)-shaped actions, which classic-control
        # envs absorb by broadcasting but grid envs reject when indexing
        # (e.g. MinAtar's `action_set[action]`). Flatten here so every gymnax
        # env presents a 1-D observation; the optional CNN encoder re-forms
        # (H, W, C) from the flat vector via `cnn_image_shape`.
        if len(env.observation_space(env_params).shape) > 1:
            env = FlattenObservationWrapper(env)
        return env, env_params  # TODO : see how to have env_params not mess up the rest

    episode_length = kwargs.get("episode_length", 1000)

    # External callers (e.g. SafetyExperiments) may register a playground
    # builder for an env id that is not part of `mp_registry.ALL_ENVS`
    # (e.g. a custom MJX env composed from our own MJCF). Honour the
    # builder registry before falling through to the upstream registry.
    if env_id in _PLAYGROUND_BUILDERS:
        return _build_playground_env(
            env_id,
            n_envs=n_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
        ), None

    try:
        from mujoco_playground import registry as mp_registry

        if env_id in mp_registry.ALL_ENVS:
            return _build_playground_env(
                env_id,
                n_envs=n_envs,
                episode_length=episode_length,
                action_repeat=action_repeat,
                fresh_reset=fresh_reset,
                differentiable_reset=differentiable_reset,
            ), None
    except ImportError:
        pass

    if env_id in _BRAX_BUILDERS or env_id in list(brax.envs._envs.keys()):
        return _build_brax_env(
            env_id,
            n_envs=n_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            differentiable_reset=differentiable_reset,
        ), None
    raise ValueError(f"Environment {env_id} not found in gymnax or brax")


def add_ajax_wrappers(
    env: EnvType,
    *,
    clip: bool = False,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    gamma: Optional[float] = None,
    apply_obs_normalization: bool = True,
    train: bool = True,
    norm_info: Optional[EnvNormalizationInfo] = None,
) -> EnvType:
    """Ajax's own layers over a task env: the observation / reward
    normaliser (``train=False`` with ``norm_info`` freezes its statistics),
    then the ``[-1, 1]`` action clip. The one place they are composed."""
    ClipAction, NormalizeVecObservation = get_wrappers(get_env_type(env))
    if normalize_obs or normalize_reward:
        env = NormalizeVecObservation(
            env,
            train=train,
            norm_info=norm_info,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            gamma=gamma if normalize_reward else None,
            apply_normalization=apply_obs_normalization,
        )
    if clip:
        env = ClipAction(env)
    return env


_CLIPS = (ClipAction, ClipActionBrax)
_NORMALISERS = (NormalizeVecObservationGymnax, NormalizeVecObservationBrax)


def strip_ajax_wrappers(env: EnvType) -> tuple[EnvType, dict]:
    """The task env under Ajax's own layers, and the keywords that rebuild
    those layers with :func:`add_ajax_wrappers` (the observation noise of
    :func:`prepare_env` is not rebuilt). Everything under them is the task:
    the user's wrappers, the flattening of :func:`build_env_from_id`, a brax
    stack's time limit."""
    layers: dict = {}
    for layer in wrapper_chain(env):
        if isinstance(layer, _CLIPS):
            layers["clip"] = True
        elif isinstance(layer, _NORMALISERS):
            layers |= {
                "normalize_obs": layer.normalize_obs,
                "normalize_reward": layer.normalize_reward,
                "gamma": layer.gamma,
                "apply_obs_normalization": layer.apply_normalization,
            }
        elif not isinstance(layer, NoiseWrapper):
            return layer, layers
    raise ValueError(f"No task env under the Ajax wrappers of {env!r}")


def prepare_env(
    env_id: Union[str, EnvType],
    episode_length: Optional[int] = None,
    env_params: Optional[EnvParams] = None,
    n_envs: int = 1,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    gamma: Optional[float] = None,  # Discount factor for reward normalization
    noise_scale: Optional[float] = None,
    apply_obs_normalization: bool = True,
    action_repeat: int = 1,
) -> Tuple[EnvType, Optional[EnvParams], Union[str, EnvType], bool]:
    check_action_repeat(action_repeat)
    if isinstance(env_id, str):
        env, env_params = build_env_from_id(
            env_id,
            episode_length=episode_length or 1000,
            n_envs=n_envs,
            action_repeat=action_repeat,
        )
    else:
        env = env_id  # Assume prebuilt env
        # Action repeat lives in the EpisodeWrapper Ajax composes when it
        # builds an env from its id, and the eval rebuild
        # (``evaluate.setup_environment``) rebuilds from that id. A prebuilt
        # env carries its own wrapper stack, so neither a requested repeat
        # nor one the env already carries can be kept consistent between
        # the env, ``EnvironmentConfig.action_repeat`` and the eval rebuild:
        # both raise. The check reads the env itself, not only the argument.
        env_repeat = env_action_repeat(env)
        if action_repeat > 1 or env_repeat != 1:
            raise ValueError(
                f"Action repeat (action_repeat={action_repeat}, the prebuilt env"
                f" repeats each action {env_repeat} times) is only supported for"
                " envs built from an id. Pass the env id and action_repeat, or"
                " register a builder that accepts `action_repeat`"
                " (register_brax_builder / register_playground_builder) and pass"
                " its id."
            )
    continuous = check_if_environment_has_continuous_actions(env)
    env = add_ajax_wrappers(
        env,
        clip=normalize_obs or normalize_reward,
        normalize_obs=normalize_obs,
        normalize_reward=normalize_reward,
        gamma=gamma,
        apply_obs_normalization=apply_obs_normalization,
    )
    if noise_scale is not None:
        print("noise wrapper")
        env = NoiseWrapper(env, scale=noise_scale)
    return env, env_params, env_id, continuous
