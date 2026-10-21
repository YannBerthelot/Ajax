# an updated version of rlpyt's `rlpyt.envs.gym` to work with gymnasium environments
import numpy as np
import gymnasium
from gymnasium import Wrapper
from collections import namedtuple

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import is_namedtuple_class


class GymEnvWrapper(Wrapper):
    """Gym-style wrapper for converting the Openai Gym interface to the
    rlpyt interface.  Action and observation spaces are wrapped by rlpyt's
    ``GymSpaceWrapper``.

    Output `env_info` is automatically converted from a dictionary to a
    corresponding namedtuple, which the rlpyt sampler expects.  For this to
    work, every key that might appear in the gym environments `env_info` at
    any step must appear at the first step after a reset, as the `env_info`
    entries will have sampler memory pre-allocated for them (so they also
    cannot change dtype or shape).  (see `EnvInfoWrapper`, `build_info_tuples`,
    and `info_to_nt` in file or more help/details)

    Warning:
        Unrecognized keys in `env_info` appearing later during use will be
        silently ignored.

    This wrapper looks for gym's ``TimeLimit`` env wrapper to
    see whether to add the field ``timeout`` to env info.
    """

    def __init__(self, env,
            act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o, _ = self.env.reset()
        o, r, d, t, info = self.env.step(self.env.action_space.sample())
        info["timeout"] = t
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        # self.spaces = namedtuple()
        # self.spaces["observation"]=self.observation_space
        # self.spaces["action"]=self.action_space
        build_info_tuples(info)

    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, t, info = self.env.step(a)
        d = d or t
        obs = self.observation_space.convert(o)
        info["timeout"] = t
        info = info_to_nt(info)
        if isinstance(r, float):
            r = np.dtype("float32").type(r)  # Scalar float32.
        return EnvStep(obs, r, d, info)

    def reset(self):
        """Resets the environment and converts the observation from gym to rlpyt
        format (i.e. if dict-to-composite)."""
        o, _ = self.env.reset()
        return self.observation_space.convert(o)

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

def build_info_tuples(info, name="info"):
    # Define namedtuples at module level for pickle.
    # Only place rlpyt uses pickle is in the sampler, when getting the
    # first examples, to avoid MKL threading issues...can probably turn
    # that off, (look for subprocess=True --> False), and then might
    # be able to define these directly within the class.
    ntc = globals().get(name)  # Define at module level for pickle.
    info_keys = [str(k).replace(".", "_") for k in info.keys()]
    if ntc is None:
        globals()[name] = namedtuple(name, info_keys)
    elif not (is_namedtuple_class(ntc) and
            sorted(ntc._fields) == sorted(info_keys)):
        raise ValueError(f"Name clash in globals: {name}.")
    for k, v in info.items():
        if isinstance(v, dict):
            build_info_tuples(v, "_".join([name, k]))


def info_to_nt(value, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = globals()[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, "_".join([name, k]))
        for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)


class EnvInfoWrapper(Wrapper):
    """Gym-style environment wrapper to infill the `env_info` dict of every
    ``step()`` with a pre-defined set of examples, so that `env_info` has
    those fields at every step and they are made available to the algorithm in
    the sampler's batch of data.
    """

    def __init__(self, env, info_example):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._sometimes_info = info_example

    def step(self, action):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        o, r, d, t, info = super().step(action)
        d = d or t
        # Try to make info dict same key structure at every step.
        return o, r, d, infill_info(info, self._sometimes_info)
    def reset(self):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        o, _ = self.env.reset()
        # Try to make info dict same key structure at every step.
        return self.observation_space.convert(o)


def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info



def gym_make(*args, info_example=None, **kwargs):
    """Use as factory function for making instances of gym environment with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    if info_example is None:
        return GymEnvWrapper(gymnasium.make(*args, **kwargs))
    else:
        return GymEnvWrapper(EnvInfoWrapper(
            gymnasium.make(*args, **kwargs), info_example))
