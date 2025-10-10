import ast
import inspect
import json
from dataclasses import fields
from types import MappingProxyType
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
import yaml
from flax.core import FrozenDict
from jax.tree_util import Partial as partial


def replace_zeros_with_ones(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x == 0, 1, x)


@partial(
    jax.jit,
    static_argnames=["train", "eps", "shift"],
)
def online_normalize(
    x: jnp.array,
    count: int,
    mean: float,
    mean_2: float,
    eps: float = 1e-8,
    train: bool = True,
    shift: bool = True,
    returns: Optional[jax.Array] = None,
) -> tuple[jnp.array, int, float, float, float]:
    input_x = x

    if train:
        x = x if returns is None else returns
        x = x.reshape(1, -1) if len(x.shape) < 2 else x
        # assert jnp.ndim(mean) == 2, f"Mean must be 2D, got {jnp.ndim(mean)}D"

        batch_size = x.shape[0]
        batch_mean = jnp.nanmean(x, axis=0, keepdims=True)
        batch_mean_2 = jnp.nanmean((x - batch_mean) ** 2, axis=0, keepdims=True)

        total_count = count + batch_size

        delta = batch_mean - mean
        mean = mean + delta * batch_size / total_count
        mean_2 = (
            mean_2
            + batch_mean_2 * batch_size
            + (delta**2) * count * batch_size / total_count
        )
        mean_2 = replace_zeros_with_ones(mean_2)
        count = total_count

    variance = mean_2 / count
    std = jnp.sqrt(variance + eps)
    x_norm = (input_x - jnp.nanmean(mean, axis=0) * shift) / jnp.nanmean(std, axis=0)

    x_norm = x_norm.reshape(input_x.shape)  # Ensure output shape matches input shape
    assert (
        x_norm.shape == input_x.shape
    ), f"x_norm shape {x_norm.shape} does not match input_x shape {input_x.shape}"

    return (
        x_norm,
        count,
        mean,
        mean_2,
        variance,
    )


def parse_hyperparams(file_name: str):
    with open(file_name) as stream:
        try:
            hyperparams_data = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
        return hyperparams_data


def linear_schedule(t: int, init_x: int, max_t: int):
    return jnp.minimum(1, (1 - t / max_t)) * init_x


def maybe_parse_linear_schedule(
    key: str, linear_schedule_str: str, n_update_steps: int, n_timesteps: int
) -> Callable | str:
    """
    Parses a linear schedule string and returns a callable function.

    Args:
        linear_schedule_str (str): A string representing the linear schedule.

    Returns:
        Callable: A function that computes the linear schedule.
    """
    if not isinstance(linear_schedule_str, str):
        return linear_schedule_str
    if "lin_" not in linear_schedule_str:
        return linear_schedule_str
    elif "learning_rate" in key:
        return optax.schedules.linear_schedule(
            init_value=float(linear_schedule_str.split("_")[1]),
            end_value=0.0,
            transition_steps=n_update_steps,
        )  # step (internal) based schedule

    else:

        def schedule(t):  # timestep (global) based schedule
            init_value = float(linear_schedule_str.split("_")[1])
            return linear_schedule(t, init_value, n_timesteps)

    return schedule


def parse_schedules(hyperparams: dict, n_update_steps: int, n_timesteps: int) -> dict:
    return {
        k: maybe_parse_linear_schedule(k, v, n_update_steps, n_timesteps)
        for k, v in hyperparams.items()
    }


def remove_keys_from_dict(hyperparams: dict, keys_to_remove: tuple[str, ...]) -> dict:
    """
    Removes specified keys from a dictionary.

    Args:
        hyperparams (dict): The dictionary from which to remove keys.
        keys_to_remove (tuple[str]): A tuple of keys to remove.

    Returns:
        dict: The dictionary with specified keys removed.
    """
    return {k: v for k, v in hyperparams.items() if k not in keys_to_remove}


def split_train_init_kwargs(
    hyperparams: dict, train_keys: tuple[str, ...]
) -> tuple[dict, dict]:
    """
    Splits hyperparameters into training and initialization arguments.
    Args:
        hyperparams (dict): The dictionary of hyperparameters.
        train_keys (tuple[str]): Keys to include in training arguments.
    Returns:
        tuple[dict, dict]: A tuple containing the initialization and training arguments.
    """
    train_kwargs = {k: v for k, v in hyperparams.items() if k in train_keys}
    init_kwargs = {k: v for k, v in hyperparams.items() if k not in train_keys}
    return init_kwargs, train_kwargs


def maybe_eval_str(value: str) -> Any:
    """
    Safely parses a string to an int, float, bool, list, or callable reference if applicable.

    Args:
        value (str): The string to parse.

    Returns:
        Parsed Python object.
    """
    # 1. Handle booleans
    val_lower = value.lower()
    if val_lower in ("true", "false"):
        return val_lower == "true"

    # 2. Try safe literal evaluation (lists, dicts, etc.)
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # # 3. Try resolving callables like "torch.nn.ReLU"
    # if "." in value:
    #     try:
    #         module_path, attr = value.rsplit(".", 1)
    #         mod = importlib.import_module(module_path)
    #         return getattr(mod, attr)
    #     except (ModuleNotFoundError, AttributeError, ValueError):
    #         pass

    # 3. Return original string if nothing else works
    return value


def smart_split_no_nested_dicts(s):
    result = []
    current = []
    depth = 0
    in_dict = False
    i = 0

    while i < len(s):
        char = s[i]

        # Detect start of dict(...) block
        if not in_dict and s[i : i + 5] == "dict(":
            in_dict = True
            depth = 1
            current.append("dict(")
            i += 5
            continue

        # Inside dict(...) block
        if in_dict:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            current.append(char)
            if depth == 0:
                in_dict = False
            i += 1
            continue

        # Normal bracket tracking
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1

        # Top-level comma split
        if char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)

        i += 1

    if current:
        result.append("".join(current).strip())

    return result


def parse_key_value_pairs(s):
    items = smart_split_no_nested_dicts(s)
    parsed = []
    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)  # only split on first =
            parsed.append([key.strip(), value.strip()])
        else:
            parsed.append([item.strip()])
    return parsed


def normalize_kv_pairs(pairs):
    result = []
    print(f"PAIRS {pairs}")
    if isinstance(pairs[0], str):
        pairs = [pairs]
    print(pairs)
    for entry in pairs:
        print(entry)
        if len(entry) == 2:
            result.append(entry)
        elif len(entry) > 2:
            key = entry[0]
            value = " ".join(entry[1:])  # join remaining parts with space
            result.append([key, value])
        else:
            raise ValueError(f"Unexpected empty entry: {entry}")
    return result


def maybe_parse_str_to_dict(
    config_str: str, delimiter: str = ",", key_value_delimiter: str = "="
) -> dict | str:
    if not isinstance(config_str, str):
        return config_str
    if not config_str.startswith("dict(") or not config_str.endswith(")"):
        return config_str
    # pre_dict = [
    #     item.split(key_value_delimiter)
    #     for item in parse_key_value_pairs(config_str.lstrip("dict(").rstrip(")"))
    # ]
    pre_dict = parse_key_value_pairs(config_str.lstrip("dict(").rstrip(")"))

    full_dict = {k.strip(): maybe_eval_str(v.strip()) for k, v in pre_dict}
    return full_dict


def parse_dict_from_hyperparams(hyperparams: dict) -> dict:
    new_hyperparams = {k: maybe_parse_str_to_dict(v) for (k, v) in hyperparams.items()}
    new_hyperparams.update(new_hyperparams.pop("policy_kwargs", {}))  # type: ignore[arg-type]
    return new_hyperparams


def parse_architecture_from_strings(activation_fn: str, net_arch: str):
    # Normalize activation function (e.g., 'nn.ReLU' -> 'relu')

    if activation_fn is None or net_arch is None:
        return None, None

    activation_name = activation_fn.split(".")[-1].lower()

    # Parse the net_arch string safely (expects dict format like "dict(pi=[256, 256], vf=[128, 128])")

    arch_dict = maybe_parse_str_to_dict(net_arch)

    if not isinstance(arch_dict, dict):
        raise ValueError(
            f"Expected net_arch to be a dict, got {type(arch_dict)} with value"
            f" {arch_dict}"
        )

    def build_string(arch_list):
        return "[" + ",".join(f"{size},{activation_name}" for size in arch_list) + "]"

    actor_arch = (
        build_string(arch_dict.get("pi", [])).lstrip("[").rstrip("]").split(",")
    )
    critic_arch = (
        build_string(arch_dict.get("vf", [])).lstrip("[").rstrip("]").split(",")
    )

    return actor_arch, critic_arch


def rename_dict_keys(dict, translate_dict: dict) -> dict:
    """
    Rename keys in a dictionary based on a translation dictionary.

    Args:
        dict (dict): The original dictionary.
        translate_dict (dict): A dictionary mapping old keys to new keys.

    Returns:
        dict: A new dictionary with renamed keys.
    """
    return {translate_dict.get(k, k): v for k, v in dict.items()}


def get_and_prepare_hyperparams(
    filename: str,
    env_id: str,
    train_keys: tuple[str, ...] = ("n_timesteps",),
    keys_to_remove: tuple[str, ...] = (
        "policy",
        "vf_coef",
        "use_sde",
        "sde_sample_freq",
    ),
    translate_dict: dict = MappingProxyType(  # type: ignore[assignment]
        {"log_std_init": "actor_bias_init"}
    ),
) -> tuple[dict, dict]:
    raw_hyperparams = parse_hyperparams(filename)
    env_hyperparams = raw_hyperparams[env_id]
    env_hyperparams = remove_keys_from_dict(env_hyperparams, keys_to_remove)
    n_update_steps = (
        (env_hyperparams["n_timesteps"] * env_hyperparams.get("n_epochs", 1))
        // (env_hyperparams["n_envs"] * env_hyperparams["n_steps"])
    ) + 1  # How many updates will be done in total

    env_hyperparams = parse_dict_from_hyperparams(env_hyperparams)

    env_hyperparams = parse_schedules(
        env_hyperparams, n_update_steps, n_timesteps=env_hyperparams["n_timesteps"]
    )
    actor_architecture, critic_architecture = parse_architecture_from_strings(
        env_hyperparams.pop("activation_fn", None),
        env_hyperparams.pop("net_arch", None),
    )
    if "normalize" in env_hyperparams.keys():
        env_hyperparams["normalize_observations"] = env_hyperparams.pop("normalize")
        env_hyperparams["normalize_rewards"] = env_hyperparams["normalize_observations"]
    if actor_architecture is not None:
        env_hyperparams["actor_architecture"] = actor_architecture
    if critic_architecture is not None:
        env_hyperparams["critic_architecture"] = critic_architecture
    if "ortho_init" in env_hyperparams.keys():
        env_hyperparams["actor_kernel_init"] = (
            "orthogonal" if env_hyperparams.pop("ortho_init") else "he_uniform"
        )
        env_hyperparams["critic_kernel_init"] = env_hyperparams["actor_kernel_init"]
    if "learning_rate" in env_hyperparams.keys():
        env_hyperparams["actor_learning_rate"] = env_hyperparams.pop("learning_rate")
        env_hyperparams["critic_learning_rate"] = env_hyperparams["actor_learning_rate"]
    env_hyperparams = rename_dict_keys(env_hyperparams, translate_dict=translate_dict)

    init_kwargs, train_kwargs = split_train_init_kwargs(
        env_hyperparams, train_keys=train_keys
    )

    return init_kwargs, train_kwargs


def fill_with_nan(dataclass):
    """
    Recursively fills all fields of a dataclass with jnp.nan.
    """
    nan = jnp.ones(1) * jnp.nan
    dict = {}
    for field in fields(dataclass):
        sub_dataclass = field.type
        if hasattr(
            sub_dataclass, "__dataclass_fields__"
        ):  # Check if the field is another dataclass
            dict[field.name] = fill_with_nan(sub_dataclass)
        else:
            dict[field.name] = nan
    return dataclass(**dict)


def get_update_kwargs(config, update_fn, **kwargs):
    sig = inspect.signature(update_fn)
    arg_names = [
        param.name for param in sig.parameters.values()
    ]  # remove agent_state and _
    kwargs = {key: val for key, val in kwargs.items() if key in arg_names}

    values_to_remove = ["agent_state", "_", "buffer", "recurrent", "action_dim"]
    arg_names = [
        param.name
        for param in sig.parameters.values()
        if param.name not in values_to_remove
    ]
    update_kwargs = {
        key: config.__dict__[key] for key in arg_names if key in config.__dict__
    }

    update_kwargs.update(kwargs)
    return FrozenDict(update_kwargs)


def get_update_scan_fn(static_kwargs, config, update_agent):
    static_kwargs = get_update_kwargs(config, update_agent, **static_kwargs)
    return partial(
        update_agent,
        **static_kwargs,
    )


def compare_frozen_dicts(dict1: FrozenDict, dict2: FrozenDict) -> bool:
    """
    Compares two FrozenDicts to check if they are equal.

    Args:
        dict1 (FrozenDict): The first FrozenDict.
        dict2 (FrozenDict): The second FrozenDict.

    Returns:
        bool: True if the FrozenDicts are equal, False otherwise.
    """
    for key in dict1.keys():
        if key not in dict2:
            return False
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, FrozenDict) and isinstance(value2, FrozenDict):
            if not compare_frozen_dicts(value1, value2):
                return False
        elif not jnp.allclose(value1, value2):
            return False
    return True


def get_one(_: Any) -> float:
    return jnp.ones(1)


def make_json_serializable(d):
    """
    Return a new dict where all values that are not JSON-serializable
    are converted to strings.
    """
    serializable_dict = {}
    for k, v in d.items():
        try:
            json.dumps(v)  # attempt to serialize
            serializable_dict[k] = v
        except (TypeError, OverflowError):
            serializable_dict[k] = str(v)
    return serializable_dict
