import jax.numpy as jnp
import pytest
from flax.linen.normalization import _l2_normalize
from jax import random

from ajax.utils import (
    get_and_prepare_hyperparams,
    linear_schedule,
    maybe_eval_str,
    maybe_parse_linear_schedule,
    maybe_parse_str_to_dict,
    online_normalize,
    parse_dict_from_hyperparams,
    parse_hyperparams,
    parse_schedules,
    remove_keys_from_dict,
    split_train_init_kwargs,
)


def test_online_normalize_initial():
    x = jnp.array([[1.0, 2.0, 3.0]])
    count = 0
    mean = jnp.zeros_like(x)
    mean_2 = jnp.zeros_like(x)

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    assert new_count == 1
    assert jnp.allclose(new_mean, x)
    assert jnp.allclose(new_mean_2, 0.0)
    assert jnp.allclose(sigma, 0.0)
    assert jnp.allclose(x_norm, 0.0)  # x == mean, std == 0 => norm = 0


def test_online_normalize_update():
    x = jnp.array([[4.0, 5.0, 6.0]])
    count = 1
    mean = jnp.array([[1.0, 2.0, 3.0]])
    mean_2 = jnp.array([[0.0, 0.0, 0.0]])

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    expected_mean = jnp.array([[2.5, 3.5, 4.5]])
    expected_mean_2 = jnp.array([[4.5, 4.5, 4.5]])
    expected_sigma = jnp.array([[2.25, 2.25, 2.25]])
    expected_std = jnp.sqrt(expected_sigma)
    expected_x_norm = (x - expected_mean) / expected_std  # delta2 / std

    assert new_count == 2
    assert jnp.allclose(new_mean, expected_mean)
    assert jnp.allclose(new_mean_2, expected_mean_2)
    assert jnp.allclose(sigma, expected_sigma)
    assert jnp.allclose(x_norm, expected_x_norm)


def test_online_normalize_multiple_updates():
    x = jnp.array([[7.0, 8.0, 9.0]])
    count = 2
    mean = jnp.array([[2.5, 3.5, 4.5]])
    mean_2 = jnp.array([[4.5, 4.5, 4.5]])

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    expected_mean = jnp.array([[4.0, 5.0, 6.0]])
    expected_mean_2 = jnp.array([[18.0, 18.0, 18.0]])
    expected_sigma = jnp.array([[6.0, 6.0, 6.0]])
    expected_std = jnp.sqrt(expected_sigma)
    expected_x_norm = (x - expected_mean) / expected_std

    assert new_count == 3
    assert jnp.allclose(new_mean, expected_mean)
    assert jnp.allclose(new_mean_2, expected_mean_2)
    assert jnp.allclose(sigma, expected_sigma)
    assert jnp.allclose(x_norm, expected_x_norm)


def test_single_vector_normalization():
    x = jnp.array([3.0, 4.0])
    x_normed = _l2_normalize(x, axis=0)
    norm = jnp.linalg.norm(x_normed, ord=2)
    assert jnp.allclose(norm, 1.0, atol=1e-6), f"Norm was {norm}"


def test_batch_vector_normalization_axis1():
    x = jnp.array([[3.0, 4.0], [0.0, 5.0]])
    x_normed = _l2_normalize(x, axis=1)
    norms = jnp.linalg.norm(x_normed, ord=2, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6).all(), f"Norms were {norms}"


def test_random_batch_normalization():
    key = random.PRNGKey(0)
    x = random.normal(key, (10, 128))
    x_normed = _l2_normalize(x, axis=1)
    norms = jnp.linalg.norm(x_normed, ord=2, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6).all(), f"Norms were {norms}"


def test_axis0_normalization():
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_normed = _l2_normalize(x, axis=0)
    norms = jnp.linalg.norm(x_normed, ord=2, axis=0)
    assert jnp.allclose(norms, 1.0, atol=1e-6).all(), f"Norms were {norms}"


def test_parse_hyperparams(tmp_path):
    yaml_content = """
    CartPole-v1:
      n_envs: 8
      n_timesteps: !!float 1e5
      policy: 'MlpPolicy'
      n_steps: 32
      batch_size: 256
      gae_lambda: 0.8
      gamma: 0.98
      n_epochs: 20
      ent_coef: 0.0
      learning_rate: lin_0.001
      clip_range: lin_0.2
    """
    yaml_file = tmp_path / "test_hyperparams.yml"
    yaml_file.write_text(yaml_content)

    hyperparams = parse_hyperparams(str(yaml_file))
    assert "CartPole-v1" in hyperparams
    assert hyperparams["CartPole-v1"]["n_envs"] == 8
    assert hyperparams["CartPole-v1"]["learning_rate"] == "lin_0.001"


def test_linear_schedule():
    result = linear_schedule(1.0, 2.0, 2)
    assert result == 1.0, "Linear schedule calculation is incorrect."


def test_parse_linear_schedule():
    schedule = maybe_parse_linear_schedule("learning_rate", "lin_0.001", 10, 100)
    assert callable(schedule), "Parsed schedule should be callable."
    assert schedule(5) == 0.0005, "Linear schedule parsing is incorrect."


def test_parse_schedules():
    hyperparams = {
        "learning_rate": "lin_0.001",
        "clip_range": "lin_0.2",
        "gamma": 0.98,
    }
    parsed = parse_schedules(hyperparams, 10, 100)
    assert callable(parsed["learning_rate"]), "Learning rate should be callable."
    assert callable(parsed["clip_range"]), "Clip range should be callable."
    assert parsed["gamma"] == 0.98, "Gamma should remain unchanged."


def test_remove_keys_from_dict():
    hyperparams = {
        "n_envs": 8,
        "policy": "MlpPolicy",
        "n_steps": 32,
    }
    result = remove_keys_from_dict(hyperparams, keys_to_remove=("policy",))
    assert "policy" not in result, "Key 'policy' should be removed."
    assert "n_envs" in result, "Key 'n_envs' should remain."


def test_split_train_init_kwargs():
    hyperparams = {
        "n_timesteps": 1e5,
        "policy": "MlpPolicy",
        "n_steps": 32,
    }
    init_kwargs, train_kwargs = split_train_init_kwargs(
        hyperparams, train_keys=("n_timesteps",)
    )
    assert "n_timesteps" in train_kwargs, "Key 'n_timesteps' should be in train_kwargs."
    assert "policy" in init_kwargs, "Key 'policy' should be in init_kwargs."


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("false", False),
        ("[1, 2, 3]", [1, 2, 3]),
        ("{'key': 'value'}", {"key": "value"}),
        ("torch.nn.ReLU", "torch.nn.ReLU"),  # Keep as string
        ("invalid_string", "invalid_string"),
    ],
)
def test_maybe_eval_str(value, expected):
    result = maybe_eval_str(value)
    assert result == expected, f"Failed to parse value: {value}"


def test_parse_str_to_dict():
    str_dict = "dict(key1= value1, key2= [1, 2, 3], key3= true, key4= torch.nn.ReLU)"

    result = maybe_parse_str_to_dict(str_dict)
    assert result["key1"] == "value1", "Failed to parse key1."
    assert result["key2"] == [1, 2, 3], "Failed to parse key2."
    assert result["key3"] is True, "Failed to parse key3."
    assert result["key4"] == "torch.nn.ReLU", "Failed to keep key4 as string."


def test_parse_dict_from_hyperparams():
    hyperparams = {
        "policy_kwargs": (
            "dict(log_std_init=-2, ortho_init=False, activation_fn=torch.nn.ReLU)"
        ),
        "gamma": 0.99,
    }
    parsed = parse_dict_from_hyperparams(hyperparams)
    assert isinstance(parsed, dict), "Failed to parse policy_kwargs."
    assert (
        parsed["activation_fn"] == "torch.nn.ReLU"
    ), "Failed to keep activation_fn as string."
    assert parsed["gamma"] == 0.99, "Gamma should remain unchanged."


def test_get_and_prepare_hyperparams(tmp_path):
    yaml_content = """
    CartPole-v1:
      n_envs: 8
      n_timesteps: !!float 1e5
      policy: 'MlpPolicy'
      n_steps: 32
      batch_size: 256
      gae_lambda: 0.8
      gamma: 0.98
      n_epochs: 20
      ent_coef: 0.0
      learning_rate: lin_0.001
      clip_range: lin_0.2
      policy_kwargs: "dict(log_std_init=-2, ortho_init=False, activation_fn=torch.nn.ReLU)"
    """
    yaml_file = tmp_path / "test_hyperparams.yml"
    yaml_file.write_text(yaml_content)

    init_kwargs, train_kwargs = get_and_prepare_hyperparams(
        filename=str(yaml_file),
        env_id="CartPole-v1",
        train_keys=("n_timesteps",),
        keys_to_remove=("policy",),
    )

    assert (
        "n_timesteps" in train_kwargs
    ), "Failed to include 'n_timesteps' in train_kwargs."
    assert "policy" not in init_kwargs, "Failed to remove 'policy' from init_kwargs."
    assert callable(
        init_kwargs["actor_learning_rate"]
    ), "Failed to parse 'learning_rate' as callable."
