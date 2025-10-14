# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# import optax
# import pytest

# from ajax.state import DoubleTrainState, LoadedTrainState


# class SimpleModel(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         return nn.Dense(1)(x)


# def create_model_and_states(key):
#     model = SimpleModel()
#     dummy_input = jnp.ones((1, 4))
#     params = model.init(key, dummy_input)

#     # Create base state
#     base_state = LoadedTrainState.create(
#         apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
#     )

#     # Create second state
#     second_state = LoadedTrainState.create(
#         apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
#     )

#     return model, base_state, second_state


# def test_double_train_state_creation():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)

#     # Test both modes
#     for mode in ["avg", "SAC"]:
#         double_state = DoubleTrainState.from_states(
#             base_state=base_state, second_state=second_state, second_state_type=mode
#         )

#         assert isinstance(double_state, DoubleTrainState)
#         assert isinstance(double_state, LoadedTrainState)
#         assert double_state.second_state_type == mode
#         assert double_state.second_state is second_state


# def test_from_states_validation():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)

#     # Test invalid state type
#     with pytest.raises(ValueError):
#         DoubleTrainState.from_states(
#             base_state=base_state,
#             second_state=second_state,
#             second_state_type="invalid",
#         )


# def test_forward_pass():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)
#     x = jnp.ones((1, 4))

#     double_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="avg"
#     )

#     # Test forward pass
#     output = double_state.apply(double_state.params, x)
#     assert output.shape == (1, 1)

#     # Verify it's the sum of both models' outputs
#     base_output = base_state.apply(base_state.params, x)
#     second_output = second_state.apply(second_state.params, x)
#     assert jnp.allclose(output, base_output + second_output)


# def test_gradient_flow():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)
#     x = jnp.ones((1, 4))

#     double_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="avg"
#     )

#     # Define loss function
#     def loss_fn(params):
#         output = double_state.apply(params, x)
#         return jnp.mean(output**2)

#     # Calculate gradients
#     grads = jax.grad(loss_fn)(double_state.params)

#     # Check that gradients exist
#     assert all(
#         v.shape == p.shape
#         for v, p in zip(
#             jax.tree_util.tree_leaves(grads),
#             jax.tree_util.tree_leaves(double_state.params),
#         )
#     )


# def test_weight_update():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)
#     x = jnp.ones((1, 4))

#     double_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="avg"
#     )

#     # Define loss function and update step
#     def loss_fn(params):
#         output = double_state.apply(params, x)
#         return jnp.mean(output**2)

#     grads = jax.grad(loss_fn)(double_state.params)
#     new_state = double_state.apply_gradients(grads=grads)

#     # Verify state updated but second_state remained unchanged
#     assert not jnp.allclose(
#         new_state.params["params"]["Dense_0"]["kernel"],
#         double_state.params["params"]["Dense_0"]["kernel"],
#     )
#     assert jnp.allclose(
#         new_state.second_state.params["params"]["Dense_0"]["kernel"],
#         double_state.second_state.params["params"]["Dense_0"]["kernel"],
#     )


# def test_different_modes():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)
#     x = jnp.ones((1, 4))

#     # Test both modes
#     avg_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="avg"
#     )

#     SAC_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="SAC"
#     )

#     # Both should work without normalization
#     avg_output = avg_state.apply(avg_state.params, x)
#     SAC_output = SAC_state.apply(SAC_state.params, x)

#     assert avg_output.shape == SAC_output.shape


# def test_from_states_preserves_attributes():
#     key = jax.random.PRNGKey(0)
#     _, base_state, second_state = create_model_and_states(key)

#     double_state = DoubleTrainState.from_states(
#         base_state=base_state, second_state=second_state, second_state_type="avg"
#     )

#     # Check that all important attributes are preserved
#     assert double_state.step == base_state.step
#     assert double_state.apply_fn == base_state.apply_fn
#     assert double_state.tx == base_state.tx
#     assert double_state.opt_state == base_state.opt_state
#     assert jax.tree_util.tree_all(
#         jax.tree.map(
#             lambda x, y: (x == y).all(), double_state.params, base_state.params
#         )
#     )
