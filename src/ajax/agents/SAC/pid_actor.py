"""
PID Actor for SAC.

The actor network predicts PID gains from the full observation. The policy mean
is then computed as:

    mean = gains @ pid_terms

where:
    pid_terms = [error]                              (P-only, obs_derivative_idx=None)
    pid_terms = [error, obs[obs_derivative_idx]]     (PD, obs_derivative_idx set)

    error = obs[obs_target_idx] - obs[obs_current_idx]

The log_std head is independent of the PID structure (standard SAC).

This is a drop-in replacement for Actor — it returns SquashedNormal(mean, std)
and is fully compatible with residual RL (clip(a_expert + a_pid, -1, 1)).
"""
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.normalization import _l2_normalize

from ajax.agents.SAC.utils import SquashedNormal
from ajax.networks.utils import parse_architecture
from ajax.types import ActivationFunction


@dataclass
class PIDActorConfig:
    """Indices into the observation vector used to build PID error terms.

    obs_current_idx:    index of the current value (e.g. altitude).
    obs_target_idx:     index of the target / set-point (e.g. target altitude).
    obs_derivative_idx: index of the derivative (e.g. altitude rate).
                        Set to None for P-only control.
    """
    obs_current_idx: int
    obs_target_idx: int
    obs_derivative_idx: Optional[int] = None


class PIDActorNetwork(nn.Module):
    """SAC actor that produces PID-structured actions.

    The neural network predicts per-dimension PID gains from the full
    observation. The policy mean for each action dimension is then:

        mean_i = K_P_i * error + K_D_i * deriv   (PD mode)
        mean_i = K_P_i * error                   (P-only mode)

    where error = obs[obs_target_idx] - obs[obs_current_idx].

    The gains head is zero-initialized so that training starts from the
    zero-gain policy (safe initialization for flight control).
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    obs_current_idx: int
    obs_target_idx: int
    obs_derivative_idx: Optional[int]
    penultimate_normalization: bool = False

    @property
    def n_terms(self) -> int:
        return 2 if self.obs_derivative_idx is not None else 1

    def setup(self):
        layers = parse_architecture(self.input_architecture)
        self._encoder_net = nn.Sequential(layers)
        self._encoder_norm = nn.LayerNorm()
        # Zero-init gains → starts from the zero-action policy.
        self.gains_head = nn.Dense(
            self.action_dim * self.n_terms,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="gains",
        )
        # Standard SAC log_std head.
        self.log_std_head = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-1.0),
            name="log_std",
        )

    def _encode(self, obs):
        features = self._encoder_net(obs)
        if self.penultimate_normalization:
            return _l2_normalize(features, axis=1)
        return self._encoder_norm(features)

    def __call__(self, obs) -> SquashedNormal:
        # --- Build PID error terms ---
        error = obs[..., self.obs_target_idx] - obs[..., self.obs_current_idx]
        if self.obs_derivative_idx is not None:
            pid_terms = jnp.stack(
                [error, obs[..., self.obs_derivative_idx]], axis=-1
            )  # (..., 2)
        else:
            pid_terms = error[..., None]  # (..., 1)

        # --- Predict gains from full observation ---
        features = self._encode(obs)
        raw_gains = self.gains_head(features)  # (..., action_dim * n_terms)
        log_std = jnp.clip(self.log_std_head(features), -20, 2)

        # Reshape gains to (..., action_dim, n_terms) and contract with pid_terms
        gains = raw_gains.reshape(
            raw_gains.shape[:-1] + (self.action_dim, self.n_terms)
        )
        mean = jnp.einsum("...ij,...j->...i", gains, pid_terms)

        return SquashedNormal(mean, jnp.exp(log_std))
