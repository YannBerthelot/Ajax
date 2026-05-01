from typing import Optional

from flax import struct

from ajax.agents.UDRL.buffer import SegmentBuffer
from ajax.state import BaseAgentConfig, BaseAgentState


@struct.dataclass
class UDRLState(BaseAgentState):
    """Adds the dynamic command target tracked across iterations and the
    episode-segment replay buffer that backs UDRL training (Algorithm 3).

    UDRL bootstrapping requires the rollout command (d_r, d_h) to stay near
    what the current policy can actually achieve; a fixed init too far from
    the policy's true return distribution leaves the (s, c) → a mapping
    untrained at c, so rollouts conditioned on c produce garbage. Each
    training iteration refreshes ``command_target_*`` from the buffer's
    top-K episode returns (Algorithm 5).
    """

    command_target_return: float = 1.0
    command_target_return_std: float = 0.0
    command_target_horizon: float = 100.0
    buffer: Optional[SegmentBuffer] = None


@struct.dataclass
class UDRLConfig(BaseAgentConfig):
    """Hyperparameters for Upside-Down RL (Schmidhuber, 2019)."""

    gamma: float = 1.0
    n_steps: int = 64
    batch_size: int = 64
    n_epochs: int = 4
    command_return_init: float = 1.0
    command_horizon_init: float = 100.0
    bc_loss_type: str = "nll"  # "nll" or "mse"
    # Dynamic-target settings: every iter, set the command_target to the
    # top-K episode returns' mean + ``command_return_boost`` * std (UDRL
    # paper §4.2). Horizon target is the mean horizon of the same top-K.
    command_topk: int = 32
    command_return_boost: float = 1.0
    # Smoothing factor on the target update: new = (1 - tau) * old + tau * proposal.
    # tau=1 means no smoothing (replace each iter).
    command_target_tau: float = 1.0
    # Replay buffer capacity (number of stored segments). With segment_length
    # = n_steps and n_envs parallel envs, the buffer holds roughly
    # ``buffer_capacity * n_steps * n_envs / mean_episode_length`` episodes.
    buffer_capacity: int = 64
    # Number of gradient batches per iteration drawn from the buffer
    # (Algorithm 3). The paper updates a fixed number of gradient steps
    # rather than the on-policy "n_epochs over current rollout".
    n_updates_per_iter: int = 64
    # Linear scaling applied to (dr, dh) BEFORE the actor sees them. Paper
    # uses 0.02 / 0.01 for LunarLander; the network learns more cleanly when
    # commands are O(1) instead of O(100). Decay/buffer logic stays in raw
    # scale; scaling is purely a per-forward-pass input transform.
    command_scale_r: float = 0.02
    command_scale_h: float = 0.01
