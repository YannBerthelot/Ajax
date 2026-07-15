from typing import Optional

from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState

PPOState = BaseAgentState


@struct.dataclass
class PPOConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float = 0.99
    ent_coef: float = 0.0
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    # When > 0, decouples num_minibatches from batch_size (brax-style).
    num_minibatches: int = 0
    # Critic loss = vf_coef * 0.5 * MSE. Brax PPO uses 0.5; Ajax legacy 1.0.
    vf_coef: float = 1.0
    # Brax's V-trace-style GAE: zero delta at truncation, propagate vs.
    use_vtrace_gae: bool = False
    # Brax-style joint global-norm clip across actor + critic grads.
    fused_grad_clip: bool = False
    # GAE-fragment length for the brax-faithful minibatcher. When set,
    # the (T=n_steps, n_envs) rollout is sub-split into chunks of
    # ``unroll_length`` steps before minibatching, and GAE bootstraps
    # at every fragment boundary (matches brax's per-minibatch
    # (T=unroll_length, B=batch_size) loss shape). When None, GAE uses
    # the full n_steps as the fragment length (one bootstrap at the
    # end of each rollout per env).
    unroll_length: Optional[int] = None
    # Brax's ``num_resets_per_eval``: do a forced env.reset with fresh
    # RNG every ``num_training_steps_per_epoch`` training iterations to
    # cycle through new randomised initial conditions. When 0, only
    # natural auto-resets fire (Ajax default; cached first_obs is
    # reused indefinitely for envs with randomised reset states like
    # PandaOpenCabinet).
    num_resets_per_eval: int = 0
    num_evals: int = 1
