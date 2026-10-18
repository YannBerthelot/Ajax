from typing import Optional

from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState

PPOState = BaseAgentState


@struct.dataclass
class PPOConfig(BaseAgentConfig):
    """PPO's hyperparameters."""

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
    # Critic loss = vf_coef * 0.5 * MSE: ppo2's vf_coef 0.5 on its
    # 0.5 * MSE (baselines ea25b9e ppo2/ppo2.py:22, model.py:75, :91).
    vf_coef: float = 0.5
    # ppo2's clip: one global norm over both networks' gradients
    # (ppo2/model.py:105-107); None leaves them unclipped.
    max_grad_norm: Optional[float] = 0.5
    # Opts into brax's time geometry: the (T=n_steps, n_envs) rollout is
    # sub-split into chunks of ``unroll_length`` steps before minibatching,
    # and GAE, recomputed per minibatch, bootstraps at every fragment
    # boundary (brax's per-minibatch (T=unroll_length, B=batch_size) loss
    # shape); n_steps keeps whole rollouts per env. When None, GAE runs
    # once on the rollout and its samples are shuffled flat.
    unroll_length: Optional[int] = None
    # Recurrent-only truncated-BPTT length. When set, each env's rollout
    # splits into (n_steps / bptt_length) contiguous sequences whose
    # start carries are recomputed chunk-wise with the current params
    # (never zero-initialized mid-episode — that variant demonstrably
    # hurts). None keeps full-rollout BPTT. Must divide n_steps. Ignored
    # by feedforward policies (unroll_length covers that geometry).
    bptt_length: Optional[int] = None
    # Brax's ``num_resets_per_eval``: do a forced env.reset with fresh
    # RNG every ``num_training_steps_per_epoch`` training iterations to
    # cycle through new randomised initial conditions. When 0 (the
    # default), only natural auto-resets fire. Playground envs already
    # start every episode from a fresh state by default; this only adds
    # diversity with ``build_env_from_id(..., fresh_reset=False)``, whose
    # cached first state is otherwise reused indefinitely.
    num_resets_per_eval: int = 0
    num_evals: int = 1
