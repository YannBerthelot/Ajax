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
    # When set (>0), use this as the number of minibatches per epoch
    # directly -- decouples num_minibatches from batch_size, matching
    # brax PPO's surface where the two are independent. When 0 (legacy
    # default), the old formula is used:
    #   num_minibatches = max(batch_size, n_steps) // min(batch_size, n_steps)
    # which couples the two and requires ``n_steps % num_minibatches == 0``.
    num_minibatches: int = 0
    # Value-loss coefficient. ``vf_coef * 0.5 * mean((v - target)**2)``
    # is the critic loss; brax PPO defaults vf_coef=0.5 → effective
    # 0.25 on the MSE (``losses.py: v_loss = mean(v_err**2) * 0.5 *
    # vf_coefficient``). Ajax legacy is 1.0 → effective 0.5 on the MSE
    # (twice brax's effective weight). The manip-tuned brax config
    # implicitly assumes vf_coef=0.5; setting it here aligns gradient
    # magnitude on the critic head with brax's update size.
    vf_coef: float = 1.0
    # Use brax PPO's V-trace-style GAE (truncation_mask zeros delta at
    # truncated steps; value target ``vs`` propagated forward through
    # the recursion). When False (legacy default), use standard GAE
    # which contaminates the policy gradient at truncation boundaries
    # by bootstrapping V(reset_obs). On envs with fixed episode
    # length << rollout-collected timesteps (e.g. mujoco_playground
    # manip: 150-step episodes, 20480 transitions/iter), ~1/150 of
    # transitions are truncations -- standard GAE injects noise at
    # those steps that the agent then trains on. See M6 in audit.
    use_vtrace_gae: bool = False
    # Joint global-norm gradient clipping across actor + critic gradients
    # (brax PPO's behavior). Default False = legacy Ajax per-network
    # clipping (each Adam chain has its own clip_by_global_norm on only
    # its own gradients). With this True, before either optimizer
    # applies updates, compute ``sqrt(||a_grads||^2 + ||c_grads||^2)``
    # and scale BOTH gradient pytrees by ``min(1, max_grad_norm /
    # joint_norm)``. Matches brax's effect because their fused Adam
    # uses one ``clip_by_global_norm`` spanning both networks.
    # Note: actor and critic Adam states stay separate -- since Adam's
    # m/v are per-parameter, splitting the optimizer OBJECT across two
    # TrainStates is behaviorally identical to fusing them into one as
    # long as the grads see the same per-param clip. Hence joint-clip
    # alone captures the M5 effect with no state-shape refactor.
    fused_grad_clip: bool = False
