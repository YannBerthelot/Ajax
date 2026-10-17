from collections.abc import Sequence
from typing import Callable, Optional, Tuple, Union

from gymnax import EnvParams

from ajax.agents.APO.state import APOConfig
from ajax.agents.APO.train_APO import make_train
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.extensions.base import Extension
from ajax.networks.utils import TORCH_LINEAR
from ajax.types import EnvType, InitializationFunction


class APO(ActorCritic):
    """
    Average-Policy Optimization (APO, Ma et al. 2021) for discrete and
    continuous action spaces.
    See  https://arxiv.org/pdf/2106.03442
    """

    name: str = "APO"
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 4,
        # APO's official code steps the whole model with one Adam at 3e-4,
        # unscheduled, torch's other defaults (xtma/apo @bfcd997,
        # apo/algos/apg/base.py:28; apo/experiments/configs/mujoco/apg/
        # mujoco_appo.py:5, :16). Adam acts per coordinate, so the two
        # networks' Adams, stepped together on one gradient
        # (train_APO.update_networks), are that Adam, each at its rate.
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        # The hidden layers of APO's official code: xtma/apo @bfcd997 sets
        # only normalize_observation in the model config
        # (apo/experiments/configs/mujoco/apg/mujoco_appo.py:20, exp.py:38),
        # so rlpyt's MujocoFfModel (xtma/rlpyt @e28bb20,
        # rlpyt/models/pg/mujoco_ff_model.py:21-22, :33-49) builds its mean
        # and its value MLPs each with two hidden layers of 64 tanh units,
        # torch Linear layers (rlpyt/models/mlp.py:27-31) at torch's default
        # initialisation: "torch_linear" (TorchLinear, networks/utils.py).
        # The output layers are not the reference's: the mean layer's
        # initialiser is below, the value layer's orthogonal(1.0), both
        # biases zero.
        actor_architecture=("64", "tanh", "64", "tanh"),
        critic_architecture=("64", "tanh", "64", "tanh"),
        env_params: Optional[EnvParams] = None,
        # The official code's clip of the gradient's global norm over the
        # whole model, actor and critic at once (apo/algos/apg/appo.py:111,
        # mujoco_appo.py:8): neither network's Adam clips its own.
        max_grad_norm: Optional[float] = 10.0,
        weight_decay: float = 0.0,
        ent_coef: float = 0,
        # The value loss's weight in the summed loss, the official code's
        # value_loss_coeff (appo.py:27, which mujoco_appo.py leaves).
        vf_coef: float = 1.0,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        num_minibatches: int = 0,
        adam_eps: float = 1e-8,
        gae_lambda: float = 0.95,
        alpha: float = 0.1,
        nu: float = 0.1,
        normalize_advantage: bool = True,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[
            Union[str, InitializationFunction]
        ] = TORCH_LINEAR,
        # The (low, high) clip of the continuous actor's log-std. None, no
        # clip, as APO's official code (below).
        log_std_bounds: Optional[Tuple[float, float]] = None,
        # Expert for the CloningPretrain extension and the eval expert-bias
        # metric; online BC is the ImitationLoss extension.
        expert_policy: Optional[Callable] = None,
        # Expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=None,  # the clip is over both networks (above)
            weight_decay=weight_decay,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            actor_bias_init=actor_bias_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            # The continuous head of APO's official code: xtma/apo @bfcd997
            # (exp.py:17, :38) runs rlpyt's MujocoFfAgent, from the authors'
            # fork xtma/rlpyt @e28bb20. The Gaussian's mean is tanh of the
            # mean layer (rlpyt/models/pg/mujoco_ff_model.py:23, :40-41);
            # one learned log-std per action dimension starts at 0 and is
            # not clipped (:24, :50, :78; rlpyt/agents/pg/gaussian.py:34-38
            # sets no min_std, max_std, clip or squash). So the sample is
            # not squashed: its log-probability and the entropy are the
            # Gaussian's, of the raw sample (rlpyt/distributions/
            # gaussian.py:81-97, :108-134, :208). The reference hands that
            # sample to the env (rlpyt/envs/gym.py:66-67); here, as for
            # every agent, agent_action_to_env clips it to [-1, 1] on its
            # way there. Discrete APO ignores the head. Not the reference's:
            # the lecun_uniform mean layer (torch's default Linear init
            # there), whose moderate initial actions keep evaluation close
            # to training (orthogonal 0.01 makes the deterministic action
            # essentially zero). As the reference's MlpModel
            # (rlpyt/models/mlp.py:27-35), the encoder has no output
            # LayerNorm (the default).
            squash=False,
            tanh_mean=True,
            log_std_state_independent=True,
            log_std_init=0.0,
            log_std_bounds=log_std_bounds,
            mean_kernel_init="lecun_uniform",
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )

        self.agent_config: APOConfig = APOConfig(
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            num_minibatches=num_minibatches,
            alpha=alpha,
            nu=nu,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            expose_recent_rollout=expose_recent_rollout,
        )
        if num_minibatches <= 0 and max(n_steps, batch_size) % min(n_steps, batch_size):
            raise ValueError(
                f"n_steps ({n_steps}) and batch_size ({batch_size}) must divide"
                " one another: the rollout splits into max // min minibatches"
                " (or set num_minibatches)."
            )

        # Adam's eps: 1e-8, torch's default the official code keeps, not
        # ActorCritic's 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=adam_eps)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=adam_eps)
        self.train_kwargs = {
            "expert_policy": self.expert_policy,
        }
