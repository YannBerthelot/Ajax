"""An encoder the actor and the critic share (NetworkConfig.share_encoder):
the critic owns it, and the actor holds a copy to act with
(LoadedTrainState.shared_params). These move it between the two.
"""

from typing import Any

from flax.core import FrozenDict

from ajax.networks.utils import SHARED_ENCODER_MODULES
from ajax.state import LoadedTrainState


def shared_encoder_params(critic_params: Any) -> FrozenDict:
    """The parameters of the encoder (and memory block) a critic shares with
    its actor, taken from the critic's parameters."""
    owned = critic_params["params"]
    return FrozenDict(
        {name: owned[name] for name in SHARED_ENCODER_MODULES if name in owned}
    )


def with_critic_encoder(
    actor_state: LoadedTrainState, critic_params: Any
) -> LoadedTrainState:
    """``actor_state`` reading the encoder in ``critic_params``, for an actor
    sharing the critic's encoder; an actor with its own is returned as is. A
    loss differentiated with respect to ``critic_params`` then trains the
    encoder through the actor too."""
    if actor_state.shared_params is None:
        return actor_state
    return actor_state.replace(shared_params=shared_encoder_params(critic_params))


def mirror_shared_encoder(agent_state: Any) -> Any:
    """``agent_state`` with its actor's copy of the shared encoder taken from
    the critic as it is now: the training loop calls it after anything that
    steps the critic (each update, the extensions' pretraining), so acting
    and evaluating read the current encoder. Unchanged without one."""
    actor_state: Any = getattr(agent_state, "actor_state", None)
    if getattr(actor_state, "shared_params", None) is None:
        return agent_state
    critic_params = agent_state.critic_state.params
    return agent_state.replace(
        actor_state=with_critic_encoder(actor_state, critic_params)
    )


def take_actor_encoder(agent_state: Any) -> Any:
    """``agent_state`` with the critic's shared encoder replaced by the
    actor's copy, after something trained the encoder through the actor
    alone (cloning pretraining); the target critic's too, which starts as
    the critic. Unchanged without a shared encoder."""
    actor_state = agent_state.actor_state
    if actor_state.shared_params is None:
        return agent_state
    critic_state = agent_state.critic_state

    def with_encoder(params: Any) -> FrozenDict:
        owned = {**params["params"], **actor_state.shared_params}
        return FrozenDict({**params, "params": owned})

    critic_state = critic_state.replace(
        params=with_encoder(critic_state.params),
        target_params=with_encoder(critic_state.target_params),
    )
    return agent_state.replace(critic_state=critic_state)


def _borrow_encoder(
    actor_state: LoadedTrainState, encoder: FrozenDict
) -> LoadedTrainState:
    """``actor_state`` without its own encoder, reading ``encoder`` instead:
    its parameters, and its optimiser's state, rebuilt on what it still
    owns (its heads)."""
    own = {
        name: value
        for name, value in actor_state.params["params"].items()
        if name not in encoder
    }
    params: FrozenDict = FrozenDict({**actor_state.params, "params": own})
    return LoadedTrainState.create(
        params=params,
        tx=actor_state.tx,
        apply_fn=actor_state.apply_fn,
        hidden_state=actor_state.hidden_state,
        target_params=params,
        batch_stats=actor_state.batch_stats,
    ).replace(shared_params=encoder)
