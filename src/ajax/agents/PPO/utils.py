"""PPO's old utilities: its names moved to :mod:`ajax.agents.PPO.PPO` and
:mod:`ajax.agents.minibatch`, from which they resolve until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    {"_compute_gae": "ajax.agents.PPO.PPO"}
    | dict.fromkeys(
        (
            "split_fragments",
            "get_minibatches_preserving_time",
            "get_minibatches_from_batch",
        ),
        "ajax.agents.minibatch",
    ),
)
