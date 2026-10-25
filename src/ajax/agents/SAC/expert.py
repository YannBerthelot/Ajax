"""SAC's expert prefill moved to ajax.extensions.pretrain, a shared home for
every replay agent. The old path serves it with a FutureWarning until
0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, {"collect_and_store_expert_transitions": "ajax.extensions.pretrain"}
)
