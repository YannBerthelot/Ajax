"""AVG's TD-error scale, now in :mod:`ajax.agents.AVG.AVG`; the old path
imports it with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(__name__, {"compute_td_error_scaling": "ajax.agents.AVG.AVG"})
