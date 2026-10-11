"""SquashedNormal moved to :mod:`ajax.networks.distributions`."""

from ajax.deprecation import moved_names
from ajax.networks import distributions

__getattr__ = moved_names(__name__, SquashedNormal=distributions.SquashedNormal)
