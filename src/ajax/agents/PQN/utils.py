"""PQN's old utilities: compute_q_lambda_targets moved to
:mod:`ajax.agents.PQN.PQN`, from which it resolves until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(__name__, {"compute_q_lambda_targets": "ajax.agents.PQN.PQN"})
