# luchris429/purejaxrl (vendored, unmodified subset)

`purejaxrl/ppo_rnn.py` and the `wrappers.py` it imports are copied unchanged
from [luchris429/purejaxrl](https://github.com/luchris429/purejaxrl) at commit
`31756b1` (2024-09-09), the reference code of Ajax's recurrent PPO. Apache
License 2.0, see `LICENSE`.

`tests/conformance/test_purejaxrl_rnn.py` checks Ajax's recurrent networks
against `ActorCriticRNN` (`ppo_rnn.py:46-79`) parameter for parameter. The file
is not run: its `nn.GRUCell()` takes no `features`, which flax 0.12 (Ajax's)
requires, so running it needs an environment with an older flax of its own.
