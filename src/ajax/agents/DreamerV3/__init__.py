"""DreamerV3 (Hafner et al., arXiv:2301.04104v2), paper-era recipe.

The world model and the training step as pure, jittable functions:
:mod:`.networks` (block-GRU RSSM, encoder, decoder, heads, actor, critic),
:mod:`.distributions` (straight-through one-hot latents, Bernoulli and
symlog-MSE outputs, the actor's distributions), :mod:`.world_model` (the
world-model loss with the replay context), :mod:`.actor_critic`
(imagination, lambda-returns, actor and critic losses, replay critic),
:mod:`.optim` (LaProp with adaptive gradient clipping and warmup),
:mod:`.learner` (one training step: joint gradient, three optimizers, slow
critic) and :mod:`.state` (static hyperparameters, size presets and the
agent state). The agent: :mod:`.replay` (the stream replay with the replay
context, online queue and latent write-back), :mod:`.train_DreamerV3` (the
acting, update and schedule of its training loop) and
:class:`.DreamerV3.DreamerV3`. See ``docs/world_models/DESIGN.md`` section 6.
"""

from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.DreamerV3.state import MODEL_SIZES, DreamerV3Config

__all__ = ["MODEL_SIZES", "DreamerV3", "DreamerV3Config"]
