"""DreamerV3 (Hafner et al., arXiv:2301.04104v2), paper-era recipe.

Milestones M5 and M6 (``docs/world_models/DESIGN.md`` section 11): the
world model and the full training step as pure, jittable functions --
:mod:`.networks` (block-GRU RSSM, encoder, decoder, heads, actor, critic),
:mod:`.distributions` (straight-through one-hot latents, Bernoulli and
symlog-MSE outputs, the actor's distributions), :mod:`.world_model` (the
world-model loss with the replay context), :mod:`.actor_critic`
(imagination, lambda-returns, actor and critic losses, replay critic),
:mod:`.optim` (LaProp with adaptive gradient clipping and warmup),
:mod:`.learner` (one training step: joint gradient, three optimizers, slow
critic) and :mod:`.state` (static hyperparameters and size presets).
"""

from ajax.agents.DreamerV3.state import MODEL_SIZES, DreamerV3Config

__all__ = ["MODEL_SIZES", "DreamerV3Config"]
