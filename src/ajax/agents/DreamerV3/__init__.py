"""DreamerV3 (Hafner et al., arXiv:2301.04104v2), paper-era recipe.

Milestone M5 (``docs/world_models/DESIGN.md`` section 11): the world model
as pure, jittable functions -- :mod:`.networks` (block-GRU RSSM, encoder,
decoder, heads), :mod:`.distributions` (straight-through one-hot latents,
Bernoulli and symlog-MSE outputs), :mod:`.world_model` (the world-model
loss with the replay context) and :mod:`.state` (static hyperparameters and
size presets).
"""

from ajax.agents.DreamerV3.state import MODEL_SIZES, DreamerV3Config

__all__ = ["MODEL_SIZES", "DreamerV3Config"]
