from collections.abc import Callable

import optax

# ----------------------------
# Optimizer step schedules
# ----------------------------


def warmup_cosine_schedule(
    peak_value: float,
    warmup_steps: int,
    total_steps: int,
    end_value_fraction: float = 0.1,
    init_value_fraction: float = 0.0,
) -> Callable[[int], float]:
    """Linear warmup to ``peak_value`` then cosine decay to ``end_value``.

    An optax step schedule (``step -> learning_rate``) usable as
    ``OptimizerConfig.learning_rate``. The warmup-cosine profile is the
    one the in-context controller reference code trains with (Busetto et
    al. 2024, GPT-style: 5k warmup steps, decay to ``peak / 10``).

    ``warmup_steps=0`` gives pure cosine decay from ``peak_value``.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        # optax needs at least one decay step after the warmup.
        raise ValueError(
            f"need 0 <= warmup_steps < total_steps, got {warmup_steps} and {total_steps}"
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=peak_value * init_value_fraction,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=peak_value * end_value_fraction,
    )
