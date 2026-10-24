"""The Ajax arms of a reference comparison (``refcmp_specs``).

usage: python refcmp_ajax.py NAME ARM FIRST_SEED N_SEEDS (refcmp_worker)

ARM is ``matched`` or ``defaults``. Every seed trains in one program; the
evaluation is Ajax's own (``n_eval_episodes`` deterministic episodes every
``eval_every`` steps), except for agents whose comparison scores the final
policy only (``eval_every == 0``), which ``refcmp_jax.evaluate_final`` scores.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from refcmp_worker import main

import ajax
from ajax.agents.critics import make_huber_td_loss
from ajax.logging.wandb_logging import LoggingConfig


def _half_squared_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
    """purejaxql's ``0.5 * mean(td**2)``; Ajax's default omits the 0.5."""
    return 0.5 * jnp.mean((q_taken - target_q) ** 2)


_LOSSES = {
    "huber": lambda: make_huber_td_loss(delta=1.0),
    "half_squared": lambda: _half_squared_td_loss,
}
# ("linear", end_value, steps) or ("exponential", rate per step), from the
# spec's learning_rate; Ajax hands a callable rate to optax as a schedule.
_DECAYS = {
    "linear": lambda lr, end, steps: optax.linear_schedule(lr, end, steps),
    "exponential": lambda lr, rate: optax.exponential_decay(lr, 1, rate),
}


def agent_kwargs(spec, arm: str) -> dict:
    """The arm's Ajax keywords, with the spec's tokens (``td_loss``,
    ``lr_decay``) turned into the callables Ajax takes."""
    kwargs = dict(spec.matched if arm == "matched" else spec.defaults)
    if "td_loss" in kwargs:
        kwargs["td_loss_fn"] = _LOSSES[kwargs.pop("td_loss")]()
    if "lr_decay" in kwargs:
        kind, *args = kwargs.pop("lr_decay")
        kwargs["learning_rate"] = _DECAYS[kind](kwargs["learning_rate"], *args)
    return kwargs


def build_agent(spec, arm: str):
    return getattr(ajax, spec.agent)(spec.env_id, **agent_kwargs(spec, arm))


def run(spec, arm: str, seeds: list, n_timesteps: int = 0, eval_every: int = -1):
    """Per-seed evaluation curves ``(seeds, evaluations)``; ``n_timesteps`` and
    ``eval_every`` override the spec's (tests run tiny budgets)."""
    n_timesteps = n_timesteps or spec.n_timesteps
    eval_every = spec.eval_every if eval_every < 0 else eval_every
    agent = build_agent(spec, arm)
    if eval_every == 0:
        from refcmp_jax import evaluate_final

        state, _ = agent.train(seed=seeds, n_timesteps=n_timesteps)
        curve = []
        for i, seed in enumerate(seeds):
            actor = jax.tree.map(lambda x, i=i: x[i], state.actor_state)
            returns = evaluate_final(
                lambda obs, actor=actor: actor.apply(actor.params, obs).mean(),
                spec.env_id,
                jax.random.PRNGKey(10_000 + seed),
                spec.n_eval_episodes,
            )
            curve.append([float(returns.mean())])
        return curve
    logging = LoggingConfig(config={}, log_frequency=eval_every, use_wandb=False)
    _, evals = agent.train(
        seed=seeds,
        n_timesteps=n_timesteps,
        num_episode_test=spec.n_eval_episodes,
        logging_config=logging,
    )
    curve = np.asarray(evals["Eval/episodic mean reward"])
    if np.isnan(curve).any():
        raise RuntimeError("an evaluation slot stayed empty: check eval_every")
    return curve.tolist()


def config(spec, arm: str) -> dict:
    """The agent's run config, its schedule and loss given as the spec's tokens."""
    resolved = build_agent(spec, arm).config
    arm_kwargs = spec.matched if arm == "matched" else spec.defaults
    return {k: v for k, v in resolved.items() if not callable(v)} | arm_kwargs


if __name__ == "__main__":
    main(run, config, ("jax", "jaxlib", "flax", "optax"))
