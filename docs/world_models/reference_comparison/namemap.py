"""Training-metric name map, reference ``train/<name>`` -> Ajax ``Train/<name>``.

PROTOCOL.md section 5.4. Only the loss names differ; every other metric
(optimizer ``opt_*``, ``ent/action/*``, ``rand/action/*``, ``act/action/*``,
``adv/*``, ``rew/*``, ``weight/*``, ``val/*``, ``ret/*``, ``ret_normed/*``,
``replay_ret/*``, ``prior_ent/*``, ``post_ent/*``, ``td_error``,
``ret_rate``, ``data_rew/*``, ``pred_rew/*``, ``activation/embed``) has the
same name on both sides. Reference-only metrics (``rewstats/*``,
``constats/*``, ``opt_param_count``) keep their name and simply have no Ajax
counterpart; ``*/dist`` vectors are dropped before mapping.
"""

REF_TO_AJAX = {
    "vector_loss": "rec_loss",
    "vector_loss_std": "rec_loss_std",
    "reward_loss": "rew_loss",
    "reward_loss_std": "rew_loss_std",
    "cont_loss": "con_loss",
    "cont_loss_std": "con_loss_std",
    "replay_critic_loss": "repval_loss",
    "replay_critic_loss_std": "repval_loss_std",
}
AJAX_TO_REF = {v: k for k, v in REF_TO_AJAX.items()}

# Metrics printed side by side by compare.py, in this order (Ajax names).
COMPARE_KEYS = [
    "rec_loss",
    "rew_loss",
    "con_loss",
    "dyn_loss",
    "rep_loss",
    "actor_loss",
    "critic_loss",
    "repval_loss",
    "ent/action/mean",
    "rand/action/mean",
    "ret/mean",
    "ret/std",
    "ret_normed/std",
    "val/mean",
    "adv/std",
    "replay_ret/mean",
    "weight/mean",
    "rew/mean",
    "prior_ent/mean",
    "post_ent/mean",
    "td_error",
    "data_rew/mean",
    "pred_rew/mean",
    "activation/embed",
    "opt_loss",
    "opt_grad_norm",
    "opt_update_norm",
    "opt_param_norm",
    "opt_grad_steps",
]


def to_ajax(name: str) -> str:
    return REF_TO_AJAX.get(name, name)
