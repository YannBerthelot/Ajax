"""One answer layer: estimators' fixed points on small MDPs, right and under
named faults (``Rule``), and closed forms. Transitions are ``FIELDS`` arrays
(B, T), a weighted rollout per row: ``rollouts``, ``stack``, ``simulate``."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy import optimize

FIELDS = ("obs", "reward", "final", "reset", "terminated", "truncated")


@dataclasses.dataclass(frozen=True)
class Rule:
    """A learner's target; the defaults are right. ``mask``: flags zeroing
    the bootstrap ("or"; "precedence": terminated, not truncated; "none").
    ``next``: the "reset" obs after an end, or the transition's ("self").
    ``rollout_end``: the last bootstrap "terminal" or "zero"; ``cut``: an end
    stops the trace; ``peng``: it corrects the bootstrap (Q(lambda)), not the
    next obs (GAE); ``anchor``: subtracted by every bootstrap (ASAC), masked
    with it unless ``shift_outside``; ``off_by_one``: step T - 2 reads step
    T - 1's bootstrap (purejaxql's PQN); ``drop_truncated``: a time-limited
    row is not fitted (xtma/apo's bootstrap_timelimit)."""

    mask: str = "terminated"
    next: str = "final"
    swap: bool = False
    rollout_end: str = "bootstrap"
    forward: bool = False
    vtrace: bool = False
    cut: bool = True
    peng: bool = False
    anchor: int | None = None
    shift_outside: bool = False
    off_by_one: bool = False
    drop_truncated: bool = False


RIGHT = Rule()


def targets(ro: dict, v: Any, gamma: float, lam: float, rule: Rule) -> np.ndarray:
    """Targets in advantage form, G_t = r_t + gamma b_t + gamma lam (G_u -
    V(s_u)), u the step the scan came from; the trace stops at the edge."""
    term, trunc = ro["terminated"].astype(bool), ro["truncated"].astype(bool)
    term, trunc = (trunc, term) if rule.swap else (term, trunc)
    done, end = term | trunc, np.arange(term.shape[1]) == term.shape[1] - 1
    masks = {"terminated": term, "or": done, "precedence": term & ~trunc}
    masked = ({**masks, "none": np.zeros_like(done)})[rule.mask]
    masked = masked | end & (rule.rollout_end == "zero")
    masked = masked | end & ~done & (rule.rollout_end == "terminal")
    after_end = np.where(done, ro["reset"], ro["final"])
    nxt = {"final": ro["final"], "reset": after_end, "self": ro["obs"]}[rule.next]
    shift = 0.0 if rule.anchor is None else v[rule.anchor]
    inside, outside = (0.0, shift) if rule.shift_outside else (shift, 0.0)
    out, u = np.zeros(term.shape), None
    for t in range(term.shape[1]) if rule.forward else range(term.shape[1] - 1, -1, -1):
        slip = rule.off_by_one and t == term.shape[1] - 2
        own = np.where(masked[:, -1], 0.0, v[nxt[:, -1]]) if slip else v[nxt[:, t]]
        boot, trace = own, np.zeros(())
        if u is not None:  # the trace goes on from step u
            go = ~done[:, t] if rule.cut else True
            boot = np.where(go, out[:, u], boot) if rule.vtrace else boot
            corrected = own if rule.peng or slip else v[ro["obs"][:, u]]
            trace = np.where(go, out[:, u] - corrected, 0.0)
        boot = np.where(masked[:, t], 0.0, boot - inside) - outside
        out[:, t] = ro["reward"][:, t] + gamma * boot + gamma * lam * trace
        u = t
    return out


def fixed_point(
    ro: dict, n: int, gamma: float, lam=0.0, rule=RIGHT, centred=False, bias=0.0
) -> np.ndarray:
    """V equal to each state's weighted mean target, solved exactly (targets
    are affine in V); ``centred``: a differential estimator's, mean 0;
    ``bias``: less that times the visited mean V (APO's nu b)."""
    k, w = ro["obs"].ravel(), np.repeat(ro["weight"], ro["obs"].shape[1])
    fit = w * (1 - ro["truncated"].ravel()) if rule.drop_truncated else w
    visits = np.bincount(k, w, n)

    def mean_target(v: np.ndarray) -> np.ndarray:
        g = targets(ro, v, gamma, lam, rule).ravel()
        mean = np.bincount(k, fit * g, n) / np.bincount(k, fit, n)
        return mean - bias * visits @ v / visits.sum()

    b = mean_target(np.zeros(n))
    m = np.eye(n) - np.stack([mean_target(e) - b for e in np.eye(n)], 1)
    if centred:  # the average reward c is free: (I - A) V + c = b, mean(V) = 0
        m, b = np.block([[m, np.ones((n, 1))], [np.ones((1, n)), 0]]), np.append(b, 0)
    return np.linalg.solve(m, b)[:n]


def stack(streams: Sequence) -> dict:
    """Weighted rollouts, (weight, rows), of equal length as arrays."""
    cols = np.array([list(zip(*r, strict=False)) for _, r in streams])  # (B, 6, T)
    ro = {f: cols[:, i].astype(float if i == 1 else int) for i, f in enumerate(FIELDS)}
    return {**ro, "weight": np.array([w for w, _ in streams], float)}


def rollouts(episodes: Sequence, n_steps: int) -> dict:
    """Every rollout of ``n_steps`` a stationary stream of these weighted,
    equal-length episodes produces: each phase and sequence, weighted."""
    length, out = len(episodes[0][1]), []
    for phase in range(length):
        for idx in np.ndindex(*(len(episodes),) * (n_steps // length + 2)):
            rows = [r for i in idx for r in episodes[i][1]][phase:][:n_steps]
            out.append((float(np.prod([episodes[i][0] for i in idx])) / length, rows))
    return stack(out)


def simulate(step: Callable[[Any], tuple], state: Any, n_steps: int) -> dict:
    """(steps - 1, envs) columns of ``step(state) -> (state, obs, reward,
    terminated, final obs)``; a row's reset obs is the next row's."""
    rows = []
    for _ in range(n_steps):
        state, *row = step(state)
        rows.append(row)
    obs, reward, term, final = (np.array(c) for c in zip(*rows, strict=False))
    cols = {"obs": obs[:-1], "reward": reward[:-1], "final": final[:-1]}
    cols.update({"reset": obs[1:], "terminated": term[:-1]})
    return {**cols, "truncated": np.zeros_like(term[:-1])}


def blocks(cols: dict, length: int) -> dict:
    """Consecutive rollouts of ``length`` steps of every env's column."""
    n = cols["obs"].shape[0] // length * length
    split = {k: v[:n].reshape(-1, length, v.shape[1]) for k, v in cols.items()}
    ro = {k: np.moveaxis(v, 2, 1).reshape(-1, length) for k, v in split.items()}
    return {**ro, "weight": np.ones(len(ro["obs"]))}


_X, _W = np.polynomial.hermite_e.hermegauss(200)  # quadrature for the closed forms
_W = _W / _W.sum()


def softmax_optimum(c: float) -> float:
    """pi(best) maximising E[r] + c H for a two-armed bandit paying 1 / 0."""
    return 1.0 / (1.0 + math.exp(-1.0 / c))


def tanh_gaussian_entropy(sigma: float, mu: float = 0.0) -> float:
    """Entropy (nats) of tanh(u), u ~ N(mu, sigma^2)."""
    u = mu + sigma * _X
    log_det = -2.0 * (u + np.logaddexp(0.0, -2.0 * u) - math.log(2.0))
    return float(0.5 * math.log(2 * math.pi * math.e * sigma**2) + _W @ log_det)


def max_entropy_sigma() -> float:
    """The sigma maximising the entropy of tanh(u), mu = 0: 0.8744."""
    res = optimize.minimize_scalar(lambda s: -tanh_gaussian_entropy(math.exp(s)))
    return math.exp(res.x)


def maxent_policy(reward: Callable, alpha: float) -> tuple[float, float]:
    """tanh(mu) and the entropy of the tanh-Gaussian maximising E[reward(a)]
    + alpha H, ``reward`` a function of the action in [-1, 1]."""

    def loss(p: np.ndarray) -> float:
        value = _W @ reward(np.tanh(p[0] + math.exp(p[1]) * _X))
        return -(value + alpha * tanh_gaussian_entropy(math.exp(p[1]), p[0]))

    options = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}
    res = optimize.minimize(loss, np.zeros(2), method="Nelder-Mead", options=options)
    return float(np.tanh(res.x[0])), tanh_gaussian_entropy(math.exp(res.x[1]), res.x[0])


def maxent_mean_action(slope: float, alpha: float) -> float:
    """A max-entropy actor's tanh(mu) when its critic pays ``slope`` an action."""
    return maxent_policy(lambda a: slope * a, alpha)[0]


def shared_temperature(rewards: Sequence[Callable], target: float) -> float:
    """The temperature at which the max-entropy policies of equally frequent
    contexts paying ``rewards`` average ``target`` nats: an auto-tuned
    temperature's fixed point, one alpha for every state."""

    def gap(log_alpha: float) -> float:
        h = [maxent_policy(r, math.exp(log_alpha))[1] for r in rewards]
        return float(np.mean(h)) - target

    return math.exp(optimize.brentq(gap, math.log(1e-4), math.log(10.0), xtol=1e-10))


def clipped_square_mean(scale: float, clip: float) -> float:
    """E[clip(scale e, -clip, clip)^2] for e ~ N(0, 1)."""
    a, inside = clip / scale, math.erf(clip / scale / math.sqrt(2.0))
    phi = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    return scale**2 * (inside - 2.0 * a * phi) + clip**2 * (1.0 - inside)


def ema_step_rms(weight: float, sd: float) -> float:
    """Root-mean-square step of a moving average of independent draws."""
    return weight * sd * math.sqrt(2.0 / (2.0 - weight))
