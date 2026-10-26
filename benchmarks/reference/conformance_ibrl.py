"""hengyuan-hu/ibrl's critic target and acting, recorded for the IBRL kernel check (tests/conformance/test_ibrl_original.py).

usage, from the repository root:
    uv run --project benchmarks/reference/envs/torch --locked python benchmarks/reference/conformance_ibrl.py CLONE OUT_DIR

CLONE is a checkout of hengyuan-hu/ibrl at COMMIT. The repository has no licence, so it is not vendored:
its records are committed instead (tests/conformance/records). Imports nothing from Ajax. Their QAgent
runs its update_critic unchanged, in state mode (MultiFcQ critic, FcActor) with bootstrap_method "ibrl",
on a fixed random batch. Recorder seams: stand-ins for their common_utils package and networks.encoder
(which import cv2, matplotlib and a compiled extension the target never reads), a tanh-linear BC policy
(theirs reads images), dropout 0 in the target actor (UPSTREAM_BUGS 2, left out), the target critics'
biases drawn (theirs start at zero), and nn.functional.mse_loss wrapped to keep the target it is handed.
Each of FORMS runs in float32 with its draws kept (the target actor's standard normal noise, then each
np.random.choice of forward_k: the selection pair, then the value pair), then in float64 with the same
draws. One .npz per form: in.* (the target critics), batch.*, draw.*, and f32.* and f64.* (the target,
the sampled RL action and the BC action at s'). Their _act_ibrl then runs on a fixed batch of states, the
agent in eval mode as their act asserts, eps_greedy 1 (every IBRL config), in each form and mode
(MODES: eval_mode True, the actor's mean; False, its sample, the noise kept as above): act_<form>.npz
holds in.*, batch.state, <mode>.draw.* and <mode>.f32.* / <mode>.f64.* (the executed action, the RL
action, the actor's mean and the BC action).
"""

import copy
import functools
import importlib.util
import json
import pathlib
import platform
import sys
import types

import numpy as np
import torch

COMMIT = "3c457da"
SEED, BATCH, OBS, ACTION, WIDTH = 1, 64, 2, 1, 16
ACT_SEED = 3  # the acting records'
GAMMA, STDDEV = 0.99, 0.1  # every IBRL config's acting stddev, the target actor's too
#: num_q critics, of which forward_k draws num_k: the pixel critic's two heads, and the state critic's
#: random pair (twice per update: selection, then value).
FORMS = {"pair": (2, 2), "state": (5, 2)}
MODES = {"eval": True, "train": False}


def print_nothing(module):
    del module


def build_no_augmentation(pad):
    del pad
    return torch.nn.Identity()


def install_stand_ins(clone: pathlib.Path) -> types.ModuleType:
    """Their ibrl_utils, loaded from its file, inside a common_utils holding the few helpers QAgent
    calls; networks.encoder's names, never built in state mode."""
    sys.modules["torchvision"] = types.ModuleType("torchvision")
    sys.modules["torchvision.transforms"] = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
    path = clone / "common_utils" / "py" / "ibrl_utils.py"
    spec = importlib.util.spec_from_file_location("common_utils.ibrl_utils", path)
    ibrl_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ibrl_utils)
    common = types.ModuleType("common_utils")
    common.ibrl_utils = ibrl_utils
    common.wrap_ruler = str
    common.count_parameters = print_nothing
    common.RandomShiftsAug = build_no_augmentation
    sys.modules["common_utils"] = common
    sys.modules["common_utils.ibrl_utils"] = ibrl_utils
    encoder = types.ModuleType("networks.encoder")
    for name in (
        "VitEncoder",
        "VitEncoderConfig",
        "ResNetEncoder",
        "ResNetEncoderConfig",
        "DrQEncoder",
        "ResNet96Encoder",
        "ResNet96EncoderConfig",
    ):
        setattr(encoder, name, type(name, (), {}))
    sys.modules["networks"] = types.ModuleType("networks")
    sys.modules["networks.encoder"] = encoder
    sys.path.insert(0, str(clone))
    return ibrl_utils


class LinearExpert(torch.nn.Module):
    """A deterministic BC stand-in: tanh(W s + b), act(obs, cpu) as their BcPolicy."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(OBS, ACTION)

    def act(self, obs, cpu=False):
        del cpu
        return torch.tanh(self.layer(obs["state"]))


def update(agent, batch: dict, tensor) -> None:
    """Their update_critic on batch, which hands mse_loss the target."""
    agent.update_critic(
        obs={"state": tensor(batch["state"])},
        reply={"action": tensor(batch["action"])},
        reward=tensor(batch["reward"]),
        discount=tensor(GAMMA * (1.0 - batch["dones"])),
        next_obs={"state": tensor(batch["next_state"])},
        stddev=STDDEV,
    )
    return None


def act(agent, batch: dict, tensor, eval_mode: bool) -> dict:
    """Their _act_ibrl on batch's states in eval mode (as their act asserts): the executed action and
    the actor's mean."""
    from common_utils import ibrl_utils

    obs = {"state": tensor(batch["state"])}
    with torch.no_grad(), ibrl_utils.eval_mode(agent):
        action = agent._act_ibrl(
            obs=obs,
            eval_mode=eval_mode,
            stddev=STDDEV,
            clip=None,
            eps_greedy=1,
            use_target=False,
        )
        mean = agent.actor(obs, STDDEV).mean
    return {"action": action.numpy().copy(), "mean": mean.numpy().copy()}


def run(agent, expert, batch: dict, draws: dict, dtype, call=update) -> dict:
    """call(agent, batch, tensor) in dtype (their update_critic by default); draws holds the noise and
    pairs to replay (empty: draw them and keep them in it). Returns what call returns, the target where
    one was handed to mse_loss, the sampled RL action and the BC action."""
    from common_utils import ibrl_utils
    from rl import critic as critic_module

    agent, expert = copy.deepcopy(agent).to(dtype), copy.deepcopy(expert).to(dtype)
    agent.bc_policies = []
    agent.add_bc_policy(expert)
    kept, pairs = {}, list(draws.get("pairs", []))
    replay = bool(draws)
    standard_normal = ibrl_utils._standard_normal
    choice = np.random.choice
    mse_loss = torch.nn.functional.mse_loss
    sample = ibrl_utils.TruncatedNormal.sample
    act = LinearExpert.act

    def draw_noise(shape, dtype, device):
        if replay:
            return torch.as_tensor(draws["noise"], dtype=dtype)
        value = standard_normal(shape, dtype=dtype, device=device)
        draws["noise"] = value.numpy().copy()
        return value

    def draw_pair(n, k, replace=True):
        if replay:
            return pairs.pop(0)
        value = choice(n, k, replace=replace)
        draws.setdefault("pairs", []).append(np.asarray(value))
        return value

    def keep_target(prediction, target, reduction="mean"):
        kept["target"] = target[:, 0].detach().numpy().copy()
        return mse_loss(prediction, target, reduction=reduction)

    def keep_rl_action(self, clip=None):
        value = sample(self, clip)
        kept["rl_action"] = value.detach().numpy().copy()
        return value

    def keep_bc_action(self, obs, cpu=False):
        value = act(self, obs, cpu)
        kept["bc_action"] = value.detach().numpy().copy()
        return value

    ibrl_utils._standard_normal = draw_noise
    critic_module.np.random.choice = draw_pair
    torch.nn.functional.mse_loss = keep_target
    ibrl_utils.TruncatedNormal.sample = keep_rl_action
    LinearExpert.act = keep_bc_action
    try:

        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype)

        kept.update(call(agent, batch, tensor) or {})
    finally:
        ibrl_utils._standard_normal = standard_normal
        critic_module.np.random.choice = choice
        torch.nn.functional.mse_loss = mse_loss
        ibrl_utils.TruncatedNormal.sample = sample
        LinearExpert.act = act
    assert not pairs, "a replayed pair was not drawn"
    return kept


def record(form: str, out: pathlib.Path) -> None:
    from rl.actor import FcActorConfig
    from rl.critic import MultiFcQConfig
    from rl.q_agent import QAgent, QAgentConfig

    num_q, num_k = FORMS[form]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = QAgentConfig(
        device="cpu",
        act_method="ibrl",
        state_critic=MultiFcQConfig(
            num_q=num_q, num_k=num_k, num_layer=2, hidden_dim=WIDTH
        ),
        state_actor=FcActorConfig(num_layer=2, hidden_dim=WIDTH, dropout=0.0),
    )
    agent = QAgent(True, (OBS,), (OBS,), ACTION, "", cfg)
    # _MultiLinear starts its one bias per net at zero: drawn here, so the check reads them.
    with torch.no_grad():
        for name, parameter in agent.critic_target.named_parameters():
            if name.endswith("biases"):
                parameter.uniform_(-0.5, 0.5)
    expert = LinearExpert()
    rng = np.random.default_rng(SEED)
    batch = {
        "state": rng.normal(size=(BATCH, OBS)),
        "action": rng.uniform(-1.0, 1.0, (BATCH, ACTION)),
        "reward": rng.normal(size=BATCH),
        "dones": (np.arange(BATCH) % 8 == 7).astype(np.float64),
        "next_state": rng.normal(size=(BATCH, OBS)),
    }
    draws: dict = {}
    f32 = run(agent, expert, batch, draws, torch.float32)
    f64 = run(agent, expert, batch, draws, torch.float64)
    assert np.all(np.abs(f32["rl_action"]) < 1.0 - 1e-6), "the +-(1 - 1e-6) clamp binds"
    arrays = {
        f"in.{k}": v.detach().numpy() for k, v in agent.critic_target.named_parameters()
    }
    arrays.update({f"batch.{k}": np.asarray(v) for k, v in batch.items()})
    arrays.update({"batch.gamma": np.asarray(GAMMA), "draw.noise": draws["noise"]})
    for name, pair in zip(("selection", "value"), draws.get("pairs", []), strict=False):
        arrays[f"draw.{name}"] = pair
    if form == "state":
        selection, value = arrays["draw.selection"], arrays["draw.value"]
        assert set(selection) != set(value), "one pair: reseed"
        with torch.no_grad():
            s = torch.as_tensor(batch["next_state"], dtype=torch.float32)
            q_rl = agent.critic_target(s, torch.as_tensor(f32["rl_action"]))
            q_bc = agent.critic_target(s, torch.as_tensor(f32["bc_action"]))

        def expert_wins(pair):
            return (q_bc[:, pair].min(-1)[0] > q_rl[:, pair].min(-1)[0]).numpy()

        assert np.any(expert_wins(selection) != expert_wins(value)), (
            "pairs agree: reseed"
        )
    arrays.update({f"f32.{k}": v for k, v in f32.items()})
    arrays.update({f"f64.{k}": v for k, v in f64.items()})
    np.savez(out / f"{form}.npz", **arrays)


def record_acting(form: str, out: pathlib.Path) -> None:
    """Their _act_ibrl in both modes on one form's agent, the target critics' biases drawn."""
    from rl.actor import FcActorConfig
    from rl.critic import MultiFcQConfig
    from rl.q_agent import QAgent, QAgentConfig

    num_q, num_k = FORMS[form]
    torch.manual_seed(ACT_SEED)
    np.random.seed(ACT_SEED)
    cfg = QAgentConfig(
        device="cpu",
        act_method="ibrl",
        state_critic=MultiFcQConfig(
            num_q=num_q, num_k=num_k, num_layer=2, hidden_dim=WIDTH
        ),
        state_actor=FcActorConfig(num_layer=2, hidden_dim=WIDTH, dropout=0.0),
    )
    agent = QAgent(True, (OBS,), (OBS,), ACTION, "", cfg)
    with torch.no_grad():
        for name, parameter in agent.critic_target.named_parameters():
            if name.endswith("biases"):
                parameter.uniform_(-0.5, 0.5)
    expert = LinearExpert()
    batch = {"state": np.random.default_rng(ACT_SEED).normal(size=(BATCH, OBS))}
    arrays = {
        f"in.{k}": v.detach().numpy() for k, v in agent.critic_target.named_parameters()
    }
    arrays["batch.state"] = batch["state"]
    for mode, eval_mode in MODES.items():
        draws: dict = {}
        call = functools.partial(act, eval_mode=eval_mode)
        f32 = run(agent, expert, batch, draws, torch.float32, call)
        f64 = run(agent, expert, batch, draws, torch.float64, call)
        rl = f32.get("rl_action", f32["mean"])
        assert np.all(np.abs(rl) < 1.0 - 1e-6), "the +-(1 - 1e-6) clamp binds"
        expert_won = f32["action"] == f32["bc_action"]
        assert expert_won.any() and not expert_won.all(), (
            "one candidate always wins: reseed"
        )
        if form == "state":
            with torch.no_grad():
                s = torch.as_tensor(batch["state"], dtype=torch.float32)
                q_rl = agent.critic_target(s, torch.as_tensor(rl))
                q_bc = agent.critic_target(s, torch.as_tensor(f32["bc_action"]))
            every = (q_bc.min(-1)[0] > q_rl.min(-1)[0]).numpy()
            assert np.any(every != expert_won[:, 0]), (
                "the pair agrees with every critic: reseed"
            )
        if "noise" in draws:
            arrays[f"{mode}.draw.noise"] = draws["noise"]
        for pair in draws.get("pairs", []):
            arrays[f"{mode}.draw.selection"] = pair
        arrays.update({f"{mode}.f32.{k}": v for k, v in f32.items()})
        arrays.update({f"{mode}.f64.{k}": v for k, v in f64.items()})
    np.savez(out / f"act_{form}.npz", **arrays)


def main(clone: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    install_stand_ins(pathlib.Path(clone))
    for form in FORMS:
        record(form, out)
        record_acting(form, out)
    versions = {"torch": torch.__version__, "numpy": np.__version__}
    versions.update(python=platform.python_version(), hengyuan_hu_ibrl=COMMIT)
    (out / "versions.json").write_text(json.dumps(versions))


if __name__ == "__main__":
    main(*sys.argv[1:3])
