"""CQL's update (aviralkumar2907/CQL's CQLTrainer, on rlkit), recorded for the conformance check of
Ajax's CQL.

usage: python conformance_cql.py CLONE ENV_ID OUT_DIR   (in envs/torch14: torch 1.4 on x86 Python 3.8)

CLONE is a checkout of aviralkumar2907/CQL at COMMIT. The repository has no licence, so it is not
vendored: its records are committed instead (tests/conformance/records). Imports nothing from Ajax.
The dataset is conformance_d4rl's, turned into rows by D4RL's qlearning_dataset and loaded as
examples/cql_mujoco_new.py's load_hdf5 does. The networks and the trainer are that script's
experiment(): four separately initialised critics (the targets are not copies), a TanhGaussianPolicy,
the README's gym-MuJoCo settings (TRAINER), WIDTH-wide networks (theirs 3 x 256) and batch BATCH so
that the records stay small. Training is rlkit's offline loop: random_batch, then trainer.train. The
trainer's update runs unchanged, on torch 1.4: its policy's backward pass runs after the critics'
optimizer steps and reads their stepped weights, as on the torch the code was written for.

Before the train calls in RECORDS, the call runs on a copy with every draw recorded (each rsample's
standard normal noise, and the uniform random actions), and again in float64 with the same draws. Two
recorder seams, for the float64 run only: the batch is handed to train_from_torch directly (rlkit's
np_to_pytorch_batch casts to float32), and torch.FloatTensor, with which cql.py builds the uniform
actions' tensor, makes a float64 one. One .npz per call: ``in.*``, ``batch.*``,
``draw.<i>.<normal|uniform>``, ``f32.*`` and ``f64.*``.
"""

import copy
import pathlib
import sys
import types

import numpy as np
import torch
from conformance_d4rl import dataset, qlearning_dataset
from conformance_old_torch import write_versions

COMMIT = "d67dbe9"
RECORDS = (
    1,
    40_000,
)  # the policy's cloning phase, and its first SAC step (policy_eval_start)
WIDTH, BATCH, SEED = 32, 64, 0
#: cql_mujoco_new.py's trainer_kwargs, with the README's gym-MuJoCo flags (--min_q_weight 5.0
#: --lagrange_thresh -1.0 --policy_lr 1e-4 --min_q_version 3).
TRAINER = {
    "discount": 0.99,
    "soft_target_tau": 5e-3,
    "policy_lr": 1e-4,
    "qf_lr": 3e-4,
    "reward_scale": 1,
    "use_automatic_entropy_tuning": True,
    "policy_eval_start": 40_000,
    "num_qs": 2,
    "temp": 1.0,
    "min_q_version": 3,
    "min_q_weight": 5.0,
    "with_lagrange": False,
    "lagrange_thresh": -1.0,
    "num_random": 10,
    "max_q_backup": False,
    "deterministic_backup": True,
}


def named(trainer) -> dict:
    """Every trained tensor by name."""
    out = {"log_alpha": trainer.log_alpha}
    for owner in ("policy", "qf1", "qf2", "target_qf1", "target_qf2"):
        for name, p in getattr(trainer, owner).named_parameters():
            out[f"{owner}.{name}"] = p
    return out


def state(trainer) -> dict:
    """The networks, log_alpha, and the Adams of the policy, both critics and log_alpha."""
    opts = {
        "policy": trainer.policy_optimizer,
        "qf1": trainer.qf1_optimizer,
        "qf2": trainer.qf2_optimizer,
        "log_alpha": trainer.alpha_optimizer,
    }
    out = {}
    for name, p in named(trainer).items():
        out[name] = p.detach().numpy().copy()
        opt = opts.get(name.split(".")[0])
        if opt is None:  # a target network
            continue
        st = opt.state.get(p, {})
        zeros = np.zeros(tuple(p.shape), p.detach().numpy().dtype)
        for moment in ("exp_avg", "exp_avg_sq"):
            out[f"opt.{name}.{moment}"] = (
                st[moment].detach().numpy().copy() if st else zeros
            )
        out[f"opt.{name}.step"] = np.asarray(
            float(st["step"]) if st else 0.0, np.float32
        )
    return out


def one_call(trainer, np_batch: dict, tape: list, double: bool) -> dict:
    """``trainer.train`` once on a copy: recording its draws into an empty ``tape``, or replaying a
    filled one (in float64 with ``double``)."""
    from rlkit.torch import distributions

    t = copy.deepcopy(trainer)
    replay, drawn = bool(tape), iter(list(tape))
    dtype = torch.float64 if double else torch.float32
    if double:
        for owner in ("policy", "qf1", "qf2", "target_qf1", "target_qf2"):
            getattr(t, owner).double()
        with torch.no_grad():
            t.log_alpha.data = t.log_alpha.data.double()
        for opt in (
            t.policy_optimizer,
            t.qf1_optimizer,
            t.qf2_optimizer,
            t.alpha_optimizer,
        ):
            for st in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()

    def taken(kind: str, fresh):
        if replay:
            got, value = next(drawn)
            assert got == kind, (got, kind)
            return torch.as_tensor(value, dtype=dtype)
        value = fresh()
        tape.append((kind, value.numpy().copy()))
        return value

    def rsample(
        self, return_pretanh_value=False
    ):  # distributions.py:60-75, its noise taped
        eps = taken("normal", lambda: torch.randn(self.normal_mean.size()))
        z = self.normal_mean + self.normal_std * eps
        z.requires_grad_()
        return (torch.tanh(z), z) if return_pretanh_value else torch.tanh(z)

    originals = distributions.TanhNormal.rsample, torch.FloatTensor

    def float_tensor(*size):  # cql.py:234's uniform actions, taped
        holder = types.SimpleNamespace()
        holder.uniform_ = lambda low, high: taken(
            "uniform", lambda: originals[1](*size).uniform_(low, high)
        )
        return holder

    batch = {k: torch.as_tensor(v, dtype=dtype) for k, v in np_batch.items()}
    distributions.TanhNormal.rsample, torch.FloatTensor = rsample, float_tensor
    try:
        t._num_train_steps += 1  # TorchTrainer.train, without its float32 cast
        t.train_from_torch(batch)
    finally:
        distributions.TanhNormal.rsample, torch.FloatTensor = originals
    return state(t)


def main(clone: pathlib.Path, env_id: str, out: pathlib.Path) -> None:
    sys.path.insert(0, str(clone / "d4rl"))
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.cql import CQLTrainer
    from rlkit.torch.sac.policies import TanhGaussianPolicy

    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ptu.set_gpu_mode(False)
    rows = qlearning_dataset(dataset(env_id))
    # load_hdf5 (cql_mujoco_new.py:17-25): rewards and terminals as columns.
    rewards, terminals = (
        rows["rewards"][:, None],
        rows["terminals"][:, None].astype(np.float32),
    )
    obs_dim, action_dim = rows["observations"].shape[1], rows["actions"].shape[1]

    def critic():
        return FlattenMlp(
            input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[WIDTH] * 3
        )

    qf1, qf2, target_qf1, target_qf2 = critic(), critic(), critic(), critic()
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[WIDTH] * 3
    )
    env = types.SimpleNamespace(action_space=types.SimpleNamespace(shape=(action_dim,)))
    trainer = CQLTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **TRAINER,
    )
    size = rows["rewards"].shape[0]
    for call in range(1, max(RECORDS) + 1):
        indices = np.random.randint(0, size, BATCH)  # SimpleReplayBuffer.random_batch
        np_batch = {
            "observations": rows["observations"][indices],
            "actions": rows["actions"][indices],
            "rewards": rewards[indices],
            "terminals": terminals[indices],
            "next_observations": rows["next_observations"][indices],
        }
        if call in RECORDS:
            tape: list = []
            f32 = one_call(trainer, np_batch, tape, double=False)
            f64 = one_call(trainer, np_batch, tape, double=True)
            arrays = {f"in.{k}": v for k, v in state(trainer).items()}
            arrays.update({f"batch.{k}": v for k, v in np_batch.items()})
            arrays.update({f"draw.{i}.{kind}": v for i, (kind, v) in enumerate(tape)})
            arrays.update({f"f32.{k}": v for k, v in f32.items()})
            arrays.update({f"f64.{k}": v for k, v in f64.items()})
            np.savez_compressed(out / f"{env_id}_{call}.npz", **arrays)
        trainer.train(np_batch)
    write_versions(out, aviralkumar2907_cql=COMMIT)


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    main(pathlib.Path(sys.argv[1]), sys.argv[2], out)
