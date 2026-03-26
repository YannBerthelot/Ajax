"""
Parse TensorBoard runs for the `tests_SAC_plane_awbc_sweep_clean` project to CSV.

How it works:
1. For each run in ./tensorboard/, find the matching wandb folder.
2. Check if the wandb run belongs to `tests_SAC_plane_awbc_sweep_clean`.
3. Extract the experiment name from the wandb display name ({exp_name}_{run_id}).
4. Read all scalar metrics and the run config from the TensorBoard event file.
5. Write one row per (run_id, step) to a CSV file.
"""

import csv
import re
import struct as pystruct
from collections import defaultdict
from pathlib import Path

from tensorboardX.proto import event_pb2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TB_DIR = Path("tensorboard")
WANDB_DIR = Path("wandb")
TARGET_PROJECT = b"tests_SAC_plane_awbc_sweep_clean"
OUTPUT_CSV = "plane_exps_awbc_sweep_clean.csv"


# ---------------------------------------------------------------------------
# TFRecord / TensorBoard helpers
# ---------------------------------------------------------------------------


def iter_tfrecord(path):
    """Iterate raw record bytes from a TensorBoard TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = pystruct.unpack("<Q", header)[0]
            f.read(4)  # skip CRC of length
            data = f.read(length)
            f.read(4)  # skip CRC of data
            if not data:
                break
            yield data


HPARAM_KEYS = [
    "use_box",
    "use_expert_warmup",
    "use_expert_guidance",
    "use_mc_critic_pretrain",
    "value_constraint_coef",
    "augment_obs_with_expert_action",
    "num_critics",
    "num_critic_updates",
    "expert_buffer_n_steps",
    "expert_mix_fraction",
    "box_threshold",
    "proximity_scale",
    "tau",
    "target_entropy_per_dim",
]


def load_event_file(event_file):
    """
    Parse a TensorBoard event file and return:
      - scalars:  {step: {tag: float}}
      - hparams:  {key: value}  (from the config/text_summary JSON written at step 0)
    """
    import json

    scalars = defaultdict(dict)
    hparams = {}

    for record in iter_tfrecord(event_file):
        e = event_pb2.Event()
        e.ParseFromString(record)
        if not e.summary:
            continue
        for v in e.summary.value:
            if v.HasField("simple_value"):
                scalars[e.step][v.tag] = v.simple_value
            elif v.tag == "config/text_summary" and v.tensor.string_val:
                try:
                    cfg = json.loads(v.tensor.string_val[0])
                    hparams = {k: cfg[k] for k in HPARAM_KEYS if k in cfg}
                except (json.JSONDecodeError, KeyError):
                    pass

    return scalars, hparams


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def get_wandb_entries(run_id):
    """Return wandb run folders for a given run_id, sorted chronologically."""
    return sorted(
        [d for d in WANDB_DIR.iterdir() if d.name.endswith(f"-{run_id}")],
        key=lambda x: x.name,
    )


def get_project_and_display_name(run_id):
    """
    Return (is_target_project: bool, display_name: str | None) for a run_id
    by reading wandb binary files until the target project is found.
    """
    entries = get_wandb_entries(run_id)
    for entry in entries:
        wandb_files = list(entry.glob("*.wandb"))
        if not wandb_files:
            continue
        data = wandb_files[0].read_bytes()

        # Project name: check if this run belongs to the target project
        if TARGET_PROJECT not in data[:500]:
            continue

        # Display name: pattern like `obs_augment_mc_pretrain_00jak40q`
        name_match = re.search(
            rb"([a-z][a-z_0-9]+_" + run_id.encode() + rb")", data
        )
        display_name = name_match.group(1).decode() if name_match else None

        return True, display_name

    return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    run_dirs = sorted(TB_DIR.iterdir())
    print(f"Found {len(run_dirs)} tensorboard runs. Filtering by project...")

    # Phase 1: identify which runs belong to the target project
    target_runs = []  # list of (run_id, exp_name)
    for run_dir in tqdm(run_dirs, desc="Scanning wandb metadata"):
        run_id = run_dir.name
        is_target, display_name = get_project_and_display_name(run_id)
        if not is_target:
            continue
        # Experiment name = display name minus the trailing `_{run_id}`
        if display_name and display_name.endswith(f"_{run_id}"):
            exp_name = display_name[: -(len(run_id) + 1)]
        else:
            exp_name = display_name or "unknown"
        target_runs.append((run_id, exp_name))

    print(f"\n{len(target_runs)} runs belong to {TARGET_PROJECT.decode()}")

    if not target_runs:
        print("Nothing to write.")
        return

    # Phase 2: read scalars + hparams and collect all tag names
    print("\nReading TensorBoard events...")
    all_data = []  # list of (run_id, exp_name, hparams, step, metrics)
    all_tags = set()

    for run_id, exp_name in tqdm(target_runs, desc="Reading events"):
        event_files = list((TB_DIR / run_id).glob("events.out.tfevents.*"))
        if not event_files:
            continue
        scalars, hparams = load_event_file(event_files[0])
        for step, metrics in sorted(scalars.items()):
            all_data.append((run_id, exp_name, hparams, step, metrics))
            all_tags.update(metrics.keys())

    all_tags = sorted(all_tags)
    print(f"Tags found: {all_tags}")

    # Phase 3: write CSV
    print(f"\nWriting {OUTPUT_CSV}...")
    fieldnames = ["run_id", "exp_name"] + HPARAM_KEYS + ["step"] + all_tags
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, exp_name, hparams, step, metrics in all_data:
            row = {"run_id": run_id, "exp_name": exp_name, "step": step}
            row.update(hparams)
            row.update(metrics)
            writer.writerow(row)

    print(f"Done. {len(all_data)} rows written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
