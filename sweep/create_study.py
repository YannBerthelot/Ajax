import os

import optuna
from utils import get_args


def main(args):
    if args.new_study:
        if os.path.exists(f"sweep/{args.agent}/study.db"):
            os.remove(f"sweep/{args.agent}/study.db")
    os.makedirs(f"sweep/{args.agent}", exist_ok=True)
    optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///sweep/{args.agent}/study.db",
        study_name=f"{args.agent}-{args.env_id}-study",
        load_if_exists=not (args.new_study),
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
