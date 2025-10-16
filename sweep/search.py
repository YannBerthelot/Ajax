from create_study import main as create_study_main
from long_run import main as long_run_main
from utils import get_args
from worker import main as worker_main


def main(args):
    create_study_main(args)
    worker_main(args)
    long_run_main(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
