# Examples

Complete scripts, a few dozen lines each, for a look at Ajax beyond the
[quickstart](https://yannberthelot.github.io/Ajax/latest/getting-started/quickstart/).
Run one from the repository's root:

```bash
python examples/train_ppo_cartpole.py
```

| Script | What it does |
| --- | --- |
| `train_ppo_cartpole.py` | trains PPO on CartPole with three seeds, then prints when it evaluated, the last returns and each seed's score on new episodes |
| `sac_pendulum_tensorboard.py` | trains SAC on Pendulum with two seeds and logs both runs to TensorBoard, in `runs/` |
| `save_and_act.py` | trains PPO on CartPole, saves and loads it, and plays an episode with `agent.predict` |
| `hyperparameter_sweep.py` | trains SAC with three learning rates and two seeds each, in one program, and prints each rate's final return |

Each script's `main` takes the number of training steps, so
`main(n_timesteps=...)` runs a shorter version; the SAC scripts take a few
minutes on a CPU. The TensorBoard viewer is installed apart:
`pip install tensorboard`, then `tensorboard --logdir runs`.
[`notebooks/quickstart.ipynb`](../notebooks/quickstart.ipynb) is the
quickstart as a notebook.
