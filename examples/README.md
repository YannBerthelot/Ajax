# Examples

Small scripts, each complete on its own. Run one from the repository's root:

```bash
python examples/train_ppo_cartpole.py
```

| Script | What it shows |
| --- | --- |
| `train_ppo_cartpole.py` | train PPO on CartPole with three seeds and read the evaluations |
| `sac_pendulum_tensorboard.py` | train SAC on Pendulum and log the runs to TensorBoard |
| `save_and_act.py` | save a trained state, load it into a new agent and play an episode |
| `hyperparameter_sweep.py` | train a grid of learning rates with every seed in one program |

Each script's `main` takes the number of training steps, so a shorter run is
`main(n_timesteps=...)` from Python. The tests run every script
(`tests/test_examples.py`). The [documentation](https://yannberthelot.github.io/Ajax/)
explains what the scripts use; `notebooks/quickstart.ipynb` is the quickstart
as a notebook.
