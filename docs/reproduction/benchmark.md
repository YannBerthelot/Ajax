# Agent benchmark

Ten agents on nineteen tasks, with 10 seeds each and their default settings
apart from `n_envs`, which follows each agent's paper. How Ajax's agents
compare with other implementations is in the [results](results.md).

## Setup

- **Tasks.** Fifteen DeepMind Control tasks on MuJoCo Playground, trained for
  1,048,576 steps (APO 1,024,000, a whole number of its rollouts), and four
  MinAtar games on gymnax, trained for 5,242,880 steps.
- **Machines.** CPU only, on 4-core cloud machines, seeds 0 to 9 trained
  together. DQN on Freeway and SpaceInvaders ran in two halves, for memory.
- **Code.** The main branch at `acd1626`, after 0.5.0.
- **Score.** The return at the last evaluation, as mean ± standard deviation
  over the seeds, with the best mean in each row in bold. With 10 seeds, a
  small gap between two agents means little. DeepMind Control scores run from
  0 to 1000.
- **Records.** [`benchmarks/suite`](https://github.com/YannBerthelot/Ajax/tree/main/benchmarks/suite)
  holds the driver and each run's records.

## DeepMind Control

A dash means the agent did not run that task: only AVG, APO and PPO are fast
enough on a CPU to run them all, so on those rows the bold compares these
three.

| Task | SAC | TD3 | RVI-SAC | ASAC | AVG | APO | PPO |
|---|---|---|---|---|---|---|---|
| CheetahRun | 846.1 ± 127.2 | **846.5 ± 87.3** | 827.4 ± 42.7 | 625.9 ± 21.5 | 330.1 ± 9.5 | 212.7 ± 67.6 | 214.9 ± 74.0 |
| WalkerWalk | 961.8 ± 28.2 | **967.5 ± 4.0** | 959.3 ± 26.0 | 877.9 ± 77.6 | 420.2 ± 203.8 | 196.1 ± 43.9 | 260.6 ± 76.4 |
| WalkerRun | – | – | – | – | 27.5 ± 5.5 | 53.9 ± 7.0 | **74.0 ± 14.7** |
| WalkerStand | – | – | – | – | 264.8 ± 46.6 | **270.1 ± 39.3** | 254.6 ± 54.2 |
| HopperHop | 0.2 ± 0.3 | 0.1 ± 0.1 | 0.1 ± 0.3 | 0.2 ± 0.3 | 0.0 ± 0.0 | **0.6 ± 0.9** | 0.3 ± 0.5 |
| HopperStand | – | – | – | – | 2.2 ± 2.1 | 13.6 ± 20.7 | **25.9 ± 55.3** |
| FingerSpin | – | – | – | – | 0.1 ± 0.2 | 231.1 ± 98.6 | **324.2 ± 275.6** |
| FingerTurnHard | – | – | – | – | **97.5 ± 103.8** | 75.7 ± 62.0 | 86.6 ± 32.0 |
| ReacherHard | – | – | – | – | **141.9 ± 250.4** | 87.5 ± 154.9 | 38.2 ± 55.2 |
| CartpoleSwingup | – | – | – | – | 144.0 ± 52.7 | **769.2 ± 146.0** | 716.7 ± 183.7 |
| AcrobotSwingup | – | – | – | – | 2.1 ± 2.1 | **48.0 ± 16.4** | 33.3 ± 15.2 |
| PendulumSwingup | – | – | – | – | 0.3 ± 0.7 | 31.0 ± 24.2 | **407.5 ± 241.2** |
| BallInCup | – | – | – | – | 0.0 ± 0.0 | 0.0 ± 0.0 | **565.1 ± 486.8** |
| FishSwim | – | – | – | – | 64.0 ± 17.3 | 76.9 ± 11.9 | **88.0 ± 31.8** |
| HumanoidWalk | – | – | – | – | **15.8 ± 4.8** | 3.3 ± 1.6, unfinished | 6.4 ± 3.3 |

## MinAtar

| Game | DQN | PPO | PQN | UDRL |
|---|---|---|---|---|
| Breakout | **83.6 ± 12.3** | 27.1 ± 14.0 | 23.8 ± 1.8 | 4.0 ± 1.3 |
| Asterix | 1.0 ± 0.6 | **5.8 ± 2.0** | 4.1 ± 2.0 | 0.4 ± 0.3 |
| Freeway | **51.9 ± 4.0** | 41.7 ± 4.7 | 23.4 ± 14.1 | 2.7 ± 5.7 |
| SpaceInvaders | 143.1 ± 20.3 | **149.9 ± 34.2** | 137.6 ± 26.3 | 8.9 ± 2.0 |

## Reading the tables

- **APO on HumanoidWalk is unfinished.** Its records stop at step 81,920 of
  1,024,000, because the next chunk hit non-finite values twice, so its score
  comes from early in training. The likely source is the simulator.
- **Two runs are second attempts, from step 0.** PPO's first attempt on
  HumanoidWalk also hit non-finite values, past step 327,680, and AVG's on
  HopperStand was cut off by a machine restart.
- **No agent learned HopperHop** at this budget.
- **The last evaluation can swing.** On BallInCup, 6 of PPO's seeds ended above
  890 and 4 at 0, and APO ended at 0 after scoring higher earlier.
- **Hopper and Finger runs turned an XLA pass off**, as
  [Environments](../guide/environments.md#brax-and-playground) explains.

## Agents not in the tables

REDQ, TQC, CrossQ, DreamerV3 and TD-MPC2 were timed and left out, as they are
too slow to train 10 seeds this way on a CPU. APG needs a differentiable
task, IQL a dataset and DAgger an expert to query, and TD-MPC2's multi-task
agent trains on a suite of tasks.
