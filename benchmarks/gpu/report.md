| agent | seeds | cpu (cpu) | gpu (NVIDIA L40S) |
| --- | --- | --- | --- |
| DQN | 1 | 8,124 | 10,279 |
| DQN | 16 | 22,545 | 163,638 |
| DQN | 128 | - | 1,217,810 |
| PPO | 1 | 9,085 | 15,339 |
| PPO | 16 | 33,696 | 129,646 |
| PPO | 128 | - | 893,265 |
| PQN | 1 | 11,407 | 17,458 |
| PQN | 16 | 63,370 | 182,008 |
| PQN | 128 | - | 1,399,572 |
| SAC | 1 | 664 | 6,526 |
| SAC | 16 | 2,467 | 58,336 |
| SAC | 128 | - | 55,029 |

- DQN: GPU faster at 1, 16 seeds; CPU at least as fast at no seeds.
- PPO: GPU faster at 1, 16 seeds; CPU at least as fast at no seeds.
- PQN: GPU faster at 1, 16 seeds; CPU at least as fast at no seeds.
- SAC: GPU faster at 1, 16 seeds; CPU at least as fast at no seeds.
