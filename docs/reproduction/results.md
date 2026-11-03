# Results

Written by `benchmarks/reference/refcmp_publish.py docs` from the
records the dataset publishes. Ajax is its matched arm. The
difference is Ajax's mean final return minus the reference's, with
its 95% bootstrap interval; the verdict sets that interval against the
registered margin. AdaStop names the difference it detected, and the
resolution is the smallest gap its budget detects 80% of the time.
Speed is measured on the CPU, medians over batches.

Each row reports its comparison's latest spec version, whose settings
the [specifications](specs.md) list, and the Ajax commit it ran.

| Comparison | Spec, Ajax | Verdict | Margin | Seeds | Reference | Ajax | Difference | AdaStop | Resolution |
|---|---|---|---|---|---|---|---|---|---|
| PPO | v2, 1adddad | similar | ±25 | 96 | 493.8 | 496.7 | +2.86 [-3.72, +10.3] | none shown | 2.6 |
| PPO-inverted_pendulum | v1, df60228 | similar | ±50 | 96 | 1000 | 1000 | +0 [+0, +0] | none shown | 0 |
| PQN | v3, c2b3318 | similar | ±25 | 96 | 498 | 499 | +1.04 [-1.95, +4.29] | none shown | 0.011 |
| PQN-Asterix | v1, f1c5012 | inconclusive | ±0.115 | 96 | 2.292 | 2.455 | +0.162 [+0.0287, +0.298] | none shown | 0.2 |
| SAC | v2, 340ed65 | similar | ±14 | 96 | -146.7 | -148 | -1.25 [-7.92, +5.35] | none shown | 9.4 |
| TD3 | v2, da495af | similar | ±14 | 96 | -148.9 | -144.2 | +4.63 [-2.8, +12] | none shown | 12 |

## Speed

| Comparison | Side | Steps per second | CPU seconds per 1M steps | Compile seconds |
|---|---|---|---|---|
| PPO | sb3 | 7,663 | 502 | 4.0 |
| PPO | Ajax | 33,506 | 54 | 1.7 |
| PPO-inverted_pendulum | brax | 15,876 | 246 | 56.1 |
| PPO-inverted_pendulum | Ajax | 10,094 | 225 | 30.5 |
| PQN | purejaxql | 17,757 | 121 | 8.3 |
| PQN | Ajax | 25,134 | 96 | 1.3 |
| PQN-Asterix | purejaxql | 31,759 | 63 | 13.6 |
| PQN-Asterix | Ajax | 33,435 | 61 | 11.9 |
| SAC | sb3 | 164 | 23,620 | 4.8 |
| SAC | Ajax | 227 | 13,242 | 1.4 |
| TD3 | sb3 | 175 | 22,334 | 4.9 |
| TD3 | Ajax | 179 | 17,257 | 1.5 |

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores.

## Learning curves

![PPO](PPO.svg)

![PPO-inverted_pendulum](PPO-inverted_pendulum.svg)

![PQN](PQN.svg)

![PQN-Asterix](PQN-Asterix.svg)

![SAC](SAC.svg)

![TD3](TD3.svg)
