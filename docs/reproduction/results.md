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
| PPO | v1, c4aeac6 | similar | ±25 | 96 | 493.8 | 485.5 | -8.36 [-22, +3.73] | none shown | 12 |
| PPO-inverted_pendulum | v1, df60228 | similar | ±50 | 96 | 1000 | 1000 | +0 [+0, +0] | none shown | 0 |
| PQN | v1, 7938ae7 | similar | ±25 | 96 | 498 | 499.9 | +1.92 [-0.0527, +4.71] | none shown | 0.0084 |
| PQN-Asterix | v1, f1c5012 | inconclusive | ±0.115 | 96 | 2.292 | 2.455 | +0.162 [+0.0287, +0.298] | none shown | 0.2 |

## Speed

| Comparison | Side | Steps per second | CPU seconds per 1M steps | Compile seconds |
|---|---|---|---|---|
| PPO | sb3 | 9,807 | 395 | 3.5 |
| PPO | Ajax | 25,087 | 63 | 7.8 |
| PPO-inverted_pendulum | brax | 15,876 | 246 | 56.1 |
| PPO-inverted_pendulum | Ajax | 10,094 | 225 | 30.5 |
| PQN | purejaxql | 25,572 | 82 | 6.7 |
| PQN | Ajax | 26,944 | 80 | 5.6 |
| PQN-Asterix | purejaxql | 31,759 | 63 | 13.6 |
| PQN-Asterix | Ajax | 33,435 | 61 | 11.9 |

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores.

## Learning curves

![PPO](PPO.svg)

![PPO-inverted_pendulum](PPO-inverted_pendulum.svg)

![PQN](PQN.svg)

![PQN-Asterix](PQN-Asterix.svg)
