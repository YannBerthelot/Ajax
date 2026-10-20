# gauthamvasan/avg (vendored, unmodified)

`avg.py`, `incremental_rl/__init__.py` and `incremental_rl/td_error_scaler.py`
are copied unchanged from [gauthamvasan/avg](https://github.com/gauthamvasan/avg)
at commit `6b6822c` (2025-02-19), the code of the AVG paper's authors (Vasan,
Elsayed, Azimi, He, Shahriar, Bellinger, White, Mahmood, *Deep Policy Gradient
Methods Without Batch Updates, Target Networks, or Replay Buffers*, NeurIPS
2024). MIT License, see `LICENSE`.

`../../conformance_avg.py` imports `avg.py` as it is (its unused video helper,
`incremental_rl.experiment_tracker`, which imports cv2, is stubbed) and drives
it with `avg.py`'s `main()` loop, written out there, on a gymnax task;
`tests/conformance/test_avg_original.py` compares one of its updates with
Ajax's.
