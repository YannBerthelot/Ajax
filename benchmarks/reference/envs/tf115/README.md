# TensorFlow 1.15 for OpenAI Baselines' ppo2

`../../conformance_ppo2.py` records ppo2's update (`openai/baselines` @ea25b9e,
`../../vendor/openai_baselines`) for `tests/conformance/test_ppo2_original.py`.
ppo2 is TensorFlow 1 code, and TensorFlow 1.15 has wheels for Python 3.7 at the
newest, which uv no longer installs, and for x86-64 only. So it runs in the
official image `tensorflow/tensorflow:1.15.5-py3` (Python 3.6.9, TensorFlow
1.15.5, numpy 1.18.5) on a Linux x86-64 machine, with `requirements.txt`
installed beside it. CI cannot run it: the records are committed in
`tests/conformance/records/openai_baselines`, each case under one second of CPU.

To remake them (with Singularity; Docker works the same way), from the
repository root:

```bash
singularity pull tf115.sif docker://tensorflow/tensorflow:1.15.5-py3
singularity exec tf115.sif pip install --no-deps --target pydeps -r benchmarks/reference/envs/tf115/requirements.txt
for case in mlp cnn; do
  PYTHONPATH=pydeps singularity exec tf115.sif python benchmarks/reference/conformance_ppo2.py $case tests/conformance/records/openai_baselines
done
```
