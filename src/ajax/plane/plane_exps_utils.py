import jax


def get_mode() -> str:
    return "GPU" if jax.default_backend() == "gpu" else "CPU"
