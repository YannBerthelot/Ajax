"""Ajax without its brax and playground extras: in a fresh interpreter where
every package only those extras bring fails to import, a gymnax agent still
trains, and a brax id names the extras to install."""

import subprocess
import sys

# Import names of the packages that only the brax and playground extras
# install (brax's and playground's own dependencies included).
EXTRAS_ONLY = (
    "brax",
    "mujoco",
    "mujoco_playground",
    "warp",
    "jaxopt",
    "ml_collections",
    "trimesh",
    "flask",
    "flask_cors",
    "glfw",
    "OpenGL",
    "lxml",
    "mediapy",
    "IPython",
)

CODE = f"""
import sys


class NotInstalled:
    \"\"\"Imports of the extras' packages fail as if they were absent.\"\"\"

    def find_spec(self, name, path=None, target=None):
        package = name.partition(".")[0]
        if package in {EXTRAS_ONLY!r}:
            raise ModuleNotFoundError(f"No module named {{package!r}}", name=package)
        return None


sys.meta_path.insert(0, NotInstalled())

from ajax import PPO
from ajax.environments.create import build_env_from_id
from ajax.environments.utils import check_env_is_brax

env, _ = build_env_from_id("CartPole-v1")
assert not check_env_is_brax(env)
try:
    build_env_from_id("ant")
except ValueError as error:
    assert "ajax-rl[brax]" in str(error) and "ajax-rl[playground]" in str(error), error
else:
    raise AssertionError("ant built without brax")
PPO("CartPole-v1", n_envs=1, n_steps=16, batch_size=8).train(seed=0, n_timesteps=32)
print("trained")
"""


def test_a_gymnax_agent_trains_without_the_brax_and_playground_extras():
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "trained"
