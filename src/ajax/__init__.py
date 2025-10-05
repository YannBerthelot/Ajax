from importlib.metadata import version

from ajax.agents.APO.APO import APO
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO_pre_train import PPO
from ajax.agents.SAC.SAC import SAC

__all__ = ["APO", "AVG", "PPO", "SAC"]
__version__ = version("ajax")
