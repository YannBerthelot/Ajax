from importlib.metadata import version

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO_pre_train import PPO
from ajax.agents.SAC.SAC import SAC
from ajax.agents.REDQ.REDQ import REDQ

__all__ = ["APO", "AVG", "PPO", "SAC", "ASAC", "REDQ"]
__version__ = version("ajax")
