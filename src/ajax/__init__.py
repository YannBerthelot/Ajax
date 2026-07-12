from importlib.metadata import version

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO import PPO
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC

__all__ = ["APO", "ASAC", "AVG", "PPO", "REDQ", "SAC"]
__version__ = version("ajax")
