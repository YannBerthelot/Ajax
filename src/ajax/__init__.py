from importlib.metadata import version

from ajax import _compile_cache

# Before the agent imports below, so every compile they trigger is cached.
_compile_cache.configure()

from ajax.agents.APG.APG import APG  # noqa: E402
from ajax.agents.APO.APO import APO  # noqa: E402
from ajax.agents.ASAC.ASAC import ASAC  # noqa: E402
from ajax.agents.AVG.AVG import AVG  # noqa: E402
from ajax.agents.CrossQ.CrossQ import CrossQ  # noqa: E402
from ajax.agents.DAgger.DAgger import DAgger  # noqa: E402
from ajax.agents.DQN.DQN import DQN  # noqa: E402
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3  # noqa: E402
from ajax.agents.IQL.IQL import IQL  # noqa: E402
from ajax.agents.PPO.PPO import PPO  # noqa: E402
from ajax.agents.PQN.PQN import PQN  # noqa: E402
from ajax.agents.REDQ.REDQ import REDQ  # noqa: E402
from ajax.agents.RVISAC.RVISAC import RVISAC  # noqa: E402
from ajax.agents.SAC.SAC import SAC  # noqa: E402
from ajax.agents.TD3.TD3 import TD3  # noqa: E402
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2  # noqa: E402
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask  # noqa: E402
from ajax.agents.TQC.TQC import TQC  # noqa: E402
from ajax.agents.UDRL.UDRL import UDRL  # noqa: E402

__all__ = [
    "APG",
    "APO",
    "ASAC",
    "AVG",
    "DQN",
    "IQL",
    "PPO",
    "PQN",
    "REDQ",
    "RVISAC",
    "SAC",
    "TD3",
    "TDMPC2",
    "TQC",
    "UDRL",
    "CrossQ",
    "DAgger",
    "DreamerV3",
    "TDMPC2MultiTask",
]
__version__ = version("ajax-rl")
