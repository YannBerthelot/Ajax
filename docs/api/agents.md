# Agents

Every agent but `TDMPC2MultiTask` subclasses `ActorCritic`. Its constructor
takes the environment, the algorithm's own hyperparameters and
`extensions=()`; `train` runs one or more seeds at once. `TDMPC2MultiTask`
learns offline from a dataset and keeps its own `train`, with the same
seeds-at-once call.

Each agent's reference, with every constructor argument, is at the end of its
[algorithm page](../algorithms/index.md), next to the paper it follows and the
notes on using it:

| Agent | Class | Reference |
| --- | --- | --- |
| SAC | `ajax.agents.SAC.SAC.SAC` | [SAC](../algorithms/sac.md#api) |
| ASAC | `ajax.agents.ASAC.ASAC.ASAC` | [ASAC](../algorithms/asac.md#api) |
| REDQ | `ajax.agents.REDQ.REDQ.REDQ` | [REDQ](../algorithms/redq.md#api) |
| CrossQ | `ajax.agents.CrossQ.CrossQ.CrossQ` | [CrossQ](../algorithms/crossq.md#api) |
| TD3 | `ajax.agents.TD3.TD3.TD3` | [TD3](../algorithms/td3.md#api) |
| TQC | `ajax.agents.TQC.TQC.TQC` | [TQC](../algorithms/tqc.md#api) |
| IQL | `ajax.agents.IQL.IQL.IQL` | [IQL](../algorithms/iql.md#api) |
| AVG | `ajax.agents.AVG.AVG.AVG` | [AVG](../algorithms/avg.md#api) |
| DQN | `ajax.agents.DQN.DQN.DQN` | [DQN](../algorithms/dqn.md#api) |
| PQN | `ajax.agents.PQN.PQN.PQN` | [PQN](../algorithms/pqn.md#api) |
| PPO | `ajax.agents.PPO.PPO.PPO` | [PPO](../algorithms/ppo.md#api) |
| APO | `ajax.agents.APO.APO.APO` | [APO](../algorithms/apo.md#api) |
| UDRL | `ajax.agents.UDRL.UDRL.UDRL` | [UDRL](../algorithms/udrl.md#api) |
| DAgger | `ajax.agents.DAgger.DAgger.DAgger` | [DAgger](../algorithms/dagger.md#api) |
| APG | `ajax.agents.APG.APG.APG` | [APG](../algorithms/apg.md#api) |
| DreamerV3 | `ajax.agents.DreamerV3.DreamerV3.DreamerV3` | [DreamerV3](../algorithms/dreamerv3.md#api) |
| TDMPC2 | `ajax.agents.TDMPC2.TDMPC2.TDMPC2` | [TD-MPC2](../algorithms/tdmpc2.md#api) |
| TDMPC2MultiTask | `ajax.agents.TDMPC2.TDMPC2MultiTask.TDMPC2MultiTask` | [TD-MPC2](../algorithms/tdmpc2.md#api) |

Every agent is also exported from `ajax` itself (`from ajax import SAC`).

::: ajax.agents.base.ActorCritic
