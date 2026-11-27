# Compatibility

Which module runs on which agent: every [extension](extensions.md) and network option (`memory`, `cnn`), as the agent is built. Check here before combining a module with an agent. The page is generated from the code by `tests/agents/test_compatibility.py`.

**yes**: supported. **no**: the algorithm lacks a [capability](extensions.md#what-an-agent-has-and-what-an-extension-needs) the module needs (listed below). **not yet**: the agent has what the module needs, and its code does not pass it on yet (the reasons are below). An agent taking both kinds of action has a second column, `/discrete`, for discrete actions.

| Module | SAC | TD3 | REDQ | ASAC | RVISAC | AVG | CrossQ | TQC | IQL | APG | TDMPC2 | PPO | PPO/discrete | APO | APO/discrete | REINFORCE | REINFORCE/discrete | UDRL | UDRL/discrete | DAgger | DAgger/discrete | DreamerV3 | DreamerV3/discrete | DQN | PQN | TDMPC2MultiTask |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AdversarialReward` | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | yes | no | not yet | no | not yet | no | not yet | no | not yet | no | not yet | no | no | no | not yet |
| `BellmanPretrain` | yes | yes | yes | not yet | not yet | no | no | yes | not yet | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | not yet | no | not yet |
| `BiasVoreDecomposition` | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes | yes | no | no | no | no | yes | yes | yes | yes | yes |
| `BiasVorePenalty` | yes | yes | yes | yes | yes | yes | yes | no | yes | no | not yet | yes | yes | yes | yes | not yet | not yet | no | no | no | no | not yet | not yet | yes | yes | not yet |
| `BumplessTracking` | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | no | yes | yes | yes | yes | not yet | not yet | not yet | not yet | yes | yes | no | no | yes | yes | no |
| `CHEQ` | yes | yes | yes | yes | yes | no | yes | no | yes | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no |
| `CalQL` | yes | no | yes | no | no | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `CloningPretrain` | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | not yet | not yet | not yet | yes | not yet | not yet | not yet | not yet | not yet | yes | not yet | not yet | not yet | no | no | not yet |
| `ConservativeQ` | yes | no | yes | yes | yes | yes | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `CriticBlend` | yes | not yet | not yet | no | no | not yet | not yet | no | not yet | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | not yet | not yet | not yet |
| `CriticDiagnostics` | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes | yes | not yet | not yet | not yet | not yet | yes | yes | yes | yes | yes |
| `EDGEExploration` | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | not yet | no | no | no | no | no | no | yes | no | yes | no | not yet | no | no | no | not yet |
| `EpsilonGreedyPresample` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | yes | no | yes | no | not yet | no | no | no | not yet |
| `ExpertActionInput` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | yes | yes | yes | yes | no | no | yes | yes | no | no | not yet | not yet | yes | yes | not yet |
| `ExpertGains` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | yes | no | yes | no | no | no | yes | no | no | no | no | no | no | no | no |
| `ExpertPrefill` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | not yet | not yet | not yet | not yet | not yet | not yet | yes | no | not yet |
| `ExpertStateInput` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | yes | yes | yes | yes | not yet | not yet | yes | yes | yes | yes | not yet | not yet | yes | yes | not yet |
| `ExpertValueMetrics` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | yes | no | not yet |
| `ExpertWarmup` | yes | yes | yes | yes | yes | no | yes | no | no | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | yes | no | no |
| `IBRL` | yes | yes | yes | no | no | no | no | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `ImitationLoss` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | yes | not yet | yes | not yet | not yet | not yet | yes | not yet | yes | not yet | not yet | not yet | no | no | not yet |
| `JSRLCurriculum` | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | not yet | no | no | no | no | no | no | yes | yes | yes | yes | not yet | not yet | yes | yes | not yet |
| `KernelRepulsion` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `LCBGatedBootstrap` | yes | yes | yes | no | no | no | no | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `MCPretrain` | yes | not yet | not yet | no | no | not yet | not yet | not yet | not yet | no | not yet | not yet | not yet | no | no | not yet | not yet | no | no | no | no | not yet | not yet | not yet | not yet | not yet |
| `MCVarianceCorrection` | yes | not yet | not yet | no | no | no | not yet | no | not yet | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `MEW` | yes | no | yes | yes | yes | yes | yes | yes | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no |
| `OnlineBC` | yes | not yet | not yet | no | no | not yet | not yet | not yet | not yet | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `OuterLoopReference` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | yes | no | yes | no | no | no | yes | no | no | no | no | no | no | no | no |
| `PIDTermsInput` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet |
| `PhiRefresh` | yes | not yet | not yet | no | no | no | not yet | not yet | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet | no | no |
| `PreviousAction` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet |
| `PreviousReward` | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet |
| `PriorData` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | not yet | not yet | not yet | not yet | not yet | not yet | yes | no | not yet |
| `RLPD` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | not yet | not yet | not yet | not yet | not yet | not yet | yes | no | not yet |
| `RepresentationProbe` | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet | no | yes | yes | yes | yes | yes | yes | not yet | not yet | yes | yes | no | no | yes | yes | no |
| `ResetCost` | yes | yes | yes | no | no | no | yes | yes | no | no | no | no | no | no | no | no | no | no | no | no | no | no | no | yes | no | no |
| `ResidualPolicy` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | yes | no | yes | no | no | no | yes | no | no | no | no | no | no | no | no |
| `RetainedDemonstrations` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | not yet | not yet | not yet | not yet | not yet | not yet | yes | no | not yet |
| `StorePolicyAction` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | yes | yes | no | no | not yet | not yet | yes | no | not yet |
| `TargetBC` | yes | yes | yes | no | no | no | yes | yes | no | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `TrainingProgressInput` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | not yet |
| `ValueBox` | yes | not yet | not yet | no | no | not yet | not yet | not yet | not yet | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | not yet | not yet | not yet |
| `ValueCentering` | yes | yes | yes | no | no | yes | yes | yes | yes | no | no | no | no | no | no | no | no | no | no | no | no | no | no | yes | no | no |
| `ImitationLoss(q_filter=True)` | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `ImitationLoss(target='replay', alpha=2.5)` | yes | yes | yes | yes | yes | no | yes | yes | yes | no | not yet | no | no | no | no | no | no | no | no | no | no | no | no | no | no | not yet |
| `memory` | yes | yes | yes | yes | yes | not yet | not yet | yes | not yet | yes | no | yes | yes | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | no | no | not yet | not yet | no |
| `cnn` | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | not yet | yes | yes | not yet | not yet | not yet | not yet | yes | yes | not yet | not yet | not yet | not yet | yes | yes | not yet |

## Refused: what the algorithm lacks

- `AdversarialReward` needs continuous_actions: not on PPO/discrete, APO/discrete, REINFORCE/discrete, UDRL/discrete, DAgger/discrete, DreamerV3/discrete, DQN, PQN
- `BellmanPretrain` needs critic, off_policy, replay, target_critic: not on APG
- `BellmanPretrain` needs critic, target_critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `BellmanPretrain` needs off_policy, replay, target_critic: not on PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `BellmanPretrain` needs replay, target_critic: not on AVG, PQN
- `BellmanPretrain` needs target_critic: not on CrossQ, DreamerV3, DreamerV3/discrete
- `BiasVoreDecomposition` needs critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `BiasVoreDecomposition` needs critic, stored_steps: not on APG
- `BiasVorePenalty` needs critic, scalar_target: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `BiasVorePenalty` needs scalar_target: not on TQC
- `BumplessTracking` needs obs_networks: not on TDMPC2, DreamerV3, DreamerV3/discrete, TDMPC2MultiTask
- `CHEQ` needs continuous_actions, critic, critic_ensemble, q_critic, scalar_target: not on UDRL/discrete, DAgger/discrete
- `CHEQ` needs continuous_actions, critic_ensemble: not on DQN
- `CHEQ` needs continuous_actions, critic_ensemble, obs_networks, q_critic: not on DreamerV3/discrete
- `CHEQ` needs continuous_actions, critic_ensemble, off_policy, q_critic, replay: not on PPO/discrete, APO/discrete, REINFORCE/discrete
- `CHEQ` needs continuous_actions, critic_ensemble, replay: not on PQN
- `CHEQ` needs critic, critic_ensemble, off_policy, q_critic, replay, scalar_target: not on APG
- `CHEQ` needs critic, critic_ensemble, q_critic, scalar_target: not on UDRL, DAgger
- `CHEQ` needs critic_ensemble, obs_networks, q_critic: not on DreamerV3
- `CHEQ` needs critic_ensemble, off_policy, q_critic, replay: not on PPO, APO, REINFORCE
- `CHEQ` needs critic_ensemble, replay: not on AVG
- `CHEQ` needs obs_networks: not on TDMPC2, TDMPC2MultiTask
- `CHEQ` needs scalar_target: not on TQC
- `CalQL` needs actor, continuous_actions, policy_density: not on DQN
- `CalQL` needs actor, continuous_actions, policy_density, replay: not on PQN
- `CalQL` needs continuous_actions, critic, discounted, policy_density, q_critic: not on DAgger/discrete
- `CalQL` needs continuous_actions, critic, discounted, q_critic: not on UDRL/discrete
- `CalQL` needs continuous_actions, discounted, q_critic, replay: not on APO/discrete
- `CalQL` needs continuous_actions, q_critic: not on DreamerV3/discrete
- `CalQL` needs continuous_actions, q_critic, replay: not on PPO/discrete, REINFORCE/discrete
- `CalQL` needs critic, discounted, policy_density, q_critic: not on DAgger
- `CalQL` needs critic, discounted, policy_density, q_critic, replay: not on APG
- `CalQL` needs critic, discounted, q_critic: not on UDRL
- `CalQL` needs discounted: not on ASAC, RVISAC
- `CalQL` needs discounted, q_critic, replay: not on APO
- `CalQL` needs policy_density: not on TD3
- `CalQL` needs q_critic: not on DreamerV3
- `CalQL` needs q_critic, replay: not on PPO, REINFORCE
- `CalQL` needs replay: not on AVG
- `CloningPretrain` needs actor, continuous_actions: not on DQN, PQN
- `ConservativeQ` needs actor, continuous_actions, policy_density: not on DQN, PQN
- `ConservativeQ` needs continuous_actions, critic, policy_density, q_critic: not on DAgger/discrete
- `ConservativeQ` needs continuous_actions, critic, q_critic: not on UDRL/discrete
- `ConservativeQ` needs continuous_actions, q_critic: not on PPO/discrete, APO/discrete, REINFORCE/discrete, DreamerV3/discrete
- `ConservativeQ` needs critic, policy_density, q_critic: not on APG, DAgger
- `ConservativeQ` needs critic, q_critic: not on UDRL
- `ConservativeQ` needs policy_density: not on TD3
- `ConservativeQ` needs q_critic: not on PPO, APO, REINFORCE, DreamerV3
- `CriticBlend` needs critic, discounted, q_critic, scalar_target: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `CriticBlend` needs discounted: not on ASAC, RVISAC
- `CriticBlend` needs discounted, q_critic: not on APO, APO/discrete
- `CriticBlend` needs q_critic: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete, DreamerV3, DreamerV3/discrete
- `CriticBlend` needs scalar_target: not on TQC
- `CriticDiagnostics` needs critic, stored_steps: not on APG
- `EDGEExploration` needs continuous_actions: not on UDRL/discrete, DAgger/discrete, DreamerV3/discrete, DQN, PQN
- `EDGEExploration` needs continuous_actions, off_policy: not on PPO/discrete, APO/discrete, REINFORCE/discrete
- `EDGEExploration` needs off_policy: not on APG, PPO, APO, REINFORCE
- `EpsilonGreedyPresample` needs actor, continuous_actions: not on DQN
- `EpsilonGreedyPresample` needs actor, continuous_actions, replay: not on PQN
- `EpsilonGreedyPresample` needs continuous_actions: not on UDRL/discrete, DAgger/discrete, DreamerV3/discrete
- `EpsilonGreedyPresample` needs continuous_actions, off_policy, replay: not on PPO/discrete, APO/discrete, REINFORCE/discrete
- `EpsilonGreedyPresample` needs off_policy, replay: not on APG, PPO, APO, REINFORCE
- `EpsilonGreedyPresample` needs replay: not on AVG
- `ExpertActionInput` needs reward_learning: not on REINFORCE, REINFORCE/discrete, DAgger, DAgger/discrete
- `ExpertGains` needs continuous_actions: not on PPO/discrete, APO/discrete, UDRL/discrete, DQN, PQN
- `ExpertGains` needs continuous_actions, obs_networks: not on DreamerV3/discrete
- `ExpertGains` needs continuous_actions, reward_learning: not on REINFORCE/discrete, DAgger/discrete
- `ExpertGains` needs obs_networks: not on TDMPC2, DreamerV3, TDMPC2MultiTask
- `ExpertGains` needs reward_learning: not on REINFORCE, DAgger
- `ExpertPrefill` needs off_policy, replay: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `ExpertPrefill` needs replay: not on AVG, PQN
- `ExpertValueMetrics` needs q_critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete, DreamerV3, DreamerV3/discrete
- `ExpertValueMetrics` needs q_critic, replay: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `ExpertValueMetrics` needs replay: not on AVG, PQN
- `ExpertWarmup` needs warmup, warmup_steps: not on TQC, APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete, UDRL, UDRL/discrete, DAgger, DAgger/discrete, DreamerV3, DreamerV3/discrete, PQN, TDMPC2MultiTask
- `ExpertWarmup` needs warmup_steps: not on AVG, IQL
- `IBRL` needs actor: not on DQN
- `IBRL` needs actor, target_critic: not on PQN
- `IBRL` needs critic, discounted, off_policy, q_critic, target_critic: not on APG
- `IBRL` needs critic, discounted, q_critic, target_critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `IBRL` needs discounted: not on ASAC, RVISAC
- `IBRL` needs discounted, off_policy, q_critic, target_critic: not on APO, APO/discrete
- `IBRL` needs off_policy, q_critic, target_critic: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete
- `IBRL` needs q_critic, target_critic: not on DreamerV3, DreamerV3/discrete
- `IBRL` needs target_critic: not on AVG, CrossQ
- `ImitationLoss` needs actor, continuous_actions: not on DQN, PQN
- `ImitationLoss(q_filter=True)` needs actor, continuous_actions: not on DQN, PQN
- `ImitationLoss(q_filter=True)` needs continuous_actions, q_critic: not on PPO/discrete, APO/discrete, REINFORCE/discrete, UDRL/discrete, DAgger/discrete, DreamerV3/discrete
- `ImitationLoss(q_filter=True)` needs q_critic: not on APG, PPO, APO, REINFORCE, UDRL, DAgger, DreamerV3
- `ImitationLoss(target='replay', alpha=2.5)` needs actor, continuous_actions: not on DQN
- `ImitationLoss(target='replay', alpha=2.5)` needs actor, continuous_actions, replay: not on PQN
- `ImitationLoss(target='replay', alpha=2.5)` needs continuous_actions, q_critic: not on UDRL/discrete, DAgger/discrete, DreamerV3/discrete
- `ImitationLoss(target='replay', alpha=2.5)` needs continuous_actions, q_critic, replay: not on PPO/discrete, APO/discrete, REINFORCE/discrete
- `ImitationLoss(target='replay', alpha=2.5)` needs q_critic: not on UDRL, DAgger, DreamerV3
- `ImitationLoss(target='replay', alpha=2.5)` needs q_critic, replay: not on APG, PPO, APO, REINFORCE
- `ImitationLoss(target='replay', alpha=2.5)` needs replay: not on AVG
- `JSRLCurriculum` needs off_policy: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `KernelRepulsion` needs critic, critic_ensemble, q_critic: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `KernelRepulsion` needs critic_ensemble: not on AVG, DQN, PQN
- `KernelRepulsion` needs critic_ensemble, q_critic: not on PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete, DreamerV3, DreamerV3/discrete
- `LCBGatedBootstrap` needs actor, critic_ensemble: not on DQN
- `LCBGatedBootstrap` needs actor, critic_ensemble, target_critic: not on PQN
- `LCBGatedBootstrap` needs critic, critic_ensemble, discounted, q_critic, target_critic: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `LCBGatedBootstrap` needs critic_ensemble, discounted, q_critic, target_critic: not on APO, APO/discrete
- `LCBGatedBootstrap` needs critic_ensemble, q_critic, target_critic: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete, DreamerV3, DreamerV3/discrete
- `LCBGatedBootstrap` needs critic_ensemble, target_critic: not on AVG
- `LCBGatedBootstrap` needs discounted: not on ASAC, RVISAC
- `LCBGatedBootstrap` needs target_critic: not on CrossQ
- `MCPretrain` needs critic, discounted: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `MCPretrain` needs discounted: not on ASAC, RVISAC, APO, APO/discrete
- `MCVarianceCorrection` needs critic, critic_ensemble, discounted, q_critic, scalar_target: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `MCVarianceCorrection` needs critic_ensemble: not on AVG, DQN, PQN
- `MCVarianceCorrection` needs critic_ensemble, discounted, q_critic: not on APO, APO/discrete
- `MCVarianceCorrection` needs critic_ensemble, q_critic: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete, DreamerV3, DreamerV3/discrete
- `MCVarianceCorrection` needs discounted: not on ASAC, RVISAC
- `MCVarianceCorrection` needs scalar_target: not on TQC
- `MEW` needs entropy_temperature: not on TD3, IQL, APG, TDMPC2, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete, UDRL, UDRL/discrete, DAgger, DAgger/discrete, DreamerV3, DreamerV3/discrete, DQN, PQN, TDMPC2MultiTask
- `OnlineBC` needs actor, continuous_actions: not on DQN, PQN
- `OnlineBC` needs continuous_actions, critic, discounted, q_critic: not on UDRL/discrete, DAgger/discrete
- `OnlineBC` needs continuous_actions, discounted, q_critic: not on APO/discrete
- `OnlineBC` needs continuous_actions, q_critic: not on PPO/discrete, REINFORCE/discrete, DreamerV3/discrete
- `OnlineBC` needs critic, discounted, q_critic: not on APG, UDRL, DAgger
- `OnlineBC` needs discounted: not on ASAC, RVISAC
- `OnlineBC` needs discounted, q_critic: not on APO
- `OnlineBC` needs q_critic: not on PPO, REINFORCE, DreamerV3
- `OuterLoopReference` needs continuous_actions: not on PPO/discrete, APO/discrete, UDRL/discrete, DQN, PQN
- `OuterLoopReference` needs continuous_actions, obs_networks: not on DreamerV3/discrete
- `OuterLoopReference` needs continuous_actions, reward_learning: not on REINFORCE/discrete, DAgger/discrete
- `OuterLoopReference` needs obs_networks: not on TDMPC2, DreamerV3, TDMPC2MultiTask
- `OuterLoopReference` needs reward_learning: not on REINFORCE, DAgger
- `PhiRefresh` needs critic, discounted, one_step_target, q_critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `PhiRefresh` needs critic, discounted, one_step_target, q_critic, replay: not on APG
- `PhiRefresh` needs discounted: not on ASAC, RVISAC
- `PhiRefresh` needs discounted, one_step_target, q_critic, replay: not on APO, APO/discrete
- `PhiRefresh` needs one_step_target: not on TDMPC2, TDMPC2MultiTask
- `PhiRefresh` needs one_step_target, q_critic: not on DreamerV3, DreamerV3/discrete
- `PhiRefresh` needs one_step_target, q_critic, replay: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete
- `PhiRefresh` needs one_step_target, replay: not on PQN
- `PhiRefresh` needs replay: not on AVG
- `PriorData` needs off_policy, replay: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `PriorData` needs replay: not on AVG, PQN
- `RLPD` needs off_policy, replay: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `RLPD` needs replay: not on AVG, PQN
- `RepresentationProbe` needs obs_networks: not on TDMPC2, DreamerV3, DreamerV3/discrete, TDMPC2MultiTask
- `ResetCost` needs critic, discounted, next_action_target, one_step_target: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `ResetCost` needs critic, discounted, next_action_target, one_step_target, replay: not on APG
- `ResetCost` needs discounted: not on ASAC, RVISAC
- `ResetCost` needs discounted, next_action_target, one_step_target, replay: not on APO, APO/discrete
- `ResetCost` needs next_action_target: not on IQL
- `ResetCost` needs next_action_target, one_step_target: not on DreamerV3, DreamerV3/discrete
- `ResetCost` needs next_action_target, one_step_target, replay: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete
- `ResetCost` needs one_step_target: not on TDMPC2, TDMPC2MultiTask
- `ResetCost` needs one_step_target, replay: not on PQN
- `ResetCost` needs replay: not on AVG
- `ResidualPolicy` needs continuous_actions: not on PPO/discrete, APO/discrete, UDRL/discrete, DQN, PQN
- `ResidualPolicy` needs continuous_actions, obs_networks: not on DreamerV3/discrete
- `ResidualPolicy` needs continuous_actions, reward_learning: not on REINFORCE/discrete, DAgger/discrete
- `ResidualPolicy` needs obs_networks: not on TDMPC2, DreamerV3, TDMPC2MultiTask
- `ResidualPolicy` needs reward_learning: not on REINFORCE, DAgger
- `RetainedDemonstrations` needs off_policy, replay: not on APG, PPO, PPO/discrete, APO, APO/discrete, REINFORCE, REINFORCE/discrete
- `RetainedDemonstrations` needs replay: not on AVG, PQN
- `StorePolicyAction` needs replay: not on AVG, APG, PPO, PPO/discrete, APO, APO/discrete, PQN
- `StorePolicyAction` needs replay, reward_learning: not on REINFORCE, REINFORCE/discrete
- `StorePolicyAction` needs reward_learning: not on DAgger, DAgger/discrete
- `TargetBC` needs actor, continuous_actions: not on DQN
- `TargetBC` needs actor, continuous_actions, replay: not on PQN
- `TargetBC` needs continuous_actions, critic, discounted, next_action_target, q_critic: not on UDRL/discrete, DAgger/discrete
- `TargetBC` needs continuous_actions, discounted, next_action_target, q_critic, replay: not on APO/discrete
- `TargetBC` needs continuous_actions, next_action_target, q_critic: not on DreamerV3/discrete
- `TargetBC` needs continuous_actions, next_action_target, q_critic, replay: not on PPO/discrete, REINFORCE/discrete
- `TargetBC` needs critic, discounted, next_action_target, q_critic: not on UDRL, DAgger
- `TargetBC` needs critic, discounted, next_action_target, q_critic, replay: not on APG
- `TargetBC` needs discounted: not on ASAC, RVISAC
- `TargetBC` needs discounted, next_action_target, q_critic, replay: not on APO
- `TargetBC` needs next_action_target: not on IQL
- `TargetBC` needs next_action_target, q_critic: not on DreamerV3
- `TargetBC` needs next_action_target, q_critic, replay: not on PPO, REINFORCE
- `TargetBC` needs replay: not on AVG
- `ValueBox` needs critic, discounted, off_policy, q_critic: not on APG
- `ValueBox` needs critic, discounted, q_critic: not on UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `ValueBox` needs discounted: not on ASAC, RVISAC
- `ValueBox` needs discounted, off_policy, q_critic: not on APO, APO/discrete
- `ValueBox` needs off_policy, q_critic: not on PPO, PPO/discrete, REINFORCE, REINFORCE/discrete
- `ValueBox` needs q_critic: not on DreamerV3, DreamerV3/discrete
- `ValueCentering` needs critic, discounted, one_step_target: not on APG, UDRL, UDRL/discrete, DAgger, DAgger/discrete
- `ValueCentering` needs discounted: not on ASAC, RVISAC
- `ValueCentering` needs discounted, one_step_target: not on APO, APO/discrete
- `ValueCentering` needs one_step_target: not on TDMPC2, PPO, PPO/discrete, REINFORCE, REINFORCE/discrete, DreamerV3, DreamerV3/discrete, PQN, TDMPC2MultiTask
- `memory` needs obs_networks: not on TDMPC2, DreamerV3, DreamerV3/discrete, TDMPC2MultiTask

## Not yet wired

- MCPretrain binds only on SAC: `CriticBlend` on TD3 REDQ CrossQ AVG DQN PQN IQL; `MCPretrain` on TD3 REDQ TQC CrossQ AVG PPO PPO/discrete DQN PQN IQL REINFORCE REINFORCE/discrete; `MCVarianceCorrection` on TD3 REDQ CrossQ IQL; `OnlineBC` on TD3 REDQ TQC CrossQ AVG IQL; `PhiRefresh` on TD3 REDQ TQC CrossQ DQN IQL; `ValueBox` on TD3 REDQ TQC CrossQ AVG DQN PQN IQL
- the agent binds no context for it yet: `BellmanPretrain` on ASAC RVISAC DQN IQL; `CloningPretrain` on APG PPO UDRL; `RepresentationProbe` on APG UDRL UDRL/discrete
- its collector steps the env itself, with no track_memory phase yet: `BumplessTracking` on APG UDRL UDRL/discrete
- no form for this agent yet (cross-entropy for discrete actors; a policy's encoder): `CloningPretrain` on PPO/discrete APO/discrete UDRL/discrete DAgger/discrete DreamerV3/discrete; `ImitationLoss` on PPO/discrete APO/discrete UDRL/discrete DAgger/discrete DreamerV3/discrete; `CriticDiagnostics` on UDRL UDRL/discrete DAgger DAgger/discrete
- world models fold no loss phase for extensions and build their own encoders: `CriticBlend` on TDMPC2 TDMPC2MultiTask; `IBRL` on TDMPC2 TDMPC2MultiTask; `KernelRepulsion` on TDMPC2 TDMPC2MultiTask; `LCBGatedBootstrap` on TDMPC2 TDMPC2MultiTask; `MCVarianceCorrection` on TDMPC2 TDMPC2MultiTask; `OnlineBC` on TDMPC2 TDMPC2MultiTask; `ValueBox` on TDMPC2 TDMPC2MultiTask; `ExpertValueMetrics` on TDMPC2 TDMPC2MultiTask; `ImitationLoss(q_filter=True)` on TDMPC2 TDMPC2MultiTask; `ImitationLoss(target='replay', alpha=2.5)` on TDMPC2 TDMPC2MultiTask; `ConservativeQ` on TDMPC2 TDMPC2MultiTask; `CalQL` on TDMPC2 TDMPC2MultiTask; `TargetBC` on TDMPC2 TDMPC2MultiTask; `JSRLCurriculum` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `MCPretrain` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `BiasVorePenalty` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `cnn` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `EDGEExploration` on TDMPC2 DreamerV3 TDMPC2MultiTask; `EpsilonGreedyPresample` on TDMPC2 DreamerV3 TDMPC2MultiTask; `BellmanPretrain` on TDMPC2 TDMPC2MultiTask; `CloningPretrain` on TDMPC2 DreamerV3 TDMPC2MultiTask; `ImitationLoss` on TDMPC2 DreamerV3 TDMPC2MultiTask
- imagination has no observation for the expert to read: inherent, refused as unwired: `ExpertActionInput` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `ExpertStateInput` on TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask
- it trains offline on logged observations, which hold no feature columns: inherent, refused as unwired: `TrainingProgressInput` on TDMPC2MultiTask; `PIDTermsInput` on TDMPC2MultiTask; `PreviousAction` on TDMPC2MultiTask; `PreviousReward` on TDMPC2MultiTask
- the cnn options are not on this agent yet: `cnn` on SAC TD3 REDQ TQC CrossQ ASAC RVISAC AVG APG APO APO/discrete DAgger DAgger/discrete IQL REINFORCE REINFORCE/discrete
- memory is not wired on this agent yet: `memory` on APO APO/discrete DQN PQN DAgger DAgger/discrete IQL REINFORCE REINFORCE/discrete
- memory on this agent has no published form to follow yet: `memory` on AVG CrossQ UDRL UDRL/discrete
- TD-MPC2's own collector does not run the choose_warmup_action phase in its seed phase yet, and it takes no expert_policy: `ExpertWarmup` on TDMPC2
- REINFORCE takes no expert and folds no loss phase for extensions yet: `BiasVorePenalty` on REINFORCE REINFORCE/discrete; `BumplessTracking` on REINFORCE REINFORCE/discrete; `CloningPretrain` on REINFORCE REINFORCE/discrete; `ExpertStateInput` on REINFORCE REINFORCE/discrete; `ImitationLoss` on REINFORCE REINFORCE/discrete
- APG's closed-loop rollout passes the features no reward yet: `PreviousReward` on APG
- only PPO folds the reward phase (GAIL's AdversarialReward): `AdversarialReward` on SAC TD3 REDQ TQC CrossQ ASAC RVISAC AVG APG APO UDRL DAgger IQL DreamerV3 REINFORCE TDMPC2 TDMPC2MultiTask
- the agent's own collector does not run choose_replay_action yet: `StorePolicyAction` on DreamerV3 DreamerV3/discrete TDMPC2 TDMPC2MultiTask
- the agent's own replay is not handed to its pretraining yet: `PriorData` on UDRL UDRL/discrete DAgger DAgger/discrete TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `ExpertPrefill` on UDRL UDRL/discrete DAgger DAgger/discrete TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `RLPD` on UDRL UDRL/discrete DAgger DAgger/discrete TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask; `RetainedDemonstrations` on UDRL UDRL/discrete DAgger DAgger/discrete TDMPC2 DreamerV3 DreamerV3/discrete TDMPC2MultiTask
