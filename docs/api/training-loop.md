# Training loop

The agents share one loop, `TrainLoop`: it collects experience, waits for
`learning_starts`, folds the extensions, evaluates, logs and resumes the same
way for every agent. An agent supplies only its `init` and `update`.
[Architecture](../developer/architecture.md) explains how the pieces fit.

::: ajax.agents.loop

The steps an agent's update is written in:

::: ajax.agents.update_steps

::: ajax.perf_utils

::: ajax.schedule

## Agent state and configuration

::: ajax.state.BaseAgentState
    options:
      heading_level: 3

::: ajax.state.BaseAgentConfig
    options:
      heading_level: 3

::: ajax.state.NetworkConfig
    options:
      heading_level: 3

::: ajax.state.OptimizerConfig
    options:
      heading_level: 3

::: ajax.state.EnvironmentConfig
    options:
      heading_level: 3
