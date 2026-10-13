# Environments and wrappers

Agents take a gymnax, Brax or MuJoCo Playground environment, or its id, and
wrap it themselves. Every continuous agent acts in `[-1, 1]`, and
`agent_action_to_env` maps its action onto the environment's bounds; an
expert speaks the environment's units, and the agent reads it through
`ExpertInAgentUnits`. See [Environments](../guide/environments.md) and
[Differentiable simulation](../guide/differentiable-simulation.md).

::: ajax.environments.create

::: ajax.wrappers

::: ajax.environments.obs_norm

::: ajax.environments.system_class

::: ajax.environments.model_reference

::: ajax.environments.pid_terms

::: ajax.environments.differentiable

## Action units

::: ajax.environments.utils.agent_action_to_env

::: ajax.environments.utils.env_action_to_agent

::: ajax.environments.utils.ExpertInAgentUnits

## Running a trained policy

::: ajax.environments.interaction.get_pi
