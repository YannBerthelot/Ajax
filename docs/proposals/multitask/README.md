# Multi-task training: design plan (deferred)

Status: **not scheduled.** Yann (2026-10-12) put multi-task training outside the refactor's scope: "Leave the plan somewhere once its done, we'll implement this later. let's finish the refactor first." This branch keeps the plan. It is not merged and has no PR.

Decisions already taken:
- **Task suite:** MuJoCo Playground, its DM Control suite first, not brax. Brax's Ant, Hopper, Walker and Humanoid terminate early, and the reference suites were DM Control and Meta-World.
- **TDMPC2MultiTask stays purely offline,** as the paper's offline trainer.
- **Abstraction first:** find the right granularity before choosing which protocols to expose. Sequential (continual) task schedules must be covered as well as all-tasks-at-once.

Files:
- **`design.md`:** the design, revised by a critic. It covers the TaskSet plus Schedule abstraction, the agent contracts, 28 published methods placed in layers, the user API, PRs 1-7 with line budgets, and the owner decisions D1-D15.
- **`map.md`:** facts about Ajax's code and the installed Playground tasks, with file:line references.
- **`lit_online.md`, `lit_sota.md`, `lit_continual.md`:** the literature, read from papers and official code with citations. Some sections went through a summariser; the files say which.
