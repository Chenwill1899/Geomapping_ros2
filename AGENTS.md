# Project Memory Policy

For this repository, store and update persistent memory in the project-local OMX
memory file:

```text
.omx/project-memory.json
```

Do not add new Geomapping_ros2 memory notes to the global
`/home/mexxiie/.codex/memories` tree unless the user explicitly asks for a
global memory update.

Automatically update `.omx/project-memory.json` at the end of any substantial
Geomapping_ros2 task when the work creates durable project knowledge, such as a
new workflow, command, artifact path, validation result, failure mode, rollback
decision, tuning conclusion, or user preference. Keep updates concise and
evidence-grounded. Do not update memory for trivial chat, transient command
output, or facts that were not verified.

Before relying on remembered facts, re-check drift-prone state such as the active
branch, dirty worktree, ROS processes, dataset manifests, run artifacts, GPU
health, and current YAML/profile values.
