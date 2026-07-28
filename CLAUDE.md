# Bigtea — local MoE inference tooling

- Purpose: open-source research + engineering on local MoE inference. Builds on ktransformers / llama.cpp as backends — never from scratch.
- All project state lives in the knowledge graph: `/docs/graph/`. **Read `/docs/graph/INDEX.md` first each session** — then open only the 2–3 nodes the task links to, never the whole tree.
- Node types: `research/` (investigated topics), `decisions/` (ADRs), `backlog/` (epics → tickets). Nodes link by relative path; never paste content between files.
- Subagents (`.claude/agents/`): `researcher` (investigate → write node, return 3 lines), `planner` (graph → ticket), `coder` (one ticket only), `reviewer` (diff vs acceptance criteria). Delegate to them; keep the main session thin.
- Build / test / lint: **no code yet.** When code lands, record the exact commands here, one line each.
- Git: commit after each completed ticket — small, described commits. Model/weight files are gitignored; keep it that way.
- Token discipline: this file stays under ~2000 tokens. If it grows past that, tell Atur to prune it — do not let it bloat.

## Compact Instructions
If this session is auto-compacted, preserve ONLY:
- Open decisions (the question + live options)
- The backlog ticket currently in progress
- Files modified this session
- Unresolved questions for Atur

Discard everything else: tool output, file contents already committed, exploration dead ends.
