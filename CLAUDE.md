# Bigtea — local MoE inference tooling

- Purpose: open-source research + engineering on local MoE inference. Builds on ktransformers / llama.cpp as backends — never from scratch.
- All project state lives in the knowledge graph: `/docs/graph/`. **Read `/docs/graph/INDEX.md` first each session** — then open only the 2–3 nodes the task links to, never the whole tree.
- Node types: `research/` (investigated topics), `decisions/` (ADRs), `backlog/` (epics → tickets). Nodes link by relative path; never paste content between files.
- Subagents (`.claude/agents/`): `researcher` (investigate → write node, return 3 lines), `planner` (graph → ticket), `coder` (one ticket only), `reviewer` (diff vs acceptance criteria). Delegate to them; keep the main session thin.
- Test: `python -m unittest discover -s tests -v` (from repo root).
- Run: `python -m bigtea --help` (stdlib only — no install needed; `pip install -e .` gives the `bigtea` entry point).
- Lint: none yet.
- Git: remote = `github.com/aturzone/Bigtea`. Push with the token from `C:\Projects\.env` inline in the push URL, output redacted — never store it in git config, never echo it. Commit after each completed ticket. Model/weight files are gitignored; keep it that way.
- Every backlog ticket has a GitHub issue (`[epic] Tn: title`; body = summary + acceptance + graph links, never full content). Planner creates the issue when it creates the ticket; the `## Issues` section in each epic maps Tn → issue #.
- Implementation work (coder) happens on branch `ticket/<epic>-Tn` + PR referencing the issue; reviewer checks the PR before Atur merges. **Never direct-to-main once implementation starts.** Docs/research/decision nodes may still commit to main.
- Sync audit at phase boundaries only (before a session restart, before merging a PR, before starting a new epic) — local vs remote branches/commits, issue states vs epic checkboxes, PR list vs rules. Fix drift, don't just note it. Not per-commit; keep it a checkpoint, not overhead.
- Token discipline: this file stays under ~2000 tokens. If it grows past that, tell Atur to prune it — do not let it bloat.

## Compact Instructions
If this session is auto-compacted, preserve ONLY:
- Open decisions (the question + live options)
- The backlog ticket currently in progress
- Files modified this session
- Unresolved questions for Atur

Discard everything else: tool output, file contents already committed, exploration dead ends.
