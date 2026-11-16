# Repository workflow

- Unless the user explicitly requests another branch, develop on
  `codex/downstream-resume-progress`, not `main`.
- Fetch and merge the latest `origin/main` into that Codex branch before
  starting an update. Preserve local edits and existing history.
- Push development changes to the Codex branch. Merge/push `main` only when
  the user explicitly requests publication there.
- Keep ARC account paths and scheduler wrappers outside this public repository.
  Portable benchmark/plotting programs belong under `benchmarks/`, separate
  from installed numerical modules.
- Use focused checks for the changed behavior; avoid redundant validation.

- Read `PROJECT_STATE.md` first for accepted decisions and pending validation.
- `codex/publication-clean` is the minimal release branch; do not add research
  scripts there. `codex/downstream-resume-progress` retains supplementary tools.
- Keep compact project decisions in `PROJECT_STATE.md`; do not paste chat logs
  or server results into source files. Retain essential provenance externally.
