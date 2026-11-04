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
