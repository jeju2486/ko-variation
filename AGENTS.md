# Repository workflow

- Unless the user explicitly requests another branch, develop on
  `codex/downstream-resume-progress`, not `main`.
- Fetch and merge the latest `origin/main` into that Codex branch before
  starting an update. Preserve local edits and existing history.
- Push development changes to the Codex branch. Merge/push `main` only when
  the user explicitly requests publication there.
- Keep ARC account paths and scheduler wrappers outside this public repository.
  Research benchmarks and diagnostics belong under `benchmarks/`; functional
  maps and paper plots are installed runtime tools.
- Use focused checks for the changed behavior; avoid redundant validation.

- Read `PROJECT_STATE.md` first: it summarizes the entire chat by importance, including scientific conclusions, rejected approaches and pending validation.
- `main` contains the minimal public release; do not add research
  scripts there. `codex/downstream-resume-progress` retains supplementary tools.
- Keep compact project decisions in `PROJECT_STATE.md`; do not paste chat logs
  or server results into source files. Retain essential provenance externally.
- When merging a minimal-publication `main` into development, preserve the
  development-only benchmark/diagnostic files from the pre-merge development
  commit. Publication cleanup deletions must not erase supplementary sources.
