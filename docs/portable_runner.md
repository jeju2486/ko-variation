# Portable runner: installation, inputs, running and restart

[`examples/run_discovery.sh`](../examples/run_discovery.sh) runs the installed
KOVAR discovery pipeline from Bash. Use it on a workstation or Linux server
without ARC paths, Slurm directives or any previous KOVAR results. Commands in
this guide assume Bash; replace the example data/tool paths with your own paths.
Keep filenames containing spaces quoted.

## 1. Requirements and installation

Requirements: Git, Bash, Python 3.10+ and enough persistent storage for the
prepared patterns, null caches and SQLite checkpoint. For genomic distance and
post-hoc beta calculation, also provide the separately installed PAN-GWES
`unitig_distance` executable and its matching graph files. Pip does not install
that external executable.

Create an isolated environment and install from the development branch:

```bash
git clone --branch codex/downstream-resume-progress https://github.com/jeju2486/ko-variation.git
cd ko-variation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m ko_variation.discovery --version
bash examples/run_discovery.sh --help
```

If you already use a suitable conda/virtual environment, activate it and run
`python -m pip install .` there instead of creating `.venv`. Optional downstream
network/phenotype dependencies can be installed with:

```bash
python -m pip install '.[network,phenotype]'
```

The portable runner is part of the source checkout; pip installs the Python
package and console commands. Keep the checkout to use the example runner.

## 2. Required input files

| Argument | Input and required correspondence |
|---|---|
| `--fasta` | Sample-labelled binary pseudo-FASTA: A=absence, C=presence; one aligned column per original unitig |
| `--tree` | Newick phylogenetic tree with branch lengths and isolate IDs matching the FASTA |
| `--unitigs` | Original PAN-GWES zero-based unitig-ID/sequence file, needed for distances |
| `--paths` | PAN-GWES graph-path list whose referenced files still exist, needed for distances |
| `--unitig-distance` | Executable name on PATH, or absolute executable path |
| `--edges` | Original graph edges when required by the installed distance-tool version |

FASTA headers must have unique isolate IDs. Raw pseudo-FASTA records repeatedly
named `contig00001` need their verified isolate IDs assigned before using this
runner. This wrapper does not infer isolate order from raw graph files. Preserve
the original column order: column zero must correspond to unitig ID zero, and
so on. Do not rename/reorder columns independently of the graph IDs.

Tree tips must identify the corresponding isolates. The wrapper does not root
or edit the supplied tree. The default tree-missing-sample policy is `error`;
use `--tree-missing-samples drop` only when excluding FASTA isolates absent from
the tree is your intended cohort policy. `--exclude-samples exclusions.txt`
explicitly removes listed isolate IDs; the file contains one ID per line.

Inspect your graph-tool interface with:

```bash
/tools/PAN-GWES/unitig_distance --help
```

If `.paths` entries are relative, start KOVAR from the directory where those
entries resolve. Their base is the process working directory, not automatically
the `.paths` file's parent. Alternatively, supply a graph-path list containing
absolute paths. Keep this convention unchanged on restart.

## 3. First run

From the checkout, with the environment activated:

```bash
bash examples/run_discovery.sh \
  --fasta /data/project/graph.sample_ids.fasta \
  --tree /data/project/core.treefile \
  --unitigs /data/project/graph.unitigs \
  --paths /data/project/graph.paths \
  --unitig-distance /tools/PAN-GWES/unitig_distance \
  --kmer-length 61 \
  --out /data/project/kovar_output/saureus/discovery \
  --threads 4
```

The initial `--out` must be empty or absent. The pipeline creates its own
preparation, numerical caches and checkpoint under this output. The species
component is a directory label; use `neisseria`, `klebsiella` or another label
for another collection, with that collection's own inputs and output.

For older distance tools, append `--edges /data/project/graph.edges`. Use the
k-mer length actually used to create your graph; `61` is the example/default.

To launch from the PAN-GWES graph directory when its paths are relative, first
save the runner location and then change directory:

```bash
KOVAR_RUNNER="$PWD/examples/run_discovery.sh"
cd /data/project/pangwes
# Use bash "$KOVAR_RUNNER" followed by the same absolute input/output arguments.
```

Select another installed environment without activating it by prefixing the
run command with `KOVAR_PYTHON=/path/to/env/bin/python`. The wrapper's default
interpreter is `python3` on PATH.

## 4. CPU, memory and default statistical settings

`--threads 4` is the total worker CPU budget. The wrapper sets one native BLAS
thread per worker; KOVAR controls its own numerical stages. `--memory-gb 8`
optionally caps the memory plan in GiB. Without it, KOVAR detects allocation
limits with headroom, with a conservative fallback when no usable limit is
available. Disk-backed processing still requires sufficient free disk space.

| Setting | Default |
|---|---|
| Minor-state frequency | `--min-maf 0.05` |
| Minimum final joint-state cell count | `--min-cell-count 20` |
| Tree-PC coarse screen | `--screen-pcs 40` |
| Target coarse-pair retention | `--coarse-retain 0.10` |
| Bonferroni family-wise alpha | `--alpha 0.05`, across all unique-pattern pairs |
| Hierarchy exploration gate | `--screen-p 0.001`, separate from final significance |
| Distal classification | `--distal-bp 10000`: distal means strictly greater than 10,000 bp |

PC/representative candidate selection is heuristic. Statistical significance
selects candidate associations; beta is fitted afterwards for selected distal
pairs. Significant local pairs remain in the table with beta skipped. Exact
P-values can be absent when bounds certify the significance decision.

All options from `python -m ko_variation.discovery --help` pass through this
runner. It changes no numerical or scientific defaults. For a run without
distance/beta yet, supply just `--fasta`, `--tree`, `--out` and resource options;
matching graph-distance inputs can be added on a compatible restart.

## 5. Logging and progress

Keep shell logs outside a new output directory. To record the first-run command
above, append `2>&1 | tee /data/project/logs/saureus.log` after its final argument,
after creating the log directory. In Bash, `set -o pipefail` preserves the
pipeline failure status when using `tee`.

Search progress reports queried pattern pairs and hierarchy counters. Those
counts are not interchangeable completion percentages. During beta fitting,
the log and `OUT/beta_progress.json` report cumulative saved/remaining pattern
pairs, new fits this invocation, OK/non-OK statuses and a rough ETA.

```bash
cat /data/project/kovar_output/saureus/discovery/beta_progress.json
```

That file is created when the beta stage begins and may lag the last SQLite
commit between progress updates. Saved means committed attempts, including
non-OK fits. It does not mean all saved attempts produced a valid beta.

## 6. Restart after interruption

Repeat the same run command. The wrapper automatically adds `--resume` when
it finds `run_identity.json` in `--out`, or in an explicitly supplied persistent
`--tmp-dir`. An explicit `--resume` is also accepted. KOVAR checks compatibility
and reuses completed preparation, fitted nulls, decisions, distance batches and
beta fits. In-flight/uncommitted work can repeat; committed beta pairs are
excluded from the pending queue. No separate beta-only command is required.

Retain the same scientific inputs/settings, working-directory convention and
numerical environment. CPU count and memory cap can change. Unlike the ARC
wrapper, this thin portable wrapper forwards your command and does not restore
omitted scientific arguments from a saved manifest: repeat those arguments.

If you use scratch storage, explicitly supply a persistent location from the
first run and every restart:

```bash
# Append to the first-run command and repeat on every restart:
--tmp-dir /persistent-scratch/project/saureus_work
```

Do not substitute a new scratch location mid-run. Keep the work directory and
any live SQLite WAL/SHM files intact; do not run two coordinators for the same
checkpoint. If you explicitly borrowed null caches from an older run, those
source directories remain dependencies until the required caches are made
independent. Fresh runs do not require any previous run or benchmark results.

## 7. Outputs

| File/directory | Contents |
|---|---|
| `unitigs.tsv.gz` | Original unitig IDs, unique-pattern assignments and state counts |
| `associations.tsv.gz` | Significant candidates, exact/bounded P-value fields, distance and beta/status fields |
| `run_summary.json` | Hypothesis family/cutoff, counts, resources and run provenance |
| `run_identity.json` | Checkpoint identity and persistent work location |
| `beta_progress.json` | Beta-stage cumulative saved/remaining and fitting statuses |
| `.kovar_work/` | Durable preparation, models, SQLite decisions/distances/beta; stored at `--tmp-dir` if supplied |

Final exports are generated after the pipeline reaches export. During a long
run, checkpoint data can exist before the final TSVs appear. Keep checkpoint
files if you need to resume or reuse models; final TSVs alone are not a complete
restart state. See [discovery](discovery.md) for stage/storage details.

## 8. Independent restart from the beginning

For an independent analysis, choose a new empty output and, if specified, a
new empty persistent work directory. Do not pass `--reuse-prepared-from` or
`--reuse-nulls-from` if you want all preparation/fitting performed afresh. The
portable runner does not import previous runs automatically. If deleting an
old checkpoint, stop jobs using it first and preserve any benchmark results
or borrowed caches still needed by another run.

## 9. Updating the checkout

Stop jobs using the code/environment before changing them. Development updates
use the Codex branch:

```bash
git fetch origin
git switch codex/downstream-resume-progress
git pull --ff-only origin codex/downstream-resume-progress
python -m pip install .
```

Keep the NumPy/SciPy versions required by a saved checkpoint. A code/environment
update is not automatically compatible with every old run; KOVAR accepts only
matching or explicitly supported checkpoint migrations. It rejects mismatches
instead of silently combining different analyses.

For reproducibility, record the checkout revision and installed environment
outside the discovery output before starting:

```bash
git rev-parse HEAD > /data/project/kovar_commit.txt
python -m pip freeze > /data/project/kovar_environment.txt
```

Also record the external PAN-GWES build/version and executable checksum used
for distances, plus the input filenames/checksums and complete run command.

## 10. Common errors

| Message/symptom | What to check |
|---|---|
| `No module named ko_variation` | Install KOVAR using the Python selected by `KOVAR_PYTHON`; activating another environment changes which Python runs |
| Duplicate FASTA sample IDs | Relabel raw pseudo-FASTA using verified isolate order; repeated contig names are not isolate IDs |
| Tree/sample mismatch | Match IDs or supply an explicit exclusion/drop policy; do not guess sample order |
| Nonempty output is not a discovery run | Use an empty output; keep preparation/log files outside it before the first start |
| Work-directory mismatch | Restore the original `--tmp-dir` recorded in `run_identity.json` |
| Input/code/statistical settings changed | Compare the original command, inputs, installed versions and checkout; use a new output for a changed analysis |
| Distance executable not found | Supply its executable path and check `--help`; pip does not install it |
| Graph files not found | Check `.paths` entries, working directory, original graph files and whether your distance tool requires `--edges` |
| Another coordinator is using the work directory | Find the existing process/job; only one coordinator may write that checkpoint |
| No final TSV yet | Check the current stage/progress and saved work; final export follows the long-running computational stages |
