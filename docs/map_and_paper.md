# Functional maps and publication figures from completed discovery

This workflow reads v0.9 `associations.tsv.gz` and `run_summary.json`. It does
not run score tests, estimate new beta values, use phenotype labels to construct
the network, or generate QQ plots. It reuses discovery's significance decisions
and saved post-hoc beta/status fields.

## Install

Use the development branch `codex/downstream-resume-progress` and install its
optional network dependencies:

```bash
python -m pip install -e '.[network]'
```

For **new unitig-to-gene annotation**, the existing mapping helper additionally
requires Pyseer, BWA, bedtools and BEDOPS (`gff2bed`). Set `--pyseer-python` if
Pyseer is installed in another environment. Its Python's executable directory
is prepended to the mapping subprocess PATH, and inherited PYTHONPATH is removed;
the same environment supplies Pyseer and its native tools. All dependencies are
checked together before reference preparation. Existing completed annotation
can still be reused without running Pyseer again.
Supply original DNA unitigs, a sample-ID binary FASTA, the Panaroo membership
table and Bakta `SAMPLE/SAMPLE.gff3` and `SAMPLE.fna` files. Do not map binary
A/C strings as if they were unitig DNA. See [annotation details](downstream.md).

For **new functional assignment**, provide matching Bakta `SAMPLE/SAMPLE.faa`
proteins, eggNOG-mapper and its compatible local database. The mapper/database
are installed separately. The workflow does not download databases. Existing
gene mapping and eggNOG results can be supplied instead of rerunning those tools.

For a separate, reproducible mapping environment:

```bash
conda create -y -p /project/envs/pyseer \
  -c conda-forge -c bioconda \
  python=3.9 pyseer=1.4.2 bwa bedtools bedops
```

Pass `--pyseer-python /project/envs/pyseer/bin/python` to the workflow. Keeping
this separate avoids Pyseer's pinned NumPy/SciPy requirements changing KOVAR's
runtime. `gff2bed` is supplied by **BEDOPS**, a separate package from BEDTools.
See [Pyseer installation](https://pyseer.readthedocs.io/en/master/installation.html)
and the [Bioconda recipe](https://github.com/bioconda/bioconda-recipes/blob/master/recipes/pyseer/meta.yaml).

To check that environment, optionally including a real 61-bp exact-match mapping:

```bash
python -c 'import json; from ko_variation.annotation_inputs import annotation_environment; print(json.dumps(annotation_environment("/project/envs/pyseer/bin/python", smoke_test=True)[1], indent=2))'
```

The smoke test runs actual Pyseer, BWA, `gff2bed` and BEDTools on a temporary
synthetic reference, checks the reported coordinates and gene name, then removes
only its temporary files. It does not fit KOVAR or modify biological inputs.
Successful production annotation records the interpreter, package versions and
native executable paths in `annotation/annotation_summary.json`.

The portable runner adds a combined preflight for mapping, eggNOG/DIAMOND,
eggNOG v2 database-file availability and plotting libraries:

```bash
ROOT=/project CHECK_ONLY=1 SMOKE_TEST=1 \
PYSEER_PYTHON=/project/envs/pyseer/bin/python \
EGGNOG_EXEC=/env/eggnog/bin/emapper.py EGGNOG_DATA_DIR=/databases/eggnog \
bash benchmarks/run_functional_map.sh saureus
```

`ROOT` has no machine-specific default. The conventional discovery input is
`ROOT/kovar_output/SPECIES/discovery/results`; override `RESULTS` for another
layout. `RUN_NAME` is optional: an explicit label inserts that subdirectory
between `SPECIES` and `discovery`/`plots`. The machine that ran discovery is not
used to select input paths. `MAP_OUT`, `PANAROO`, `BAKTA`, `GFF_DIR`, `UNITIGS` and `FASTA`
override inputs/outputs. The runner prefers original Bakta inputs, then retained
Panaroo GFFs, under the species's upstream output. It selects an existing
`ROOT/pyseer_annotation_env/bin/python` unless `PYSEER_PYTHON` overrides it.
No environments or databases are installed implicitly.

For a copied run, the runner finds a unique `*sample_ids.fasta` or
`*sample_ids.fa` in `discovery/prepared_inputs/` (also `results/prepared_inputs/`).
It passes that local file explicitly, so an old absolute path in
`run_identity.json` does not need to be rewritten. Multiple candidates require
`FASTA=/exact/path`; no candidate leaves the original run-identity lookup in
place. `RESULTS` accepts the discovery parent, results directory or association
file, as does the Python helper. An explicit `FASTA` always takes precedence.
The generated figures default to `ROOT/kovar_output/SPECIES/plots/functional_map`;
use a distinct `MAP_OUT` when plotting different discovery datasets separately.

`CHECK_ONLY=1` stops after the report; remove it to run the map. `SMOKE_TEST=1`
additionally exercises real native mapping and PNG/SVG export on tiny temporary
fixtures; ordinary runs perform dependency checks only. The combined report is
written to `MAP_OUT/map_environment.json`. Checks confirm executable/import/file
availability and the synthetic mapping/export contracts; they do not validate
the full eggNOG database contents or run DIAMOND annotation against it.

## 1. Functional network from existing annotation

```bash
python benchmarks/functional_map.py \
  --results /project/discovery/results \
  --annotation /project/annotation/locus_to_gene.tsv \
  --gene-catalog /project/annotation/gene_catalog.tsv \
  --eggnog /project/functional/network.emapper.annotations \
  --out /project/plots/functional_map
```

Run this script from the source checkout. `--results` accepts the results
directory, a parent containing `results/`, or the association-file path. The
adjacent `run_summary.json` supplies the original Bonferroni family/cutoff.
`--annotation` needs one resolved stable gene/cluster ID per zero-based locus;
duplicate locus IDs are rejected. Display names do not replace stable join IDs.
If `--gene-catalog` is omitted, a catalogue beside the annotation is used when
present. eggNOG query IDs must match those same stable cluster IDs.

For mapping from scratch, replace `--annotation` with:

```bash
--unitigs /project/pangwes/graph.unitigs \
--panaroo /project/panaroo/gene_presence_absence_roary.csv \
--bakta /project/assemblies
```

These are arguments to the command above, not a separate shell command. The
binary sample-ID FASTA is inferred from discovery's `run_identity.json`, or
supplied explicitly with `--fasta`. The helper reuses a matching completed
annotation cache. `--nearby-bp 500` is the default strand-aware gene-assignment
window; it is separate from the **pairwise distal exclusion** of 10,000 bp.
Ambiguous mappings remain in annotation audit files and are excluded from the
resolved gene map; they are never assigned arbitrarily to the first gene.

### Recover missing annotation files from Panaroo GFFs

When the original Bakta files were not copied, use `--gff-dir` instead of
`--bakta`. It requires `SAMPLE.gff` (or `.gff3`) filenames matching Panaroo
sample columns, with CDS records and an embedded `##FASTA` assembly section:

```bash
python benchmarks/functional_map.py \
  --results /project/discovery/results \
  --unitigs /project/pangwes/graph.unitigs \
  --panaroo /project/panaroo/gene_presence_absence_roary.csv \
  --gff-dir /project/panaroo_gff \
  --run-eggnog --emapper /env/eggnog/bin/emapper.py \
  --eggnog-data-dir /databases/eggnog \
  --out /project/plots/functional_map --threads 4
```

Recovery preserves original CDS IDs, coordinates, attributes and assembly DNA.
It creates the expected `SAMPLE/SAMPLE.gff3`, `.fna`, `.faa` layout under
`recovered_annotations/`. Derived proteins are **not** claimed to reproduce the
original Bakta protein files. No gene calling, Panaroo rerun or KOVAR refit occurs.

For protein derivation, the default genetic code is bacterial table 11
(`--translation-table` changes this fallback); a CDS's explicit `transl_table`
or `translation_table` takes precedence. Only unambiguous, single-record,
zero-phase complete CDSs with valid start/terminal-stop structure are translated.
Strand orientation and alternative start codons are handled by Biopython's
[`translate(cds=True)`](https://biopython.org/docs/latest/api/Bio.Seq.html).
Pseudogenes, partial CDSs, duplicated/multipart IDs, exceptional translations
and failed complete-CDS checks receive no derived protein. Their original CDS
records remain available for unitig annotation. This intentionally conservative
protein subset can leave additional nodes with unassigned function; it does
not filter statistical network edges.

Each sample has `cds_recovery.tsv.gz`, `recovery_summary.json` and a completed
cache manifest. The combined `recovered_annotations/recovery_summary.json`
records counts and input identities. Recovery processes one assembly at a time;
reruns reuse matching completed samples. Sources are never overwritten. CDS
coordinates/phase follow the [GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md).

## 2. Run or resume eggNOG only when needed

Instead of `--eggnog`, pass these arguments with the mapping inputs:

```bash
--run-eggnog --emapper /env/eggnog/bin/emapper.py \
--eggnog-data-dir /databases/eggnog --threads 4
```

Representative proteins use the existing deterministic Panaroo/Bakta matching
rule. `protein_manifest.tsv` records the selected protein, source sample/ID,
sequence hash and missing matches. A representative's predicted function does
not establish every member/allele's function.

Protein preparation is cached by matching source signatures and output hashes.
eggNOG uses a run directory identified by protein content, executable/version
and database file inventory. Database inventory records paths, sizes and mtimes;
it is not a full checksum of very large database files. Completed matching
annotation is reused. An interrupted identical run invokes the tool's
`--resume` when that executable advertises it; otherwise the workflow reports
the unfinished directory instead of silently marking it complete. CPU count is
recorded in the command and can change without discarding completed annotations.
Inspect `functional/eggnog_*/emapper.log` for external-tool failures.

`--threads` controls eggNOG CPUs. Native mapping tools retain the existing
annotation helper's execution behaviour; this option does not add new parallel
annotation/refitting implementations. Explicit `--allow-unassigned` skips
functional assignment; it does not invent missing categories.

## Selection, aggregation and memory

The workflow streams association chunks (`--chunk-rows 50000` by default).
Candidates must be source-certified significant, source-classified `DISTAL`,
and strictly beyond `--distal-bp` (10,000 bp by default). Local and unknown
distances remain in discovery's source exports. The source `p_upper` must
certify the recorded Bonferroni cutoff. No exact P-value or new gene-level test
is constructed.

Mapped support is indexed on disk in SQLite with file-backed temporary storage.
One unordered gene edge summarizes all resolved, distinct supporting unitig
pairs. The representative is the smallest recorded `p_upper` among successful
beta fits, ties by unitig IDs; if none succeeded, its beta is unavailable. Mixed
coefficient signs are recorded. Failed/unestimated beta fits remain significant
edges with an explicit unavailable magnitude. Same-gene pairs are excluded from
this gene-to-gene map.

The complete support list is stored in `gene_pair_support.tsv.gz` rather than
duplicated into every HTML/JSON edge. Its stable IDs and unitig IDs allow the
gene summaries to be traced back to original associations. The in-memory display
object is the collapsed gene graph, so an extremely large **gene** graph still
needs sufficient memory; streaming does not make graph rendering memory-free.
The existing mapping helper also reads the selected locus set when annotating.

## Functional-map outputs

| File | Meaning |
|---|---|
| `selected_distal.tsv` | Certified distal unitig associations and saved effects |
| `selection_counts.json` | Input/selected pairs, distance and beta-status counts, original significance family |
| `annotation/` | New resolved mapping/catalogue and ambiguity/status audit when mapping was requested |
| `recovered_annotations/` | Optional GFF/DNA recovery, derived proteins and per-CDS/sample audit |
| `genes.nodes.tsv`, `genes.edges.tsv` | Collapsed biological summaries before display enrichment |
| `gene_pair_support.tsv.gz` | Complete resolved intergene unitig support, including unsuccessful beta statuses |
| `aggregation_counts.json` | Missing annotation, same-gene exclusion, unitig/gene/edge counts |
| `functional/` | Representative proteins and optional cached eggNOG execution |
| `map.html` | Offline searchable D3 covariation map |
| `map.png`, `map.svg` | Complete static maps, including two-gene components |
| `map.nodes.tsv`, `map.edges.tsv`, `map.layout.json` | Functional labels/colours, edge summaries and saved coordinates |
| `map_audit.json` | Combined counts, input fingerprints and outputs; refitting is false |
| `map_work.sqlite`, `*.cache.json` | Downstream aggregation work and completed-stage cache records |

Unassigned function is grey. Node identity remains the stable cluster ID;
annotation aliases and products supply display/search names. Shared names do
not merge clusters. Nearby-only assignments retain their `near` qualifier.
COG bins describe broad predicted functions; GO terms remain searchable details.
Functional assignment never adds/deletes a selected edge. See
[category definitions](functional_map.md#display-categories).

## 3. Generate publication PNG/SVG figures

```bash
python benchmarks/paper_plots.py \
  --layout /project/plots/functional_map/map.layout.json \
  --out /project/plots/paper --focus mprF --dpi 300
```

Run this script from the source checkout. It reads only the saved map
layout, without annotation, eggNOG, KOVAR fitting or a database download.
Repeat `--focus` for additional exact aliases or stable cluster IDs.

- `network_overview[ PAGE].png/svg`: components with at least three genes.
  Omitted dimers are summarized as `+N two-gene components`; isolated nodes,
  if present, are counted separately. The complete map/tables retain them.
- `focus_NAME_N1.png/svg`: focal cluster(s) and direct network neighbours.
- `focus_NAME_N2.png/svg`: focal cluster(s), direct and second-order neighbours.
- `component_sizes.png/svg`: component-size counts for the complete graph.
- `functional_composition.png/svg`: functional-category counts for the complete
  graph, including dimers omitted from the overview.
- `figure_captions.txt`, `paper_figures.json`: factual counts, display settings,
  target presence, input fingerprint and output inventory.

Missing focal genes are reported as `NOT_IN_SELECTED_NETWORK`; no target or
neighbourhood is fabricated. A missing selected-network node is not evidence
that the gene is absent from the collection. Multiple clusters with the same
alias are all retained and reported by stable ID.

Labels occupy separate margins with leader lines and measured wrapping; names
are not truncated. Named nodes are larger than unnamed nodes, with a common
size reduction for dense panels. Red outlines/text identify focal clusters.
Functional fills remain unchanged. Solid/dashed/dash-dot edges indicate
positive/negative/mixed coefficient signs; dotted edges have no fitted beta.
Widths inherit the source map's within-map magnitude scaling, so do not compare
stroke widths across species as a common absolute beta scale.

The generator uses saved coordinates, paginates disconnected components
(`--components-per-page 6`), and writes vector SVG with editable text. PNG
resolution is controlled by `--dpi`; default 300. No embedded figure titles or
QQ plots are generated. All named labels can require a tall canvas for very
large components; use the focal plots or supplementary overview pages when
assembling a journal-sized figure. The renderer does not guarantee that a whole
large network fits legibly into one small journal panel.

Gene edges describe genomic association summaries, not causality, molecular
interaction or demonstrated epistasis. Neighbour order is network shortest-path
distance, not genomic distance. No MIC prediction is run by either plotting
workflow.
