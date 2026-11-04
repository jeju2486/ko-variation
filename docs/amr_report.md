# Species-specific AMR report

`ko-variation-amr-report` links recorded antimicrobial phenotypes to established
presence/absence determinant **annotation labels** and their representation in an
existing selected distal covariation map. It reads files only: no scanner rerun,
annotation rerun, model refit, resistance caller or phenotype prediction.

## Run

After updating the development checkout, register the helper with
`pip install -e .`. No network plotting extra or external executable is needed
for reporting. Run one command for each species with its own files:

```bash
ko-variation-amr-report --species saureus \
  --metadata saureus_collection_metadatatxt.txt \
  --gene-catalog covariation_map/annotation/gene_catalog.tsv \
  --annotation-status covariation_map/annotation/annotation_status.tsv \
  --network covariation_map/saureus_gene_covariation.html \
  --out amr_reports/saureus
```

Use `--species klebsiella` or `--species neisseria` for the other collections.
Full scientific species names are also accepted. `--network` accepts the helper's
offline HTML or `.layout.json`; HTML JavaScript is never executed. The full gene
catalogue is important: network nodes alone cannot establish annotation absence.
Omit unavailable optional inputs to get phenotype availability with explicit
`not_checked` statuses. Inputs must describe the same species and analysis.

The same command is available as `python -m ko_variation.amr_report`.
An old installed `ko-variation-trace-tet` launcher may need reinstalling to remove
it; the tet-specific module and its documentation were removed.

## Outputs

All files share the output prefix:

| File | Content |
| --- | --- |
| `.phenotypes.tsv` | All recorded drugs, quantitative/categorical counts, exact/limited MIC counts, missing qualifiers, source conflicts and distinct exact values |
| `.determinants.tsv` | Main species–drug–determinant report: MIC counts, annotation/map statuses, N1/N2 counts, next action and evidence reference |
| `.matches.tsv` | Every matching annotation label and cluster ID, including rejected product homonyms and unverified products; coding/nearby/ambiguous audit counts when supplied |
| `.neighbours.tsv` | N1 and N2 members by annotation label, with internal cluster IDs for traceability |
| `.todo.tsv` | Follow-up for every screened case, including mutation calling and uncovered catalogue entries |
| `.summary.json` | Input SHA-256 hashes, helper version, cohort counts, catalogue identity, network metadata and interpretation rules |

Tables retain negative and deferred cases. `NETWORK_CONTEXT_AVAILABLE` marks a
candidate for follow-up, **not** a prediction-ready phenotype. No MIC sample-size
cutoff, power claim, causal interpretation or mechanistic epistasis claim is made.
Assess sample size, exact phenotype variability, censoring, units and determinant
state variability in genome-matched isolates before modelling. Construct the full
established-determinant baseline independently of network membership; do not drop
baseline determinants merely because they lack selected distal neighbours.

## Annotation-name matching

Matching uses the `label` column, never the Panaroo `gene`/`group_####` identifier.
It splits semicolon and `~~~` aliases, ignores case and nomenclature punctuation,
and matches complete normalized tokens: `tet(M)` matches `tetM`, but `tetR` and
`tet38` do not. All clusters with the same biological label are retained. Cluster
IDs only join the catalogue, audit and network; they never supply a biological
label when annotation is missing. `group_####` fallback tokens are excluded from
matching and displayed as `unannotated` in neighbour labels, with the original
cluster ID retained only in the supporting ID column.

Catalogue product rules reject known homonyms. For example, `mecA` annotated as
an adaptor protein is rejected, while the resistance determinant requires a
PBP2a/penicillin-binding/transpeptidase product. A missing required product remains
`product_unverified`. Even an accepted annotation match needs sequence/genotype
verification; the helper does not infer gene integrity or resistance from a label.
Nearby assignments remain distinct from coding overlap, and ambiguous evidence
does not establish a resolved network assignment.

N1 is the union of direct neighbours of every accepted target cluster, excluding
target clusters. N2 is the union of neighbours of N1, excluding targets and N1.
Each cluster is counted once per order. N2 means two steps in the selected graph;
it does not imply a directly significant edge or a physical distance from the
target. The helper uses the supplied map unchanged. Preserve and verify its
original significance and >10 kb selection provenance upstream.

## Metadata and cohort

Metadata is a tab-delimited collection export with `id`, `species`, `*_mic`,
`*_mic_sign` and optional categorical `*_SIR` fields. Override `id` with
`--id-column`. IDs must be unique. Other species are excluded and counted in
provenance. The helper also reads valid drug/MIC/SIR JSON dictionaries in
`comments`. It counts one isolate per drug, prefers dedicated MIC fields, and
flags disagreement with embedded MIC records. Free-text comments are not mined.

Only finite positive numeric MICs count as quantitative records. `=` counts as
exact; `<`, `<=`, `>`, `>=` count as limits. Inline qualifiers are accepted.
Missing/unknown signs remain separately reported, without assuming exactness.
Conflicting inline/column qualifiers are excluded from positive numeric counts.
No bound is imputed or log-transformed. Categorical standards are recorded
separately by field name and do not become quantitative MICs. Units are not
assumed from a numeric value.

By default, counts describe species-matched metadata records, not the scanner
cohort. To restrict counts to the scanner's retained isolates, supply both:

```bash
  --sample-map metadata_to_sample.tsv --sample-inclusion results/sample_inclusion.tsv
```

The explicit one-to-one mapping TSV has columns `metadata_id` and `sample`;
scanner inclusion has `sample` and `status`, with `included` selecting retained
isolates. No naming convention is guessed. Unmapped metadata records are counted
in provenance; no overlap stops the report with an error. Counts still do not
establish determinant-state variation in those isolates.

## Curated starter catalogue and extension

[Bundled TSV](../ko_variation/data/amr_determinants.tsv), catalogue revision
2026-10-09, is a manually curated **starter**, not a complete CARD export or a
complete resistance baseline. Each mapping contains a primary publication or
official CARD evidence URL. The exact shipped catalogue hash is recorded in every
report; no live database download changes a run. Inspect the notes before using a
mapping. Presence of core `gyrA`, `grlA`/`parC` or `penA` is not treated as a
resistance allele; supported mutation-dependent cases are reported as deferred.
Unknown drugs receive `CATALOGUE_NOT_COVERED`, not a false claim of no determinants.

The starter includes S. aureus acquired mecA, blaZ, tet, erm, fus, mup, dfr,
aminoglycoside and cfr candidates; selected K. pneumoniae tet, carbapenemase,
CTX-M-15 and qnr candidates; and N. gonorrhoeae tetM and TEM penicillin candidates.
Coverage remains incomplete even for these phenotypes. For instance, specific
KPC-2/NDM-1/CTX-M-15 labels do not cover all beta-lactamases, and a trimethoprim
determinant does not alone explain the trimethoprim/sulfamethoxazole combination.
The report never extrapolates a whole drug class to every drug: tet presence does
not automatically map to tigecycline, and TEM penicillin mappings do not map to
third-generation cephalosporins.

Use `--determinants custom.tsv` to **replace** the starter after evidence review.
Copy its column schema: `species`, `antimicrobial`, `determinant`, `aliases`,
`determinant_type`, `reference`, `product_require`, `product_exclude`, `notes`.
Species must use the full scientific name. Use semicolon-separated exact aliases,
not cluster IDs. Types are `presence_absence` or `mutation_deferred`. Optional
product requirements/exclusions are case-insensitive regular expressions.
Keep a source reference for each mapping; annotate substrate, species and
allele-specific limitations rather than adding all antibiotics from a class.
