# Annotation names and functional node colours

These helpers redraw completed selected-pair effects. They do not change the
scanner, significance criterion, LD rule, annotation geometry, or fitted effects.

## Refresh names and search

```bash
ko-variation-network --effects map.distal.tsv --gene-catalog annotation/gene_catalog.tsv \
  --out map_named
```

The optional catalogue updates names/products by exact stable cluster ID.
Existing biological aliases are retained, including multi-name annotations such
as `araJ; lmrS`. Canvas labels, tooltips and edge descriptions use annotation
names. Unknown nodes use a product description or `Unannotated protein`.
Internal IDs are available in the collapsible details field and remain unchanged
in tables for joins. Shared names do not merge distinct clusters.

Search checks every gene alias, product, eggNOG name, functional category, GO
term and internal ID. It lists matching nodes, prioritizes exact alias matches,
highlights all matches, and brings the selected node's neighbourhood into view.
Parentheses and punctuation are ignored, so `tetM` also matches `tet(M)`.
Selection changes node outlines, preserving functional fill colours.

## Prepare representative proteins and annotate

```bash
ko-variation-prepare-proteins --effects map.distal.tsv \
  --panaroo panaroo/gene_presence_absence_roary.csv --bakta bakta \
  --out functional
emapper.py -i functional/network_proteins.faa --itype proteins \
  -o network --output_dir functional --data_dir /path/to/eggnog_data --cpu 8
ko-variation-network --effects map.distal.tsv --gene-catalog annotation/gene_catalog.tsv \
  --eggnog functional/network.emapper.annotations --out map_functional
```

The protein preparation expects `bakta/SAMPLE/SAMPLE.faa`. It matches Bakta
protein IDs to Panaroo member IDs, including `_len`, `_pseudo`, and
`_len_pseudo` suffixes. It selects the first available unambiguous member in
Panaroo sample-column and FASTA order. FASTA query IDs are cluster IDs, not gene
names. `protein_manifest.tsv` records sequence hashes, source IDs, and missing
protein matches; `protein_preparation.json` records counts and the selection rule.
Mixed cluster names remain mixed: a representative protein's predicted function
does not establish the identity/function of every allele or member in a cluster.

Install eggNOG-mapper separately with its compatible database. Version 2.x uses
eggNOG 5 data; version 3 uses eggNOG 7 data. Do not mix these databases. The
importer reads the named `query`, `COG_category`, `GOs`, and optional
`Preferred_name` columns rather than fixed column positions. Duplicate query
IDs and tables with no matching query IDs are rejected. Missing/no-hit nodes
remain in the graph as unassigned. Existing annotations take priority; an
eggNOG-only name carries `(predicted)` and a recorded name source.

Official usage: <https://github.com/eggnogdb/eggnog-mapper/blob/main/USAGE.md>.

## Display categories

Node fill uses these explicit broad **COG display bins**, not inferred GO labels:

| Display bin | COG letters |
| --- | --- |
| Metabolism | C, G, E, F, H, I, P, Q |
| Transport / secretion | U, W |
| Cell envelope | M |
| Genetic information processing | J, A, L, B |
| Regulation / signalling | T, K |
| Cellular processes | D, N, Y, Z, O, V |
| Unassigned | R, S, missing or unsupported categories |

More than one assigned bin produces `Multiple categories`; all component bins
and COG letters remain in the tooltip/table. A GO term can describe several
aspects of function, so detailed GO assignments are retained for inspection and
search rather than forced into a single colour. Categories describe predicted
gene-product functions; they do not identify resistance mechanisms or epistasis.

PNG, SVG, offline HTML and node tables use the same colours. Edge colour remains
coefficient direction; width/spacing remains representative absolute beta.
Functional annotation does not add or remove nodes or edges. The layout JSON
records the colour mapping and catalogue/eggNOG input paths and SHA-256 hashes.

The private ARC wrapper is distributed separately, outside this repository. It
reuses completed proteins and eggNOG annotations only for matching source
identities/settings. Its eggNOG run directory records the command, tool version,
protein hash and database file inventory. An incomplete identical run uses
eggNOG's `--resume`; changed inputs/settings use a separate directory.
