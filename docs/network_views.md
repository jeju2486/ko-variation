# Community overview, focal networks and adjacency matrix

The development branch now renders the full retained map through complementary
views. Biological selection, representative statistics and annotation are
unchanged. This is a display update; it requires no KOVAR fitting or eggNOG run.

## Redraw an existing map

```bash
python -m ko_variation.network_cli \
  --layout /project/plots/functional_map/map.html \
  --out /project/plots/functional_map/map_explorer
```

`--layout` accepts the offline HTML or its `.layout.json`. Use a different output
prefix to preserve the original. This mode retains all saved names, eggNOG
categories, GO terms, coefficients, P-value fields, distances and supporting
unitig information. It does not follow stale annotation-source paths in copied
metadata. Only the saved payload is needed. It cannot recover unitig details
that were external to the original HTML; preserve `gene_pair_support.tsv.gz`.

The existing `--effects` redraw and functional-map workflow automatically produce
the same new interface. `--layout` preserves existing biological annotation;
use `--effects` when changing annotation or gene-edge selection.

## Views and interaction

- **Community overview** is the default. Rings show the genes' broad functional
  composition; the centre gives community ID and gene count. Click a community
  to inspect its members. Link widths count retained **gene edges**, not unitig
  support or a pooled association effect. Hover for connection counts/density.
- **Community members** shows that group's internal gene edges. External edges
  remain in the data and in the overview, with their counts in community details.
- **Gene neighbourhood** opens from a gene search. Direct neighbours are the
  default; second neighbours can be added explicitly. Labels use collision
  avoidance, unknown nodes are smaller/pale, and node fill retains eggNOG roles.
  Click a neighbour to centre its neighbourhood. Hover always retains full names.
- **Adjacency matrix** orders genes by community, with coloured retained edges
  and function tracks. Hover cells for gene names and original edge statistics.
  White means **no retained map edge**; it does not distinguish untested pairs
  from nonsignificant pairs. Zoom to inspect dense blocks.
- **Full network** preserves the original gene coordinates and complete edge
  set. Selection fades unrelated edges. Reflow remains available explicitly;
  positions are not genomic coordinates.

Direction and minimum-|beta| controls are display filters. The interface reports
the hidden edge count. They do not rewrite the biological map or its exports.
Unestimated effects are hidden when a positive minimum magnitude is requested.
SVG/PNG buttons export the current view. Layout JSON retains the full network
and records the current display state separately. Source scientific fields are
not replaced by aggregate community statistics.

## Community definition and limitations

The current dependency-free choice is NetworkX **Louvain** with unweighted,
unsigned retained gene connectivity, resolution 1 and the saved random seed.
Disconnected groups are split into connected components. This is not Leiden.
Positive and negative edges both contribute connectivity; their signs remain
visible separately. These groups are descriptive, not validated functional
modules, isolate lineages, unique-pattern classes or epistasis groups. IDs are
local to a particular map, not comparable labels between species/runs.

## Output files

- `.html`, `.layout.json`: interactive explorer and complete display payload.
- `.png`, `.svg`: community overview with editable SVG text.
- `.matrix.png`, `.matrix.svg`: ordered matrix; sparse vector cells avoid a
  dense N-by-N in-memory representation.
- `.communities.tsv`, `.community_edges.tsv`: membership, internal/external
  counts, functional composition and distinct gene-edge connection summaries.
- `.nodes.tsv`, `.edges.tsv`: original per-gene/per-edge fields, with a new
  descriptive community assignment on nodes.
- `.full.png`, `.full.svg`: original full-network static exports, when building
  from an effect table or running the functional-map workflow. Saved-layout-only
  redraw retains the full graph interactively; its current full view can also
  be exported with the SVG/PNG buttons.

For a paper, use the overview to establish global structure and selected focal
panels to explain particular genes. Retain the full graph and matrix as
supplementary exploratory views. Connectivity grouping supplies no new
association test and does not establish a phenotype interaction.
