# Statistical model in KOVAR 0.9.2

For predictor pattern A and response pattern B, the alternative association model
is logit P(B_i=1) = alpha + beta A_i + b_i, with b ~ N(0,tau K). K comes from
shared root-to-MRCA branch lengths and is normalized to mean diagonal one.
The null fixes beta=0. Computational orientation is deterministic and does not
establish biological directionality.

PQL estimates a response-specific null model. Exact and complementary binary
response patterns can reuse null information. The eigenvectors of K are shared;
working weights, weighted decompositions and variance components depend on the
response. Discovery uses a score statistic U^2/V. Adaptive bounds can certify
which side of a predefined significance threshold the statistic falls on;
exact evaluation resolves remaining candidates. Bounds are not exact P-values.

The default final threshold is 0.05 divided by all unordered unique-pattern
pairs, not the number of screened candidates. Locus MAF and four-cell eligibility
are applied as documented in the discovery guide. Coarse PC and representative
filters are heuristic; they can miss associations. Exhaustive search bypasses
those heuristic candidate reductions, retaining numerical bounds.

Graph distance is calculated after selection. All significant local candidates
remain available. Full alternative PQL fits estimate beta for significant pairs
with confirmed distal distance. Neither beta nor statistical significance proves
phenotype-dependent interaction. Selected estimates may show selection bias.

SPA helpers remain available to historical numerical comparisons; production
v0.9.2 discovery uses SPA off. The old candidate-table scanner is reproducible
from the v0.8.3 tag in a separate environment.
