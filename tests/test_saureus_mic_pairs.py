"""Regression test for sparse cells, matched controls and the pairs entry point."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks import saureus_mic as mic


class FourCellTests(unittest.TestCase):
    def test_first_neighbour_models_controls_and_sparse_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rng = np.random.default_rng(19)
            size = 120
            states = np.tile(np.arange(4), size // 4)
            a, b = states // 2, states % 2
            y = .5 * a + 2 * a * b + rng.normal(0, .3, size)
            censored = y < -.2
            values = 2 ** np.maximum(y, -.2)
            columns = np.column_stack((b, a, a * b, 1 - b,
                                       rng.integers(0, 2, (size, 15))))
            mic.npz_write(out/"prepared.npz", X=columns, K=np.eye(size))
            mic.npz_write(out/"genotypes.npz", G=a[:, None], names=np.array(["blaZ_presence"]))
            mic.json_write(out/"audit_identity.json", dict(revision=mic.AUDIT_REVISION))
            mic.json_write(out/"prepared.json", dict(
                samples=[str(i) for i in range(size)], groups=[str(i // 20) for i in range(size)],
                loci=list(range(columns.shape[1])),
                locus_gene={str(i): f"gene{i}" for i in range(columns.shape[1])},
                target_columns={"penicillin": []}, neighbour_columns={"penicillin": [0, 1, 2]},
                second_columns={"penicillin": [3]}, random_pool=list(range(4, columns.shape[1])),
                metadata=[dict(benzylpenicillin_mic=str(v),
                               benzylpenicillin_mic_sign="<" if c else "=")
                          for v, c in zip(values, censored)]))
            args = ["pairs", "--out", directory, "--folds", "3", "--threads", "2",
                    "--random-sets", "2", "--pair-alpha", "1"]
            mic.main(args)
            prediction = Path(json.loads((out/"latest_pair_analysis.json").read_text())["path"])
            cells = list(mic.records(prediction/"observed_cells.tsv"))
            self.assertEqual({r["neighbour"] for r in cells}, {"0", "1", "2"})
            self.assertEqual([r["n"] for r in cells if r["neighbour"] == "0"], ["30"] * 4)
            fits = {(r["neighbour"], r["model"], r["status"])
                    for r in mic.records(prediction/"pair_fit_status.tsv") if r["source"] == "neighbour"}
            self.assertIn(("1", "additive", "SPARSE_TRAIN"), fits)
            self.assertIn(("2", "additive", "OK"), fits)
            self.assertIn(("2", "interaction", "SPARSE_TRAIN"), fits)
            summaries = list(mic.records(prediction/"random_comparison.tsv"))
            supported = next(r for r in summaries if r["neighbour"] == "0" and r["model"] == "interaction")
            self.assertEqual(supported["complete_random_sets"], "2")
            self.assertGreater(float(supported["delta_nll_vs_additive"]), 0)
            matches = list(mic.records(prediction/"random_matches.tsv"))
            self.assertTrue(all(int(r["unitig"]) >= 4 for r in matches))
            checkpoint = prediction/"fold0.json"
            before = checkpoint.stat().st_mtime_ns
            mic.main(args)
            self.assertEqual(checkpoint.stat().st_mtime_ns, before)
