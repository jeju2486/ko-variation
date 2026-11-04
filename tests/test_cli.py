"""Contracts for the direct-discovery public entry point."""
import contextlib
import io
import unittest
from unittest.mock import patch
from ko_variation.cli import main

class CommandLineContractTests(unittest.TestCase):
    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--version"]), 0)
        self.assertEqual(output.getvalue().strip(), "KO-Variation 0.9.2")

    def test_discover_alias_forwards_arguments(self):
        with patch("ko_variation.discovery.main", return_value=0) as discover:
            self.assertEqual(main(["discover", "--help"]), 0)
            discover.assert_called_once_with(["--help"])

    def test_default_invokes_discovery(self):
        with patch("ko_variation.discovery.main", return_value=0) as discover:
            main(["--help"])
            discover.assert_called_once_with(["--help"])

    def test_retired_pair_scanner_is_explicit(self):
        with self.assertRaisesRegex(SystemExit, "v0.8.3"):
            main(["--pairs", "old.tsv"])
