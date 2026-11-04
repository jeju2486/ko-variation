from __future__ import annotations

import unittest

from ko_variation import __version__


class PackageMetadataTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(__version__, "0.9.2")


if __name__ == "__main__":
    unittest.main()
