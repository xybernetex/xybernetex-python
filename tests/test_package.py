import unittest

import xybernetex


class PackageTest(unittest.TestCase):
    def test_imports_and_has_a_version(self):
        self.assertRegex(xybernetex.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
