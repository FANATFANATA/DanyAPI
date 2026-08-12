import shutil
import unittest

from danyapi.pow import (
    _find_native_solver,
    deepseek_hash_v1_hex,
    solve_native,
    solve_node,
)


def _challenge(counter: int) -> tuple[str, str, int]:
    salt = "chk"
    expire_at = 1700000000000
    prefix = f"{salt}_{expire_at}_".encode()
    target = deepseek_hash_v1_hex(prefix + str(counter).encode())
    return target, salt, expire_at


class TestDeepSeekHashV1(unittest.TestCase):
    def test_vectors(self):
        vectors = {
            b"": "e594808bc5b7151ac160c6d39a02e0a8e261ed588578403099e3561dc40c26b3",
            b"A": "7157d45adfe495122cb4a12198a2d603b2e09019177c5efe5d0a12b00247e407",
            b"hello": "50605e468e6d6ead913d7d7ccc4687b83ded157cf0a0c5e011eefece12712fa5",
            b"DeepSeekHashV1": "3fc52c4ae40faa946b1bc0eeb747059a35fba6efaa3d616074e720d6e99436cd",
        }
        for data, expected in vectors.items():
            self.assertEqual(deepseek_hash_v1_hex(data), expected)


class TestSolvers(unittest.TestCase):
    def test_native_matches_python(self):
        if _find_native_solver() is None:
            self.skipTest("native pow_solver not built")
        target, salt, expire_at = _challenge(42)
        self.assertEqual(solve_native(target, salt, expire_at, 200000), 42)

    def test_node_matches_python(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        target, salt, expire_at = _challenge(42)
        self.assertEqual(solve_node(target, salt, expire_at, 200000), 42)


if __name__ == "__main__":
    unittest.main()
