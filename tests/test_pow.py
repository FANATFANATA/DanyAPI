"""Тесты DeepSeekHashV1 против контрольных векторов, снятых с wasm-экспорта
`wasm_deepseek_hash_v1` из sha3_wasm_bg.7b9ca65ddd.wasm.
"""

import unittest

from danyapi.pow import deepseek_hash_v1_hex


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


if __name__ == "__main__":
    unittest.main()
