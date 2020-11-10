import unittest

import sympy as sp

from solids import poissons_ratio


class TestBase(unittest.TestCase):

    def test_poissons_ratio(self):
        self.assertEqual(poissons_ratio(30e6, 12e6), 0.25)
        self.assertEqual(poissons_ratio(sp.Integer(30e6), sp.Integer(12e6)), sp.sympify('1/4'))

        with self.assertRaises(ValueError):
            _ = poissons_ratio(-1, 1)
            _ = poissons_ratio(1, -1)
            _ = poissons_ratio(-1, -1)
            _ = poissons_ratio(1, 0)


if __name__ == '__main__':
    unittest.main()
