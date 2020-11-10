import unittest

import sympy as sp

from solids._base import voigt_to_matrix
from solids._compatibility import equilibrium


class TestCompatibility(unittest.TestCase):

    def test_equilibrium_pass(self):
        v = sp.symbols('x_(1:4)')
        x1, x2, x3 = v

        sig_11 = x1 ** 2 + x2 + 3 * x3 ** 2
        sig_22 = 2 * x1 + x2 ** 2 + 2 * x3
        sig_33 = -2 * x1 + x2 + x3 ** 2
        sig_23 = x1 ** 2 - x2 * x3
        sig_13 = x2 ** 2 - x1 * x3
        sig_12 = -x1 * x2 + x3 ** 3
        f = [0] * 3

        sigma = voigt_to_matrix([sig_11, sig_22, sig_33, sig_23, sig_13, sig_12])
        conditions, satisfied = equilibrium(sigma, f, v)

        self.assertEqual(conditions, sp.Matrix([0] * 3))
        self.assertEqual(satisfied, True)

    def test_equilibrium_fail(self):
        v = sp.symbols('x_(1:4)')
        x1, x2, x3 = v

        sig_11 = x1 ** 3 + x2 + 3 * x3 ** 2
        sig_22 = 2 * x1 + x2 ** 2 + 2 * x3 ** 2
        sig_33 = -2 * x1 + x2 + x3 ** 2
        sig_23 = x1 ** 2 - x2 * x3 ** 2
        sig_13 = x2 ** 2 - x1 * x3
        sig_12 = -x1 * x2 + x3 ** 3
        f = [0] * 3

        sigma = voigt_to_matrix([sig_11, sig_22, sig_33, sig_23, sig_13, sig_12])
        conditions, satisfied = equilibrium(sigma, f, v)

        expected = sp.Matrix([[3 * x1 ** 2 - 2 * x1], [-2 * x2 * x3 + x2], [-x3 ** 2 + x3]])
        self.assertEqual(conditions, expected)
        self.assertEqual(satisfied, False)


if __name__ == '__main__':
    unittest.main()
