import unittest
from typing import Any

import sympy as sp

from solids.strain import StrainState


def compute_difference(a: Any, b: Any) -> Any:
    difference = a - b
    if isinstance(a, (sp.Matrix, sp.Integer, sp.Float)):
        difference = sp.N(difference, chop=True)
    else:
        raise ValueError(f"Expected types: (sp.Matrix, sp.Integer, sp.Float) for `a` but received: {type(a)}")
    return difference


class TestStrainState(unittest.TestCase):

    def setUp(self) -> None:
        epsilon = sp.Matrix([20, 3, 2, 3, -10, 5, 2, 5, -8]).reshape(3, 3) / 1000000
        self.strain_state = StrainState(epsilon, dtype='sympy')

    def test_matrix(self):
        self.assertEqual(self.strain_state._sigma, sp.Matrix([20, 3, 2, 3, -10, 5, 2, 5, -8]).reshape(3, 3) / 1000000)

    def test_invariants(self):
        self.assertEqual(self.strain_state.invariants,
                         sp.Matrix([['1/500000'], ['-159/500000000000'], ['159/125000000000000000']]))

    def test_octahedral_shear(self):
        # Test does not pass due to numerical error if a simple comparison (==) is made.
        # Must compute the error (difference) and "chop" off very small values.
        difference = compute_difference(self.strain_state.octahedral_shear, 1.45907124188262e-5)
        self.assertEqual(difference, sp.Integer(0))

    def test_principal_strain(self):
        # Test does not pass due to numerical error if a simple comparison (==) is made.
        # Must compute the error (difference) and "chop" off very small values.
        expected = sp.Matrix([[2.05189991467570e-5, -4.38640972623869e-6, -1.41325894205183e-5]])
        difference = compute_difference(self.strain_state.principal_strain, expected)
        self.assertEqual(difference, sp.Matrix([[0, 0, 0]]))

    def test_max_shear(self):
        # Test does not pass due to numerical error if a simple comparison (==) is made.
        # Must compute the error (difference) and "chop" off very small values.
        difference = compute_difference(sp.Float(self.strain_state.max_shear), 1.732579428363763e-05)
        self.assertEqual(difference, sp.Integer(0))

    def test_decompose(self):
        mean_strain, deviator = self.strain_state.decompose
        self.assertEqual(mean_strain, sp.Matrix([['1/1500000', 0, 0], [0, '1/1500000', 0], [0, 0, '1/1500000']]))
        self.assertEqual(deviator, sp.Matrix([['29/1500000', '3/1000000', '1/500000'],
                                              ['3/1000000', '-1/93750', '1/200000'],
                                              ['1/500000', '1/200000', '-13/1500000']]))
