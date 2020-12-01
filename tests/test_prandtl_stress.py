import unittest

import sympy as sp

from solids._stress import PrandtlStress

class TestPrandtlStress(unittest.TestCase):
    def setUp(self) -> None:
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.B = sp.Symbol('B')
        self.C = sp.Symbol('C')
        self.D = sp.Symbol('D')
        self.G = sp.Symbol('G')
        self.h = sp.Symbol('h')
        self.phi = (self.C * (self.x - sp.sqrt(3)*self.y - sp.sympify('2/3')*self.h)
                    * (self.x + sp.sqrt(3)*self.y - sp.sympify('2/3')*self.h)
                    * (self.x + self.h/3))
        self.ps = PrandtlStress(self.phi, self.x, self.y)

    def test_H(self):
        self.assertTrue(self.ps.H.equals(-4 * self.C * self.h))

    def test_sigma_xz(self):
        self.assertTrue(self.ps.sigma_xz.equals(-2 * self.C * self.y * (self.h + 3*self.x)))

    def test_sigma_yz(self):
        self.assertTrue(self.ps.sigma_yz.equals(self.C * (-3*self.x**2 + 2*self.h*self.x + 3*self.y**2)))

    def test_eps_xz(self):
        self.assertTrue(self.ps.eps_xz.equals(self.ps.sigma_xz / self.G / 2))

    def test_eps_yz(self):
        self.assertTrue(self.ps.eps_yz.equals(self.ps.sigma_yz / self.G / 2))

    def test_alpha(self):
        self.assertTrue(self.ps.alpha.equals(-self.ps.H / self.G / 2))

    def test_tau(self):
        true_tau = sp.sqrt(self.C**2 * (4 * self.h**2 * self.x**2
                                        + 4 * self.h**2 * self.y**2
                                        - 12 * self.h * self.x**3
                                        + 36 * self.h * self.x * self.y**2
                                        + 9 * self.x**4
                                        + 18 * self.x**2 * self.y**2
                                        + 9 * self.y**4))
        self.assertTrue(self.ps.tau.equals(true_tau))










