from builtins import ValueError
from typing import Tuple, Union

import sympy as sp

from solids.base import show
from solids.math import jacobian
from solids.stress import StressState, octahedral_normal, principal_stresses

__all__ = [
    'calc_strain',
    'StrainState'
]


def calc_strain(u: list = None, v: list = None, H: sp.Matrix = None, ret_H: bool = False
                ) -> Union[Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix],
                      Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]]:
    """
    Calculate infinitesimal strain, infinitesimal rotation, Lagrangian strain, and Eulerian strain. Either specify `u`
    and `v` together or just `H`. Not all arguments are required.

    Parameters
    ----------
    u : list, optional
        Symbolic strain field, by default None
    v : list, optional
        Symbols to differentiate wrt, by default None
    H : sp.Matrix, optional
        Derivative matrix of the strain field equations, by default None
    ret_H : bool, optional
        Return the deriavtives matrix H.

    Returns
    -------
    Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]
        infinitesimal strain, infinitesimal rotation, Lagrangian strain, and
        Eulerian strain. (epsilon, omega, E, e)

    If ret_H is True

    Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]
        infinitesimal strain, infinitesimal rotation, Lagrangian strain, and
        Eulerian strain. (epsilon, omega, E, e, H)

    Raises
    ------
    ValueError
        If `H` is None, `u` and `v` must not be not None.

    """
    if H is None:
        if u is None or v is None:
            raise ValueError("If `H` is None, `u` and `v` must not be None.")
        else:
            H = jacobian(u, v)

    n, m = H.shape
    epsilon = sp.zeros(n, m)
    omega = sp.zeros(n, m)
    E = sp.zeros(n, m)
    e = sp.zeros(n, m)

    for i in range(n):
        for j in range(m):
            epsilon[i, j] = (H[i, j] + H[j, i]) / 2
            omega[i, j] = (H[i, j] - H[j, i]) / 2

            finite_contrib = sum([H[k, i] * H[k, j] for k in range(n)]) / 2
            E[i, j] = epsilon[i, j] + finite_contrib
            e[i, j] = epsilon[i, j] - finite_contrib

    if ret_H:
        return epsilon, omega, E, e, H
    return epsilon, omega, E, e


class StrainState(StressState):
    """
    State of strain.

    Attributes
    ----------
    principal_stresses
    principal_axes
    invariants
    octahedral_shear
    octahedral_normal
    max_shear
    decompose
    char_poly

    Methods
    -------
    full_report()

    """
    def __init__(self, sigma, dtype):
        super(StrainState, self).__init__(sigma, dtype)

    def full_report(self):
        self._get_principal()
        _ = self.invariants
        _ = self.char_poly

        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                _ = self.octahedral_normal
                _ = self.octahedral_shear
                _ = self.max_shear

        if self._dtype == 'numpy':
            self._report_numpy()
        elif self._dtype == 'sympy':
            self._report_sympy()
        return

    @property
    def principal_strain(self):
        if self._pr_stress is None:
            self._get_principal()
        return self._pr_stress

    # @property
    # def octahedral_shear(self):
    #     if self._octahedral_shear is None:
    #         self._octahedral_shear = octahedral_shear(self._sigma, principal=False)
    #     return self._octahedral_shear

    @property
    def octahedral_normal(self):
        if self._octahedral_normal is None:
            self._octahedral_normal = octahedral_normal(self._sigma, principal=False)

        return self._octahedral_normal

    # @property
    # def max_shear(self):
    #     if self._max_shear is None:
    #         if self._pr_stress is None:
    #             self._pr_stress, _ = principal_stresses(
    #                 self._sigma, dtype='numpy', display=False)
    #
    #         self._max_shear = 2. * max_shear(self._pr_stress, principal=True)
    #
    #     return self._max_shear

    def _report_numpy(self):
        self._custom_print('State of Stress', self._sigma)
        self._custom_print('Principal Stresses', self._pr_stress)
        self._custom_print('Principal Axes', self._pr_ax)
        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                self._custom_print('Octahedral Normal Stress', self._octahedral_normal)
                self._custom_print('Octahedral Shear Stress', self._octahedral_shear)
                self._custom_print('Max Shear Stress', self._max_shear)
        self._custom_print('Invariants', self._invariants)
        self._custom_print('Characteristic Polynomial', self._char_poly[0].lhs)
        return

    def _report_sympy(self):
        show(self._sigma, r"\text{State of Stress}\\\varepsilon=", r"\\")
        show(self._pr_stress, r"\text{Principal Stresses}\\\varepsilon^{(i)}=", r"\\")
        show(self._pr_ax, r"\text{Principal Axes}\\\hat{n}^{(i)}=", r"\\")
        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                show(self._octahedral_normal, r"\text{Octahedral Normal Stress}\\\varepsilon_{nn}^{oct}=", r"\\")
                show(self._octahedral_shear, r"\text{Octahedral Shear Stress}\\\varepsilon_{ns}^{oct}=", r"\\")
                show(self._max_shear, r"\text{Max Shear Stress}\\\varepsilon_{oct}^{max}=", r"\\")
        show(self._invariants.T, r"\text{Invariants}\\R_i=", r"\\")
        show(self._char_poly[0].lhs, r"\text{Characteristic Polynomial}\\p=", r"\\")

    def _get_principal(self):
        self._pr_stress, self._pr_ax = principal_stresses(
            self._sigma, display=False, dtype=self._dtype)
