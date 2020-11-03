import numpy as np
import sympy as sp
from typing import List, Tuple
from collections import deque
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
from ._base import jacobian, show
from ._stress import (StressState, octahedral_shear, octahedral_normal,
                      principal_stresses, char_poly, max_shear)


__all__ = [
    'strain',
    'st_venant_compatability',
    'StrainState'
]


def strain(u: list = None,
           v: list = None,
           H: sp.Matrix = None
           ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Calculate infitesimal strain, infitesimal rotation, Lagrangian strain, and
    Eulerian strain.

    Parameters
    ----------
    u : list, optional
        Symbolic strain field, by default None
    v : list, optional
        Symbols to differentiate wrt, by default None
    H : sp.Matrix, optional
        Derivative matrix forof the strain field equations, by default None

    Returns
    -------
    Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]
        Infitesimal strain, infitesimal rotation, Lagrangian strain, and
        Eulerian strain. (epsilon, omega, E, e)

    Raises
    ------
    ValueError
        If `u` and `H` are both unspecified.
    ValueError
        If `u` is specified, `v` must also be specified.
    """
    if u is None and H is None:
        raise ValueError("`u` or `H` must be specified.")

    if u is not None and v is None:
        raise ValueError("Must specify `v` if `u` is provided.")
        
    do_diff = False if H is not None else True
    H = jacobian(u, v) if do_diff else H
    
    if do_diff:
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
            
    return epsilon, omega, E, e


def st_venant_compatability(symbols: list, strain: dict, full=False) -> list:
    """
    Compute the St. Venant compatability equations for a state of strain.

    Parameters
    ----------
    symbols : list
        Sympy symbols by which `strain` are defined.
    strain : dict
        Mapping of strain to state-equations.
        Keys must be {11, 22, 33, 12, 23, 13}.
        Diaglonal symmetry is assumed.
        i.e. (12 = 21), (23 = 32), and (13 = 31).
    full : bool, optional
        If True, shows latex representation of the compatabillity equations.
        The default is False.

    Returns
    -------
    list
        Simplified compatability equations. Use Sympy's `solve` to calculate
        the solution to the system of linear system of equations.
    """
    x1, x2, x3 = symbols
    
    strain = {k: sp.sympify(v) for (k, v) in strain.items()}
    
    e11 = strain[11]
    e22 = strain[22]
    e33 = strain[33]
    
    e12 = strain[12]
    e23 = strain[23]
    e13 = strain[13]
    
    e21 = e12.copy()
    e32 = e23.copy()
    e31 = e13.copy()
    
    conds = [
        sp.Eq(e11.diff(x2, 2) + e22.diff(x1, 2), 2*e12.diff(x1).diff(x2)),
        sp.Eq(e22.diff(x3, 2) + e33.diff(x2, 2), 2*e23.diff(x2).diff(x3)),
        sp.Eq(e33.diff(x1, 2) + e11.diff(x3, 2), 2*e31.diff(x3).diff(x1)),
        sp.Eq(e12.diff(x1).diff(x3) + e13.diff(x1).diff(x2) - e23.diff(x1, 2),
              e11.diff(x2).diff(x3)),
        sp.Eq(e23.diff(x2).diff(x1) + e21.diff(x2).diff(x3) - e31.diff(x2, 2), 
              e22.diff(x3).diff(x1)),
        sp.Eq(e31.diff(x3).diff(x2) + e32.diff(x3).diff(x1) - e21.diff(x3, 2), 
              e33.diff(x1).diff(x2))
    ]
    
    if full:
        _print_conds(conds)
    
    return conds


def _print_conds(conds):
    equations = [
        r"\varepsilon_{11,22} + \varepsilon_{22,11} &= 2 \varepsilon_{12,12}",
        r"\varepsilon_{22,33} + \varepsilon_{33,22} &= 2 \varepsilon_{23,23}",
        r"\varepsilon_{33,11} + \varepsilon_{11,33} &= 2 \varepsilon_{31,31}",
        r"\varepsilon_{12,13} + \varepsilon_{13,12} - \varepsilon_{23,11} &= \varepsilon_{11,23}",
        r"\varepsilon_{23,21} + \varepsilon_{21,23} - \varepsilon_{31,22} &= \varepsilon_{22,31}",
        r"\varepsilon_{31,32} + \varepsilon_{32,31} - \varepsilon_{31,22} &= \varepsilon_{33,12}"
    ]
    
    msg = r"\begin{alignat*}{4}"
    
    for i, _ in enumerate(equations):
        equations[i] += r" &&\quad \Longleftrightarrow \quad&&"
        
        if isinstance(conds[i], (bool, BooleanTrue, BooleanFalse)):
            lhs = rhs = '0'
        elif isinstance(conds[i], sp.Eq):
            lhs = sp.latex(conds[i].lhs)
            rhs = sp.latex(conds[i].rhs)
        else:
            raise ValueError("Unknown solution for `conds`.")
        
        equations[i] += " " + lhs + r" &&= " + rhs + r"\\"
        
        msg += equations[i]
    
    msg += r"\end{alignat*}"
    
    show(msg)


class StrainState(StressState):
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
    
    @property
    def octahedral_shear(self):
        if self._octahedral_shear is None:
            self._octahedral_shear = 2. * octahedral_shear(
                self._sigma, principal=False)
        
        return self._octahedral_shear
    
    @property
    def octahedral_normal(self):
        if self._octahedral_normal is None:
            self._octahedral_normal = octahedral_normal(
                self._sigma, principal=False)
        
        return self._octahedral_normal
    
    @property
    def max_shear(self):
        if self._max_shear is None:
            if self._pr_stress is None:
                self._pr_stress, _ = principal_stresses(
                    self._sigma, dtype='numpy', display=False)
            
            self._max_shear = 2. * max_shear(self._pr_stress, principal=True)
        
        return self._max_shear
    
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