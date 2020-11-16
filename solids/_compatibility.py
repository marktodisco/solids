from typing import Union, Tuple

import sympy as sp
from sympy.logic.boolalg import BooleanTrue, BooleanFalse

from solids import laplacian, show

__all__ = [
    'from_stress',
    'equilibrium',
    'st_venant_compatibility'
]


def from_stress(sigma: sp.Matrix, f: Union[sp.Matrix, list, tuple], v: Union[sp.Matrix, list, tuple]) -> sp.Matrix:
    """
    Compute the compatibility conditions of a state of stress for body forces.

    Parameters
    ----------
    sigma : sp.Matrix
        Stress matrix.
    f : Union[sp.Matrix, list, tuple]
        Body forces.
    v: Union[sp.Matrix, list, tuple]
        Differentiation variables of which `f` and `sigma` are functions.

    Returns
    -------
    sp.Matrix[sp.Eq]
        Compatibility equations  Sympy equations in the form of a 2D (3 x 3) matrix.

    References
    ----------
    [1] Introduction to Linear Elasticity 3rd Edition. pg. 102. Equation 5.16.

    """
    # Initialization and calculation of constant values.
    fkk = sum([f[k].diff(v[k]) for k in range(3)])
    Q1 = sigma.trace()
    nu_ = sp.symbols('nu')
    condition = sp.zeros(3, 3)

    # Populate the matrix
    for i in range(3):
        for j in range(3):
            # Calculate the right-hand side of the compatibility condition.
            lhs = laplacian(sigma[i, j], v)
            lhs += Q1.diff(v[i], v[j]) / (1 + nu_)

            # Calculate the left-hand side of the compatibility condition.
            rhs = f[i].diff(v[j]) + f[j].diff(v[i])
            if i == j:
                rhs += nu_ / (1 - nu_) * fkk

            # Combine the LHS and RHS to form the full condition.
            condition[i, j] = sp.Eq(lhs, rhs)

    return condition


def equilibrium(sigma: sp.Matrix, f: Union[list, sp.Matrix], v: list) -> Tuple[sp.Matrix, bool]:
    conditions = sp.zeros(len(f), 1)
    satisfied = False

    # eq. 2.45 pg. 30
    for i in range(3):
        for j in range(3):
            conditions[i] += sigma[i, j].diff(v[j]) + f[i]

    if sum(abs(conditions)).equals(0):
        satisfied = True

    return conditions, satisfied


def st_venant_compatibility(symbols: list, strain_field: dict, full=False) -> list:
    """
    Compute the St. Venant compatibility equations for a state of strain_field.

    Parameters
    ----------
    symbols : list
        Sympy symbols by which `strain_field` are defined.
    strain_field : Union[dict, sp.Matrix]
        If dict:
            Mapping of strain_field to state-equations.
            Keys must be {11, 22, 33, 12, 23, 13}.
            Diagonal symmetry is assumed.
            i.e. (12 = 21), (23 = 32), and (13 = 31).
        If sp.Matrix:
            Full-form 3 x 3 strain matrix must be supplied.
    full : bool, optional
        If True, shows latex representation of the compatibility equations.
        The default is False.

    Returns
    -------
    list
        Simplified compatibility equations. Use Sympy's `solve` to calculate
        the solution to the system of linear system of equations.
    """
    x1, x2, x3 = symbols

    if isinstance(strain_field, dict):
        strain_field = {k: sp.sympify(v) for (k, v) in strain_field.items()}

        e11 = strain_field[11]
        e22 = strain_field[22]
        e33 = strain_field[33]

        e12 = strain_field[12]
        e23 = strain_field[23]
        e13 = strain_field[13]

        e21 = e12.copy()
        e32 = e23.copy()
        e31 = e13.copy()

    elif isinstance(strain_field, sp.Matrix):
        e11, e12, e13, e21, e22, e23, e31, e32, e33 = strain_field

    else:
        raise ValueError("`strain_field` must be either a dict or sp.Matrix.")

    conds = [
        sp.Eq(e11.diff(x2, 2) + e22.diff(x1, 2), 2 * e12.diff(x1).diff(x2)),
        sp.Eq(e22.diff(x3, 2) + e33.diff(x2, 2), 2 * e23.diff(x2).diff(x3)),
        sp.Eq(e33.diff(x1, 2) + e11.diff(x3, 2), 2 * e31.diff(x3).diff(x1)),
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



































