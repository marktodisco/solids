from typing import Union, Tuple

import sympy as sp

from solids import laplacian

__all__ = [
    'from_stress',
    'equilibrium'
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






































