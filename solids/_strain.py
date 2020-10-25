import numpy as np
import sympy as sp
from typing import List, Tuple
from collections import deque
from ._base import gradient


__all__ = [
    'strain'
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
    H = gradient(u, v) if do_diff else H
    
    if do_diff:
        H = gradient(u, v)
    
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