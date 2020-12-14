from collections import deque
from typing import List, Union, Any, Tuple

import sympy as sp

from solids.base import show

__all__ = [
    'ref',
    'jacobian',
    'cross',
    'laplacian',
    'laplacian_matrix',
    'even_odd_test',
    'char_poly',
    'permutation'
]


def ref(S: sp.Matrix, display: bool = True) -> sp.Matrix:
    """
    Transform the matrix S to row echelon form.

    Parameters
    ----------
    S : Matrix
        Matrix to transform.
    display : bool, optional
        Display the results in Jupyter notebook. The default True.

    Returns
    -------
    Matrix
        [description]
    """
    Sc = S.copy().n(chop=True)
    if display:
        show(Sc.n(6), prefix=r"A=")

    Sc[0, :] /= Sc[0, 0]
    Sc[1, :] -= Sc[1, 0] * Sc[0, :]
    Sc[2, :] -= Sc[2, 0] * Sc[0, :]
    # Sc = Sc.n(chop=True)

    if display:
        show(Sc.n(6))

    Sc[1, :] = Sc[1, :] / Sc[1, 1]
    Sc[2, :] -= Sc[2, 1] * Sc[1, :]
    # Sc = Sc.n(chop=True)

    if display:
        show(Sc.n(6))

    Sc[2, :] = Sc[2, :] / Sc[2, 2]
    Sc = Sc.n(chop=True)

    if display:
        show(Sc.n(6))

    return Sc


def jacobian(u: list, v: List[sp.Symbol]) -> sp.Matrix:
    n = len(u)
    m = len(v)

    u = sp.Matrix(u)
    Ju = sp.zeros(n, m)

    for i in range(n):
        for j in range(m):
            Ju[i, j] = u[i].diff(v[j])

    return Ju


def even_odd_test(i, j, k, test):
    permutation = deque([i, j, k])

    if test == 'even':
        baseline = deque([1, 2, 3])
    elif test == 'odd':
        baseline = deque([3, 2, 1])
    else:
        raise ValueError("`test` must be one of {'even', 'odd'}")

    for _ in range(3):
        if permutation == baseline:
            return True
        permutation.rotate(1)

    return False


def permutation(i, j, k):
    if even_odd_test(i, j, k, test='even'):
        return 1.
    elif even_odd_test(i, j, k, test='odd'):
        return -1.
    return 0.


def cross(a, b):
    c = sp.zeros(3, 1)

    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                c[i - 1] += sp.Integer(permutation(i, j, k)) * a[j - 1] * b[k - 1]

    return c


def laplacian(expression: Union[sp.Add, sp.Mul, sp.Pow], symbols: List[sp.Symbol]) -> Any:
    """
    Calculate the Laplacian of a scalar value.

    Parameters
    ----------
    expression : Union[sp.Add, sp.Mul, sp.Pow]
        Symbolic Sympy expression.
    symbols : List[sp.Symbol]

    Returns
    -------
    Any
        Symbolic Sympy expression.

    References
    ----------
    [1] https://www.mathworks.com/help/symbolic/laplacian.html

    """
    N = len(symbols)
    L = sp.Integer(0)

    for i in range(N):
        L += expression.diff(symbols[i], 2)

    return L


def laplacian_matrix(matrix: sp.Matrix, symbols: List[sp.Symbol]) -> sp.Matrix:
    """
    Compute the element-wise scalar Laplacian of a 2-dimensional matrix.

    Parameters
    ----------
    matrix : sp.Matrix
        2-dimensional matrix.
    symbols : List[sp.Symbol]
        Variables of which to compute the Laplacian with respect.

    Returns
    -------
    sp.Matrix
        The element-wise scalar Laplacian.

    References
    ----------
    [1] https://www.mathworks.com/help/symbolic/laplacian.html

    """
    N = len(symbols)
    L_matrix = sp.zeros(N)

    for i in range(N):
        for j in range(N):
            L_matrix[i, j] = laplacian(matrix[i, j], symbols)

    return L_matrix


def char_poly(sig: sp.Matrix) -> Tuple[sp.Eq, sp.Symbol]:
    _lambda = sp.symbols('lambda')
    size = min(sig.shape)
    A = sig - sp.diag(*[_lambda] * size)
    d = sp.Eq(A.det(), 0)
    return d, _lambda
