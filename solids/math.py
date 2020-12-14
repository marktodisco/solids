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
        ``S`` in reduced row echelon form.

    Notes
    -----
    .. warning:: This function is very sensitive to the choice of ``S`` and is not guaranteed to converge. Use with caution.

    """
    Sc = S.copy().n(chop=True)
    steps = [Sc]

    Sc[0, :] /= Sc[0, 0]
    Sc[1, :] -= Sc[1, 0] * Sc[0, :]
    Sc[2, :] -= Sc[2, 0] * Sc[0, :]
    steps.append(Sc)

    Sc[1, :] = Sc[1, :] / Sc[1, 1]
    Sc[2, :] -= Sc[2, 1] * Sc[1, :]
    steps.append(Sc)

    Sc[2, :] = Sc[2, :] / Sc[2, 2]
    Sc = Sc.n(chop=True)
    steps.append(Sc)

    if display:
        for step in steps:
            show(step.n(6))

    return Sc


def jacobian(u: list, v: List[sp.Symbol]) -> sp.Matrix:
    """
    Compute the Jacobian matrix.

    Parameters
    ----------
    u : list
        List of functions.
    v : List[Symbol]
        List of variables.

    Returns
    -------
    Matrix
        Jacobian matrix of :math:`u` with respect to :math:`v`.

    """
    n = len(u)
    m = len(v)

    u = sp.Matrix(u)
    Ju = sp.zeros(n, m)

    for i in range(n):
        for j in range(m):
            Ju[i, j] = u[i].diff(v[j])

    return Ju


def even_odd_test(i: int, j: int, k: int, test: str) -> bool:
    """
    Test if even or odd permutation of (1, 2, 3).

    Parameters
    ----------
    i : int
    j : int
    k : int
    test : {'even', 'odd'}
        Test case. Must be 'even' or 'odd'.

    Returns
    -------
    bool
        Returns True if (i, j, k) is an even/odd permutation of (1, 2, 3) based on `test`.

    """
    permutation_ijk = deque([i, j, k])

    if test == 'even':
        baseline = deque([1, 2, 3])
    elif test == 'odd':
        baseline = deque([3, 2, 1])
    else:
        raise ValueError("`test` must be one of {'even', 'odd'}")

    for _ in range(3):
        if permutation_ijk == baseline:
            return True
        permutation_ijk.rotate(1)

    return False


def permutation(i: int, j: int, k: int) -> int:
    """
    Return the value of the permutation tensor.

    Parameters
    ----------
    i : int
    j : int
    k : int

    Returns
    -------
    float
        Return 1 if (i, j, k) is an even permutation.
        Return -1 if (i, j, k) is an odd permutation.
        Return 0 if (i, j, k) otherwise.

    """
    if even_odd_test(i, j, k, test='even'):
        return 1
    elif even_odd_test(i, j, k, test='odd'):
        return -1
    return 0


def cross(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """
    Compute the cross product of two vectors.

    Parameters
    ----------
    a : Matrix
        First vector.
    b : Matrix
        Second vector.

    Returns
    -------
    Matrix
        Returns the cross product between `a` and `b`.

    """
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


def char_poly(matrix: sp.Matrix) -> Tuple[sp.Eq, sp.Symbol]:
    """
    Compute the characteristic polynomial of a matrix.

    Parameters
    ----------
    matrix : Matrix
        2D matrix of values.

    Returns
    -------
    (Equation, Symbol)
        Returns the characteristic polynomial equation, and the symbolic variable.

    """
    _lambda = sp.symbols('lambda')
    size = min(matrix.shape)
    A = matrix - sp.diag(*[_lambda] * size)
    d = sp.Eq(A.det(), 0)
    return d, _lambda
