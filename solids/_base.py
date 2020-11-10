from collections import deque
from typing import List, Tuple, Union, Any

import numpy as np
import sympy as sp
from IPython.display import Math
from IPython.display import display as ipy_display

__all__ = [
    'show',
    'rotation_matrix',
    'matrix_round',
    'save_latex',
    'ref',
    'to_numpy',
    'char_poly',
    'symmetric',
    'jacobian',
    'cross',
    'mean_deviator',
    'elasticity_matrix',
    'compatibility_matrix',
    'get_missing',
    'lame_constant',
    'voigt_to_matrix',
    'matrix_to_voigt',
    'poissons_ratio',
    'laplacian',
    'laplacian_matrix'
]


def show(expr, prefix: str = None, postfix: str = None) -> None:
    """
    Display latex math in Jupyter Notebook.txt

    Parameters
    ----------
    expr
        Sympy expression to display.
    prefix : str, optional
        Insert `prefix` before `expr`, by default None
    postfix : str, optional
        Insert `postfix` before `expr`, by default None
    
    """
    latex_str = sp.latex(expr)
    if prefix is not None:
        latex_str = prefix + latex_str
    if postfix is not None:
        latex_str = latex_str + postfix
    ipy_display(Math(latex_str))


def rotation_matrix(theta: int, unit: str = 'rad', numpy: bool = False):
    """
    Create a z-axis rotation matrix given an angle.

    Parameters
    ----------
    theta : int
        Rotation about the z-axis
    unit : {'rad', 'deg'}, optional
        Radians or degrees. By default 'rad'.
    numpy : bool, optional
        If True, return a Numpy array, otherwise return a sympy matrix.
        By default False.

    Returns
    -------
    {np.array, sp.Matrix}
        If `numpy` is True, will return rotation matrix as np.array. Otherwise,
        will return a sp.Matrix.
        
    """
    if unit == 'deg':
        theta = sp.rad(theta)
    R = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                   [sp.sin(theta), sp.cos(theta), 0],
                   [0, 0, 1]])
    return sp.matrix2numpy(R, dtype='float32') if numpy else R


def matrix_round(mat: sp.Matrix, precision=0):
    mat_copy = mat.copy()
    n, m = mat_copy.shape
    for i in range(n):
        for j in range(m):
            mat_copy[i, j] = mat_copy[i, j].round(precision)
    return mat_copy


def save_latex(expr, filename):
    with open(filename, 'w') as fp:
        fp.write(sp.latex(expr))


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


def to_numpy(x: sp.Matrix, diag: bool = False, dtype='float64'):
    """
    Covert a sympy matrix to a numpy array.
    
    Arguments
    ---------
    x : Matrix (N x M) | (N x 1) | (1 X M) | (1,)
        Sympy matrix to convert.
    diag : bool, optional
        If True, return a diagonal matrix consisting of the elements of x.
        To use this option, x must be a row or column vector, or a 1D array.
        Default is False.
    dtype : {str, np.dtype}, optional
        Convert x to a numpy data type. Default is float64.
    
    Returns
    -------
    (N x M) ndarray.
    If diag=True: (max(x.shape) x max(x.shape)) ndarray
    """
    if (1 in x.shape or len(x.shape) == 1) and diag:
        return np.diagflat(np.asarray(x, dtype=dtype))
    return np.asarray(x, dtype=dtype)


def char_poly(sig: sp.Matrix) -> Tuple[sp.Eq, sp.Symbol]:
    _lambda = sp.symbols('lambda')
    size = min(sig.shape)
    A = sig - sp.diag(*[_lambda] * size)
    d = sp.Eq(A.det(), 0)
    return d, _lambda


def symmetric(sig: list) -> sp.Matrix:
    t = sp.zeros(3, 3)

    t[0, 0] = sig[0]
    t[0, 1] = sig[1]
    t[0, 2] = sig[2]
    t[1, 0] = t[0, 1]
    t[1, 1] = sig[3]
    t[1, 2] = sig[4]
    t[2, 0] = t[0, 2]
    t[2, 1] = t[1, 2]
    t[2, 2] = sig[5]

    return t


def jacobian(u: list, v: List[sp.Symbol]) -> sp.Matrix:
    n = len(u)
    m = len(v)

    u = sp.Matrix(u)
    Ju = sp.zeros(n, m)

    for i in range(n):
        for j in range(m):
            Ju[i, j] = u[i].diff(v[j])

    return Ju


def _even_odd_test(i, j, k, test):
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


def _permutation(i, j, k):
    if _even_odd_test(i, j, k, test='even'):
        return 1.
    elif _even_odd_test(i, j, k, test='odd'):
        return -1.
    return 0.


def cross(a, b):
    c = sp.zeros(3, 1)

    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                c[i - 1] += sp.Integer(_permutation(i, j, k)) * a[j - 1] * b[k - 1]

    return c


def mean_deviator(eps: sp.Matrix) -> Tuple:
    sympy = True if isinstance(eps, sp.Matrix) else False
    numpy = True if isinstance(eps, np.ndarray) else False
    if not (sympy ^ numpy):
        raise ValueError('`eps` must be a sympy Matrix or Numpy array.')

    def eval_sympy():
        e0 = eps.trace() / 3
        mean_strain = sp.diag(*[e0] * 3)
        devaiatoric_strain = eps - mean_strain
        return mean_strain, devaiatoric_strain

    def eval_numpy():
        e0 = np.trace(eps) / 3
        mean_strain = np.diag([e0] * 3)
        deviatoric_strain = eps - mean_strain
        return mean_strain, deviatoric_strain

    if numpy:
        return eval_numpy()
    return eval_sympy()


def elasticity_matrix(lame: sp.Integer = None,
                      G: sp.Integer = None
                      ) -> Tuple[sp.Matrix, sp.Symbol, sp.Symbol]:
    # Create symbols
    lame_constant_, G_ = sp.symbols('lambda, G')

    # Create symbolic elasticity matrix for an isotropic material
    C = sp.zeros(6, 6)
    C[:3, :3] = sp.ones(3, 3) * lame_constant_
    for i, j, in zip(range(3), range(3, 6)):
        C[i, i] = 2 * G_ + lame_constant_
        C[j, j] = 2 * G_

    # Substitute numeric values is available
    if lame is not None:
        C = C.subs(lame_constant_, lame)
    if G is not None:
        C = C.subs(G_, G)

    return C, lame_constant_, G_


def compatibility_matrix(E=None, G=None, nu=None):
    E_, G_, nu_ = sp.symbols('E, G, nu')

    D = sp.zeros(6, 6)
    D[:3, :3] = -sp.ones(3, 3) * nu_ / E_

    for i, j in zip(range(3), range(3, 6)):
        D[i, i] = 1 / E_
        D[j, j] = 1 / G_ / 2

    E, G, nu = get_missing(E, G, nu)

    if E is not None:
        D = D.subs(E_, E)
    if G is not None:
        D = D.subs(G_, G)
    if nu is not None:
        D = D.subs(nu_, nu)

    return D, E_, G_, nu_


def get_missing(E=None, G=None, nu=None):
    E_, G_, nu_ = sp.symbols('E, G, nu')
    symbol_map = {E_: E, G_: G, nu_: nu}
    missing_map = {E_: True, G_: True, nu_: True}
    equation = sp.Eq(nu_, E_ / G_ / 2 - 1)
    missing_counter = 3

    # Determine which parameter is missing
    for symbol, value in symbol_map.items():
        if value is not None:
            missing_map[symbol] = False
            missing_counter -= 1

    # Exit if less than two parameters are not specified
    if missing_counter > 1:
        msg = "Must specify at least two values from [`E`, 'G`, `nu`].\n"
        msg += "\tE: {E}\n".format(E=symbol_map[E_])
        msg += "\tG: {G}\n".format(G=symbol_map[G_])
        msg += "\tnu: {nu}".format(nu=symbol_map[nu_])
        raise ValueError(msg)

    # Calculate the missing parameter
    missing_symbol = sorted(missing_map.items(), key=lambda x: x[1], reverse=True)[0][0]
    solution = sp.solve(equation, missing_symbol)[0]
    for symbol, value in symbol_map.items():
        if value is not None:
            solution = solution.subs(symbol, value)
    symbol_map[missing_symbol] = solution

    E, G, nu = symbol_map.values()
    return E, G, nu


def lame_constant(E, nu):
    return nu * E / (1 + nu) / (1 - 2*nu)


def voigt_to_matrix(voight: Union[sp.Matrix, list]) -> sp.Matrix:
    """
    Convert a vector in standard Voight notation to a 2D symmetric matrix.

    Parameters
    ----------
    voight: (6, 1) sp.matrix
        1D vector.

    Notes
    -----
    https://en.wikipedia.org/wiki/Voigt_notation

    Returns
    -------
    (3, 3) sp.matrix
        2D matrix with 3 dimensions.

    """
    matrix: sp.Matrix = sp.zeros(3, 3)

    matrix[0, 0] = voight[0]
    matrix[1, 1] = voight[1]
    matrix[2, 2] = voight[2]

    matrix[1, 2] = voight[3]
    matrix[2, 1] = voight[3]

    matrix[0, 2] = voight[4]
    matrix[2, 0] = voight[4]

    matrix[0, 1] = voight[5]
    matrix[1, 0] = voight[5]

    return matrix


def matrix_to_voigt(matrix: sp.Matrix) -> sp.Matrix:
    """
    Convert a 2D symmetric matrix to a vector in standard Voight notation.

    Parameters
    ----------
    matrix: (3, 3) sp.matrix
        2D matrix with 3 dimensions.

    Returns
    -------
    (6, 1) sp.matrix
        1D vector.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Voigt_notation

    """
    voight = sp.zeros(6, 1)
    voight[0] = matrix[0, 0]
    voight[1] = matrix[1, 1]
    voight[2] = matrix[2, 2]
    voight[3] = matrix[1, 2]
    voight[4] = matrix[0, 2]
    voight[5] = matrix[0, 1]
    return voight


def poissons_ratio(E: Union[sp.Integer, sp.Float, int, float],
                   G: Union[sp.Integer, sp.Float, int, float]
                   ) -> Union[sp.Integer, sp.Float, int, float]:
    """
    Calculate Poisson's ratio using the elastic and shear moduli of a material.
    Parameters
    ----------
    E : Union[sp.Integer, sp.Float, int, float].
        Modulus of elasticity (Young's modulus).
    G : Union[sp.Integer, sp.Float, int, float].
        Shear modulus. Sometimes also referred to as mu (one of lame's constants in literature).

    Returns
    -------
    Union[sp.Integer, sp.Float, int, float]
        Poisson's ratio.

    """
    if E < 0:
        raise ValueError("`E` must be >= 0.")
    if G <= 0:
        raise ValueError("`G` must be > 0.")
    return E / G / 2 - 1


def laplacian(expression: Union[sp.Add, sp.Mul, sp.Pow],
              symbols: List[sp.Symbol]
              ) -> Any:
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




























