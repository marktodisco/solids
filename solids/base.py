from typing import Tuple, Union

import numpy as np
import sympy as sp
from IPython.display import Math
from IPython.display import display as ipy_display

__all__ = [
    'show',
    'rotation_matrix',
    'matrix_round',
    'save_latex',
    'to_numpy',
    'symmetric',
    'mean_deviator',
    'elasticity_matrix',
    'compatibility_matrix',
    'get_missing',
    'lame_constant',
    'voigt_to_matrix',
    'matrix_to_voigt',
    'poissons_ratio',
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


def rotation_matrix(theta: int, axis='z', unit: str = 'rad', numpy: bool = False):
    """
    Create a z-axis rotation matrix given an angle.

    Parameters
    ----------
    theta : int
        Rotation about the z-axis
    axis : {'x', 'y', 'z'}
        Axis of rotation
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
    sin = sp.sin
    cos = sp.cos

    if unit == 'deg':
        theta = sp.rad(theta)

    R = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0], [sp.sin(theta), sp.cos(theta), 0], [0, 0, 1]])
    return sp.matrix2numpy(R, dtype='float32') if numpy else R


def matrix_round(mat: sp.Matrix, precision=0):
    """
    Round the elements of a SymPy matrix to a desired precision.

    Parameters
    ----------
    mat : Matrix
        Matrix to round.
    precision : float
        Number of decimal places to round to.

    Returns
    -------
    Matrix
        Rounded `mat` to `precision` decimal places.

    Notes
    -----
    .. warning:: Deprecated. Use the bound method `n` of `sp.Matrix`.
    """

    mat_copy = mat.copy()
    n, m = mat_copy.shape
    for i in range(n):
        for j in range(m):
            mat_copy[i, j] = mat_copy[i, j].round(precision)
    return mat_copy


def save_latex(expr, filename):
    """
    Save LaTeX representation of a SymPy expression to file.

    Parameters
    ----------
    expr : Any
        SymPy expression to save.
    filename : str
        Path to file.

    Returns
    -------

    """
    with open(filename, 'w') as fp:
        fp.write(sp.latex(expr))


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


def symmetric(x: list) -> sp.Matrix:
    """
    Construct a symmetric 2D matrix from a list of values.

    Parameters
    ----------
    x : list
        List of 6 values.

    Returns
    -------
    Matrix
        Symmetric matrix constructed from `x`.

    Raises
    ------
    ValueError
        Length of `x` does not equal 6.

    """
    if len(x) != 6:
        raise ValueError

    t = sp.zeros(3, 3)

    t[0, 0] = x[0]
    t[0, 1] = x[1]
    t[0, 2] = x[2]
    t[1, 0] = t[0, 1]
    t[1, 1] = x[3]
    t[1, 2] = x[4]
    t[2, 0] = t[0, 2]
    t[2, 1] = t[1, 2]
    t[2, 2] = x[5]

    return t


def mean_deviator(matrix: sp.Matrix) -> Tuple[sp.Matrix, sp.Matrix]:
    """

    Parameters
    ----------
    matrix : Matrix
        Matrix from which to calculate the mean deviator.

    Returns
    -------
    (array, array) if `matrix` is a NumPy array.
    (Matrix, Matrix) if `matrix` is a SymPy matrix.

    Raises
    ------
    ValueError
        `matrix` must be either a NumPy array or SymPy matrix.

    """
    sympy = True if isinstance(matrix, sp.Matrix) else False
    numpy = True if isinstance(matrix, np.ndarray) else False
    if not (sympy ^ numpy):
        raise ValueError('`eps` must be a sympy Matrix or Numpy array.')

    def eval_sympy():
        e0 = matrix.trace() / 3
        mean_strain = sp.diag(*[e0] * 3)
        devaiatoric_strain = matrix - mean_strain
        return mean_strain, devaiatoric_strain

    def eval_numpy():
        e0 = np.trace(matrix) / 3
        mean_strain = np.diag([e0] * 3)
        deviatoric_strain = matrix - mean_strain
        return mean_strain, deviatoric_strain

    if numpy:
        return eval_numpy()
    return eval_sympy()


def elasticity_matrix(lame: sp.Integer = None, G: sp.Integer = None) -> Tuple[sp.Matrix, sp.Symbol, sp.Symbol]:
    """
    Compute the elasticity matrix :math:`C_{ij}`.

    Parameters
    ----------
    lame : Union[Integer, Float]
        Lame constant. Often called :math:`\lambda` in the literature.
    G : Union[Integer, Float]
        Shear modulus of elasticity.

    Returns
    -------
    Tuple[Matrix, Symbol, Symbol]
        Elasticity matrix, Lame constant symbol, and shear modulus of elasticity symbol.

    See Also
    --------
    solids.base.lame_constant

    """
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


def count_missing(*args):
    count = 0
    for arg in args:
        if not arg:
            count += 1
    return count


def compatibility_matrix(E=None, G=None, nu=None) -> Tuple[sp.Matrix, sp.Symbol, sp.Symbol, sp.Symbol]:
    """
    Compute the compatibility matrix :math:`D_{ij}`.

    Parameters
    ----------
    G : Union[Integer, Float]
        Shear modulus of elasticity.
    E : Union[Integer, Float]
        Modulus of elasticity.
    nu : Union[Integer, Float]
        Poisson's ratio.

    Returns
    -------
    Tuple[Matrix, Symbol, Symbol, Symbol]
        Compatibility matrix, E symbol, G symbol, nu symbol.

    """
    E_, G_, nu_ = sp.symbols('E, G, nu')

    D = sp.zeros(6, 6)
    D[:3, :3] = -sp.ones(3, 3) * nu_ / E_

    for i, j in zip(range(3), range(3, 6)):
        D[i, i] = 1 / E_
        D[j, j] = 1 / G_ / 2

    if count_missing(E, G, nu) == 1:
        E, G, nu = get_missing(E, G, nu)

        if E is not None:
            D = D.subs(E_, E)
        if G is not None:
            D = D.subs(G_, G)
        if nu is not None:
            D = D.subs(nu_, nu)

    return D, E_, G_, nu_


def get_missing(E=None, G=None, nu=None):
    """
    Calculate the third material property, if two are known.

    Parameters
    ----------
    E :
        Young's modulus
    G :
        Shear modulus
    nu
        Poisson's ratio

    Returns
    -------
    Tuple[Symbol, Symbol, Symbol]
        E, G, nu.

    """
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
    """
    Calculate the Lame constant :math:`\lambda`.

    Parameters
    ----------
    E : Union[Integer, Float]
        Elastic modulus
    nu : Union[Integer, Float]
        Poission's ratio

    Returns
    -------
    Union[Integer, Float]
        Lame constant :math:`\lambda`.

    """
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
    Matrix : (3, 3)
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

