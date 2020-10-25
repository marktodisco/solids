from IPython.display import Math, display
import numpy as np
from sympy import *
import sympy as sp
from typing import List
from collections import deque


__all__ = [
    'show',
    'rotation_matrix',
    'matrix_round',
    'save_latex',
    'ref',
    'to_numpy',
    'char_poly',
    'symmetric',
    'gradient',
    'mean_normal_deviator',
    'cross'
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
    display(Math(latex_str))
    


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
    return matrix2numpy(R, dtype='float32') if numpy else R


def matrix_round(mat: Matrix, precision=0):
    mat_copy = mat.copy()
    n, m = mat_copy.shape
    for i in range(n):
        for j in range(m):
            mat_copy[i, j] = mat_copy[i, j].round(precision)
    return mat_copy


def save_latex(expr, filename):
    with open(filename, 'w') as fp:
        fp.write(sp.latex(expr))


def ref(S: Matrix, display: bool = True) -> sp.Matrix:
    """
    Tranform the matrix S to row echelon form.

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


def to_numpy(x: Matrix, diag: bool = False, dtype='float64'):
    """
    Covert a sympy matrix to a numpy array.
    
    Arguments
    ---------
    x : Matrix (N x M) | (N x 1) | (1 X M) | (1,)
        Sympy matrix to convert.
    diag : bool, optional
        If True, return a diagonal matrix consisting of the elements of x.
        To use this option, x must be a row or column vector, or a 1D array.
        Def
        ault is False.
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


def char_poly(sig: Matrix) -> Poly:
    
    _lambda = sp.symbols('lambda')
    size = min(sig.shape)
    A = sig - sp.diag(*[_lambda]*size)
    d = sp.Eq(A.det(), 0)
    return d, _lambda
    

def symmetric(sig: list) -> Matrix:
    
    t = zeros(3, 3)
    
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


def gradient(u: list, v: List[sp.Symbol]) -> sp.Matrix:
    
    n = len(u)
    m = len(v)
    
    u = sp.Matrix(u)
    Ju = sp.zeros(n, m)

    for i in range(n):
        for j in range(m):
            Ju[i, j] = u[i].diff(v[j])

    return Ju


def mean_normal_deviator(x: Matrix):
    x0 = sp.trace(x) / 3
    xm = sp.diag(*[x0]*3)
    xd = x - xm
    return xm, xd


def _even_odd_test(i, j, k, test='even'):
    permuation = deque([i, j, k])
    
    if test == 'even':
        baseline = deque([1, 2, 3])
    elif test == 'odd':
        baseline = deque([3, 2, 1])
        
    for _ in range(3):
        if permuation == baseline:
            return True
        permuation.rotate()
        
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
                c[i-1] += sp.Integer(_permutation(i, j, k)) * a[j-1] * b[k-1]
    
    return c



