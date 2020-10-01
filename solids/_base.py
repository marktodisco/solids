from IPython.display import Math, display
import numpy as np
from sympy import *


__all__ = [
    'show',
    'principal_stresses',
    'invariants',
    'octahedral_shear',
    'check_invariants',
    'rotation_matrix',
    'matrix_round',
    'save_latex',
    'ref',
    'stress_field',
    'to_numpy'
]


def show(expr, prefix=None, postfix=None):
    latex_str = latex(expr)
    if prefix is not None:
        latex_str = prefix + latex_str
    if postfix is not None:
        latex_str = latex_str + postfix
    display(Math(latex_str))
    return None


def principal_stresses(sig: Matrix, precicion=100, dtype=None):
    # Calculate the characteristic polynomial.
    symbol = symbols('__lambda__')
    A = sig - symbol*eye(*sig.shape)
    p = Eq(A.det(), 0)
    
    # Calculate the roots (i.e. the principal stresses).
    # Also, correct for numerical errors.
    sig_pr = solve(p, symbol)
    sig_pr = [s.n(precicion).round(24) for s in sig_pr]
    
    # Sort by magnitude, in decesning order
    # sig_pr = sorted(sig_pr, key=lambda x: abs(x), reverse=True)
    sig_pr = sorted(sig_pr, reverse=True)
    if dtype is not None:
        if dtype is Matrix:
            sig_pr = Matrix(sig_pr)
        else:
            sig_pr = [dtype(ps) for ps in sig_pr]
    return sig_pr


def invariants(sig: Matrix, dtype: callable = Matrix):
    q = zeros(3, 1)
    q[0] = sig.trace()
    q[1] = sig[[1, 2], [1, 2]].det()
    q[1] += sig[[0, 1], [0, 1]].det()
    q[1] += sig[[0, 2], [0, 2]].det()
    q[2] = sig.det()
    # if dtype is not None:
    #     if dtype is Matrix:
    #         q = Matrix(q)
    #     else:
    #         q = [dtype(qi) for qi in q]
    return q


def octahedral_shear(sig: np.ndarray, principal=False):        
    if not principal:
        sig_pr = principal_stresses(sig)
    else:
        if 1 in sig.shape:
            sig_pr = np.asarray(sig).flatten()
    
    result = (sig_pr[0] - sig_pr[1])**2
    result += (sig_pr[1] - sig_pr[2])**2
    result += (sig_pr[2] - sig_pr[0])**2
    result = sqrt(result) / 3
    
    return float(result)


def check_invariants(Q, SS, symbol):
    def check(x):
        return x**3 - Q[0]*x**2 + Q[1]*x - Q[2]
    return [float(check(s).round(16)) for s in SS]


def rotation_matrix(theta, unit='rad', numpy=False):
    if unit == 'deg':
        theta = rad(theta)
    R = Matrix([[cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
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
        fp.write(latex(expr))


def ref(S, display=True):
    Sc = S.copy()
    Sc = matrix_round(Sc.n(100), 32)
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


def stress_field(x: Matrix, v: list):
    """
    Calculate the stress field, and determine admissibility.
    
    Arguments
    ---------
    x : sympy.matrices.dense.MutableDenseMatrix
        State of stress in matrix form.
    v : sympy.core.symbol.Symbol
        list of sympy variables
    
    Returns
    -------
    sf : sympy.matrices.dense.MutableDenseMatrix
        Stress corresponding to `x` and `v`.
    admissible : bool
        Return True if the state of stress is admissable; False otherwise.
    """
    
    xc = x.copy()
    n, m = xc.shape
    
    assert len(v) == m, \
        "Number of variables must match the number of columns of x."
    
    # Compute the derivatives.
    for i in range(n):
        for j in range(m):
            xc[i, j] = xc[i, j].diff(v[j])
            
    # Sum the rows and test for equilibrium.
    all_zero = 0.
    for i in range(n):
        all_zero += sum(xc[i, :])
    
    if not all_zero:
        return xc, True
    
    return xc, False


def to_numpy(x: Matrix, diag=False, dtype='float64'):
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