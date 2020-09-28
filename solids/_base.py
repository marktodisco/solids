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
    'save_latex'
]



def show(expr, prefix=None, postfix=None):
    latex_str = latex(expr)
    if prefix is not None:
        latex_str = prefix + latex_str
    if postfix is not None:
        latex_str = latex_str + postfix
    display(Math(latex_str))
    return None


def principal_stresses(sig: Matrix, symbol, precicion=100, dtype=None):    
    # Calculate the characteristic polynomial.
    A = sig - symbol*eye(*sig.shape)
    p = Eq(A.det(), 0)
    
    # Calculate the roots (i.e. the principal stresses).
    # Also, correct for numerical errors.
    sig_pr = solve(p, symbol)
    sig_pr = [s.n(precicion).round(24) for s in sig_pr]
    
    # Sort by magnitude, in decesning order
    sig_pr = sorted(sig_pr, key=lambda x: abs(x), reverse=True)
    if dtype is not None:
        if dtype is Matrix:
            sig_pr = Matrix(sig_pr)
        else:
            sig_pr = [dtype(ps) for ps in sig_pr]
    return sig_pr


def invariants(sig: Matrix, dtype: callable = Float):
    q = [float()] * 3
    q[0] = sig.trace()
    q[1] = sig[[1, 2], [1, 2]].det()
    q[1] += sig[[0, 1], [0, 1]].det() 
    q[1] += sig[[0, 2], [0, 2]].det()
    q[2] = sig.det()
    if dtype is not None:
        if dtype is Matrix:
            q = Matrix(q)
        else:
            q = [dtype(qi) for qi in q]
    return q


def octahedral_shear(SS: Matrix, dtype=Float):
    result = (sig_pr[0] - sig_pr[1])**2
    result += (sig_pr[1] - sig_pr[2])**2
    result += (sig_pr[2] - sig_pr[0])**2
    result = sqrt(result) / 3
    return dtype(result)


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