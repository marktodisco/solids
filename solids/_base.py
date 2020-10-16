from IPython.display import Math, display
import numpy as np
from sympy import *
import sympy as sp
from typing import List


__all__ = [
    'StressState',
    'show',
    'principal_stresses',
    'invariants',
    'octahedral_shear',
    'octahedral_normal',
    'check_invariants',
    'rotation_matrix',
    'matrix_round',
    'max_shear',
    'save_latex',
    'ref',
    'stress_field',
    'to_numpy',
    'char_poly',
    'symmetric',
    'div',
    'strain_rotation_tensor',
    'mean_normal_deviator'
]


def show(expr, prefix=None, postfix=None):
    latex_str = latex(expr)
    if prefix is not None:
        latex_str = prefix + latex_str
    if postfix is not None:
        latex_str = latex_str + postfix
    display(Math(latex_str))
    return None


def principal_stresses(
        sig: Matrix, dtype='numpy', display: bool = True) -> Matrix:
    """
    Calculate the principal stress and principal axes of the stress matrix
    `sig`.

    Parameters
    ----------
    sig : Matrix
        Stress matrix.
    dtype : str, {'numpy', 'matrix'}, optional
        Return type of the solution. Use 'numpy' to return np.ndarray. Use
        'matrix' to return sympy.Matrix. The default is 'numpy'.
    display : bool, optional
        Display the results automatically.

    Returns
    -------
    Tuple[Matrix, Matrix]
        First element is the sorted principal stresses.
        The second element is the eigenvectors of sig in the form of a Matrix.

    Raises
    ------
    ValueError
        If dtype is not one of {'numpy', 'sympy'}.
        
    """
    if dtype == 'numpy':
        # Calculate the eigenvalues and eigenvectors
        sig = np.asarray(sig, dtype='float64')
        d, v = np.linalg.eig(sig)
        
        # Sort eigenvalues and eigenvectors
        sort_idx = sorted(range(d.size), key=lambda i: d[i], reverse=True)
        sorted_d = np.zeros_like(d)
        sorted_v = np.zeros_like(v)
        for i in range(d.size):
            sorted_d[i] = d[sort_idx[i]]
            sorted_v[:, i] = v[:, sort_idx[i]]
        
        ret = (sorted_d, sorted_v)
    
    elif dtype == 'sympy':
        # Calculate the eigevnvalues and eigenvectors.
        sig = Matrix(sig).n()
        pr = Matrix(sig).eigenvects(multiple=True)

        # Sort the eigenvalues and eigenvectors.
        if not sig.is_symbolic():
            pr = sorted(pr, key=lambda x: x[0], reverse=True)
        pr_sig = zeros(1, 3)
        pr_ax = zeros(3)

        for i in range(len(pr)):
            pr_sig[0, i] = pr[i][0]
            pr_ax[:, i] = pr[i][2]
            
        # Round for numerical precision.
        pr_sig = pr_sig.n(chop=True)
        pr_ax = pr_ax.n(chop=True)
        
        ret = (pr_sig, pr_ax)
    
    else:
        raise ValueError(f'Invalid input for dtype: {dtype}')
    
    if display:
        if dtype == 'sympy':
            show(ret[0], prefix=r"\sigma^{(i)}=")
            show(ret[1], prefix=r"\hat{n}_{i}^{*}=")
        
        elif dtype == 'numpy':
            print('pr_sig =\n', ret[0])
            print('pr_ax =\n', ret[1])
        
    return ret


def invariants(sig: Matrix) -> Matrix:
    """
    Compute the invariants if sig.

    Parameters
    ----------
    sig : Matrix
        Matrix whose invariants will be computed.

    Returns
    -------
    Matrix
        The invariants of sig.
    """
    if isinstance(sig, Matrix):
        q = zeros(3, 1)
        q[0] = sig.trace()
        q[1] = sig[[1, 2], [1, 2]].det()
        q[1] += sig[[0, 1], [0, 1]].det()
        q[1] += sig[[0, 2], [0, 2]].det()
        q[2] = sig.det()
    
    elif isinstance(sig, np.ndarray):
        temp = sig.copy()
        M = []
        for i in range(3):
            M.append(np.delete(np.delete(temp, i, axis=0), i, axis=1))
        q = np.zeros((3,))
        q[0] = sig.trace()
        q[1] = np.linalg.det(M[0])
        q[1] += np.linalg.det(M[1])
        q[1] += np.linalg.det(M[2])
        q[2] = np.linalg.det(sig)
    
    return q


def octahedral_shear(
        sig: np.ndarray, principal: bool = False) -> float:
    """
    Compute the shear component on an octahedral plane.

    Parameters
    ----------
    sig : (3, 3) np.ndarray
        Stress matrix.
    principal : bool, optional
        If True, allows the user to pass a 1D array of principal stresses to
        sig. Default if False.

    Returns
    -------
    float
        The shear component of stress on an octehedral plane.
    
    """
    if not isinstance(sig, np.ndarray):
        sig = np.asarray(sig, dtype='float64')
    
    if not principal:
        pr_sig, _ = principal_stresses(sig, dtype='numpy', display=False)
    else:
        if 1 in sig.shape:
            pr_sig = np.asarray(sig).flatten()
    
    result = (pr_sig[0] - pr_sig[1])**2
    result += (pr_sig[1] - pr_sig[2])**2
    result += (pr_sig[2] - pr_sig[0])**2
    result = sqrt(result) / 3
    
    return float(result)


def octahedral_normal(sig: np.ndarray, principal=False) -> float:
    """
    Compute the normal component of stress on an octahedral plane.

    Parameters
    ----------
    sig : (3, 3) np.ndarray
        Stress matrix.
    principal : bool, optional
        If True, allows the user to pass a 1D array of principal stresses to
        sig. Default if False.

    Returns
    -------
    float
        The normal component of stress on an octahedral plane.
    
    """
    if not isinstance(sig, np.ndarray):
        sig = np.asarray(sig, dtype='float64')
    
    if not principal:
        pr_sig, _ = principal_stresses(sig, dtype='numpy', display=False)
    else:
        pr_sig = np.asarray(sig).flatten()
            
    return pr_sig.mean()


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


def ref(S: Matrix, display: bool = True) -> Matrix:
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
    
    _lambda = symbols('lambda')
    size = min(sig.shape)
    A = sig - diag(*[_lambda]*size)
    d = Eq(A.det(), 0)
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


def max_shear(sigma: np.ndarray, principal=False) -> float:
    if not isinstance(sigma, np.ndarray):
        sigma = np.asarray(sigma, dtype='float64')
        
    if not principal:
        pr_stress, _ = principal_stresses(sigma, dtype='numpy', display=False)
    else:
        pr_stress = sigma
        
    return 0.5 * (pr_stress.max() - pr_stress.min())


class StressState:
    
    def __init__(self, sigma, dtype):
        self._sigma = sigma
        self._sigma_sympy = Matrix(sigma)
        self._dtype = dtype
        self._convert()
            
        self._pr_stress = None
        self._pr_ax = None
        self._invariants = None
        self._octahedral_shear = None
        self._octahedral_normal = None
        self._max_shear = None
        self._char_poly = None
        
    def full_report(self):
        self._get_principal()
        _ = self.invariants
        _ = self.char_poly
        
        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                _ = self.octahedral_normal
                _ = self.octahedral_shear
                _ = self.max_shear
        
        if self._dtype == 'numpy':
            self._report_numpy()
        elif self._dtype == 'sympy':
            self._report_sympy()
        return
        
    @property
    def principal_stresses(self):
        if self._pr_stress is None:
            self._get_principal()
        return self._pr_stress
    
    @property
    def principal_axes(self):
        if self._pr_ax is None:
            self._get_principal()
        return self._pr_ax
    
    @property
    def invariants(self):
        if self._invariants is None:
            self._invariants = invariants(self._sigma)
        return self._invariants
    
    @property
    def octahedral_shear(self):
        if self._octahedral_shear is None:
            self._octahedral_shear = octahedral_shear(
                self._sigma, principal=False)
        
        return self._octahedral_shear
    
    @property
    def octahedral_normal(self):
        if self._octahedral_normal is None:
            self._octahedral_normal = octahedral_normal(
                self._sigma, principal=False)
        
        return self._octahedral_normal
    
    @property
    def max_shear(self):
        if self._max_shear is None:
            if self._pr_stress is None:
                self._pr_stress, _ = principal_stresses(
                    self._sigma, dtype='numpy', display=False)
            
            self._max_shear = max_shear(self._pr_stress, principal=True)
        
        return self._max_shear
    
    @property
    def char_poly(self):
        if self._char_poly is None:
            self._char_poly = char_poly(self._sigma_sympy)
        
        return self._char_poly
    
    def _report_numpy(self):
        self._custom_print('State of Stress', self._sigma)
        self._custom_print('Principal Stresses', self._pr_stress)
        self._custom_print('Principal Axes', self._pr_ax)
        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                self._custom_print('Octahedral Normal Stress', self._octahedral_normal)
                self._custom_print('Octahedral Shear Stress', self._octahedral_shear)
                self._custom_print('Max Shear Stress', self._max_shear)
        self._custom_print('Invariants', self._invariants)
        self._custom_print('Characteristic Polynomial', self._char_poly[0].lhs)
        return
    
    @staticmethod
    def _custom_print(title, value, w=79):
        print('-' * w)
        print(title)
        print('-' * w)
        print(value)
        print('\n')
        return
    
    def _report_sympy(self):
        show(self._sigma, r"\text{State of Stress}\\\sigma=", r"\\")
        show(self._pr_stress, r"\text{Principal Stresses}\\\sigma^{(i)}=", r"\\")
        show(self._pr_ax, r"\text{Principal Axes}\\\hat{n}^{(i)}=", r"\\")
        if self._dtype == 'sympy':
            if not self._sigma.is_symbolic():
                show(self._octahedral_normal, r"\text{Octahedral Normal Stress}\\\sigma_{nn}^{oct}=", r"\\")
                show(self._octahedral_shear, r"\text{Octahedral Shear Stress}\\\sigma_{ns}^{oct}=", r"\\")
                show(self._max_shear, r"\text{Max Shear Stress}\\\sigma_{oct}^{max}=", r"\\")
        show(self._invariants, r"\text{Invariants}\\Q_i=", r"\\")
        show(self._char_poly[0].lhs, r"\text{Characteristic Polynomial}\\p=", r"\\")
    
    def _get_principal(self):
        self._pr_stress, self._pr_ax = principal_stresses(
            self._sigma, display=False, dtype=self._dtype)
    
    def _convert(self):
        if self._dtype == 'numpy':
            self._sigma = np.asarray(self._sigma, dtype='float64')

        elif self._dtype == 'sympy':
            self._pr_stress = Matrix(self._sigma)
    
    def __str__(self):
        if self._dtype == 'numpy':
            return str(self._sigma)
        if self._dtype == 'sympy':
            show(self._sigma)
            return ''
    
    def __repr__(self):
        return self.__str__()


def div(u: Matrix, v: List[Symbol]) -> Matrix:
    Ju = zeros(3, 3)

    for i in range(3):
        for j in range(3):
            Ju[i, j] = u[i].diff(v[j])

    return Ju


def strain_rotation_tensor(u: {Matrix, list}, v: List[Symbol]):
    "0.5 * (u_i,j + u_j,i)"
    if isinstance(u, Matrix):
        eps = zeros(3, 3)
        omg = zeros(3, 3)
    elif isinstance(u, np.ndarray):
        eps = np.zeros(3, 3)
        omg = np.zeros(3, 3)
    else:
        raise ValueError('u  must be of type {Matrix, list}.')

    Ju = div(u, v)
    
    for i in range(3):
        for j in range(3):
            eps[i, j] = (Ju[i, j] + Ju[j, i]) / 2
            omg[i, j] = (Ju[i, j] - Ju[j, i]) / 2
            
    return eps, omg


def mean_normal_deviator(x: Matrix):
    x0 = sp.trace(x) / 3
    xm = sp.diag(*[x0]*3)
    xd = x - xm
    return xm, xd
