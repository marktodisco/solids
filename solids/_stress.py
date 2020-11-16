from typing import Union, Tuple

import numpy as np
import sympy as sp

from ._base import show, char_poly

__all__ = [
    'principal_stresses',
    'invariants',
    'octahedral_shear',
    'octahedral_normal',
    'check_invariants',
    'stress_field',
    'StressState',
    'max_shear'
]


def principal_stresses(sig: Union[sp.Matrix, np.ndarray],
                       dtype='numpy',
                       display: bool = True
                       ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[sp.Matrix, sp.Matrix]]:
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
        sort_idx = sorted(range(d.size), key=lambda x: d[x], reverse=True)
        sorted_d = np.zeros_like(d)
        sorted_v = np.zeros_like(v)
        for i in range(d.size):
            sorted_d[i] = d[sort_idx[i]]
            sorted_v[:, i] = v[:, sort_idx[i]]
        
        ret = (sorted_d, sorted_v)
    
    elif dtype == 'sympy':
        # Calculate the eigenvalues and eigenvectors.
        sig = sp.Matrix(sig).n()
        pr = sp.Matrix(sig).eigenvects(multiple=True)

        # Sort the eigenvalues and eigenvectors.
        if not sig.is_symbolic():
            pr = sorted(pr, key=lambda x: x[0], reverse=True)
        pr_sig = sp.zeros(1, 3)
        pr_ax = sp.zeros(3)

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


def invariants(sig: Union[sp.Matrix, np.ndarray]) -> Union[sp.Matrix, np.ndarray]:
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
    if isinstance(sig, sp.Matrix):
        q = sp.zeros(3, 1)
        q[0] = sig.trace()
        q[1] = sum([sig.minor(i, i) for i in range(3)])
        q[2] = sig.det()
    
    elif isinstance(sig, np.ndarray):
        temp = sig.copy()
        M = []

        for i in range(3):
            M.append(np.delete(np.delete(temp, i, axis=0), i, axis=1))

        q = np.zeros((3,))
        q[0] = sig.trace()
        q[1] = sum([np.linalg.det(m) for m in M])
        q[2] = np.linalg.det(sig)

    else:
        raise ValueError("Argument `sig` must be a sympy matrix or numpy array.")
    
    return q


def octahedral_shear(sig: np.ndarray, principal: bool = False):
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
    The shear component of stress on an octahedral plane.
    
    """
    if isinstance(sig, (list, sp.Matrix)):
        sig = np.asarray(sig, dtype='float64')
    else:
        raise ValueError("Argument `sig` must be a numpy array, list, or sympy matrix.")

    if principal:
        pr_sig = np.asarray(sig).flatten()
    else:
        pr_sig, _ = principal_stresses(sig, dtype='numpy', display=False)
    
    result = (pr_sig[0] - pr_sig[1])**2
    result += (pr_sig[1] - pr_sig[2])**2
    result += (pr_sig[2] - pr_sig[0])**2
    result = sp.sqrt(result) / 3
    
    return result


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


def check_invariants(Q, SS):
    def check(x):
        return x**3 - Q[0]*x**2 + Q[1]*x - Q[2]
    return [check(s).n(chop=True) for s in SS]


def stress_field(x: sp.Matrix, v: list):
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
        Return True if the state of stress is admissible; False otherwise.

    """
    xc = x.copy()
    n, m = xc.shape
    
    assert len(v) == n, "Number of variables must match the number of columns of x."
    
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
        self._sigma_sympy = sp.Matrix(sigma)
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
            self._pr_stress = sp.Matrix(self._sigma)
    
    def __str__(self):
        if self._dtype == 'numpy':
            return str(self._sigma)
        if self._dtype == 'sympy':
            show(self._sigma)
            return ''
    
    def __repr__(self):
        return self.__str__()
