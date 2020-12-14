from typing import Union, Tuple

import sympy as sp

from solids import show

__all__ = [
    'from_cartesian',
    'ThickWalledCylinder'
]


def from_cartesian(sigma: sp.Matrix, theta: Union[sp.Integer, sp.Float] = None
                   ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Convert cartesian matrix to cylindrical coordinate system.

    Parameters
    ----------
    sigma : Matrix
        2D matrix
    theta : Union[sp.Integer, sp.Float]
        Rotation angle

    Returns
    -------
    Tuple[Matrix, Matrix, Matrix]
        sig_rr, sig_tt, sig_rt

    """
    sin = sp.sin
    cos = sp.cos

    sig_xx = sigma[0, 0]
    sig_yy = sigma[1, 1]
    sig_xy = sigma[0, 1]

    sig_rr = sig_xx * cos(theta) ** 2 + sig_yy * sin(theta) ** 2 + 2 * sig_xy * sin(theta) * cos(theta)
    sig_tt = sig_xx * sin(theta) ** 2 + sig_yy * cos(theta) ** 2 - 2 * sig_xy * sin(theta) * cos(theta)
    sig_rt = (-sig_xx + sig_yy) * sin(theta) * cos(theta) + sig_xy * (cos(theta) ** 2 - sin(theta) ** 2)

    return sig_rr, sig_tt, sig_rt


class ThickWalledCylinder:
    """
    Make functions for the case of a thick-walled cylinder.

    Parameters
    ----------
    axial_symmetric : bool, optional
        Default is false
    plane : {'stress', 'strain'}
        Plane condition
    display : bool, optional
        Display in Jupyter notebook. Default is False.
    Ri : Union[Integer, Float]
        Inner radius
    Ro : Union[Integer, Float]
        Outer radius
    Pi : Union[Integer, Float]
        Internal pressure
    Po : Union[Integer, Float]
        External pressure

    Attributes
    ----------
    u_t
    u_r
    eps_rr
    eps_tt
    eps_zz
    sig_rr
    sig_tt
    sig_tot
    sig_zz

    """
    def __init__(self,
                 axial_symmetric=False,
                 plane=None,
                 display=False,
                 Po=None,
                 Pi=None,
                 Ro=None,
                 Ri=None,
                 E=None,
                 nu=None):
        """
        Make functions for the case of a thick-walled cylinder.

        Parameters
        ----------
        axial_symmetric : bool, optional
            Default is false
        plane : {'stress', 'strain'}
            Plane condition
        display : bool, optional
            Display in Jupyter notebook. Default is False.
        Ri : Union[Integer, Float]
            Inner radius
        Ro : Union[Integer, Float]
            Outer radius
        Pi : Union[Integer, Float]
            Internal pressure
        Po : Union[Integer, Float]
            External pressure

        """
        Po = sp.Symbol('P_o') if Po is None else Po
        Pi = sp.Symbol('P_i') if Pi is None else Pi
        Ro = sp.Symbol('R_o') if Ro is None else Ro
        Ri = sp.Symbol('R_i') if Ri is None else Ri
        E = sp.Symbol('E') if E is None else E
        nu = sp.Symbol('nu') if nu is None else nu

        r, theta = sp.symbols('r, theta')
        C1 = sp.Integer(0)
        C2 = (Pi * Ri ** 2 - Po * Ro ** 2) / 2 / (Ro ** 2 - Ri ** 2)
        C3 = (Ri * Ro) ** 2 * (Po - Pi) / (Ro ** 2 - Ri ** 2)
        C4, C5, C6 = sp.symbols("C_(4:7)")
        if axial_symmetric:
            C4 = C5 = sp.Integer(0)

        u_t = (4 * C1 * r * theta + C4 * sp.cos(theta) - C5 * sp.sin(theta) + C6 * r) / E
        u_r = (C1 * r * ((1 - nu) * (2 * sp.log(r) - 1) - 2 * nu)
               + 2 * C2 * (1 - nu) * r
               - C3 * (1 + nu) / r
               + C4 * sp.sin(theta)
               + C5 * sp.cos(theta)) / E

        eps_rr = u_r.diff(r)
        eps_tt = (u_r + u_t.diff(theta)) / r
        eps_zz = 2 * nu / E * (Po * Ro ** 2 - Pi * Ri ** 2) / (Ro ** 2 - Ri ** 2)

        sig_rr = ((Pi * Ri ** 2 - Po * Ro ** 2) / (Ro ** 2 - Ri ** 2)
                  + (Ri * Ro) ** 2 * (Po - Pi) / r ** 2 / (Ro ** 2 - Ri ** 2))
        sig_tt = ((Pi * Ri ** 2 - Po * Ro ** 2) / (Ro ** 2 - Ri ** 2)
                  - (Ri * Ro) ** 2 * (Po - Pi) / r ** 2 / (Ro ** 2 - Ri ** 2))
        sig_tot = sig_rr + sig_tt
        sig_zz = nu * sig_tot

        if plane == 'stress':
            sig_zz = sp.Integer(0)
        if plane == 'strain':
            eps_zz = sp.Integer(0)

        funcs = {'u_r': u_r, 'u_t': u_t,
                 'eps_rr': eps_rr, 'eps_tt': eps_tt, 'eps_zz': eps_zz,
                 'sig_rr': sig_rr, 'sig_tt': sig_tt, 'sig_tot': sig_tot, 'sig_zz': sig_zz}
        self._funcs = {k: v.simplify() for (k, v) in funcs.items()}

        if display:
            self.display_funcs()

        self.u_t = u_t
        self.u_r = u_r
        self.eps_rr = eps_rr
        self.eps_tt = eps_tt
        self.eps_zz = eps_zz
        self.sig_rr = sig_rr
        self.sig_tt = sig_tt
        self.sig_tot = sig_tot
        self.sig_zz = sig_zz

    def display_funcs(self):
        """
        Display all functions in Jupyter notebook.

        """
        show(self._funcs['u_r'], r'u_{r}=')
        show(self._funcs['u_t'], r'u_{\theta}=')
        show(self._funcs['eps_rr'], r'\varepsilon_{rr}=')
        show(self._funcs['eps_tt'], r'\varepsilon_{\theta\theta}=')
        show(self._funcs['eps_zz'], r'\varepsilon_{zz}=')
        show(self._funcs['sig_rr'], r'\sigma_{rr}=')
        show(self._funcs['sig_tt'], r'\sigma_{\theta\theta}=')
        show(self._funcs['sig_tot'], r'\sigma_{tot}=')
        show(self._funcs['sig_zz'], r'\sigma_{zz}=')

    def subs(self, mapping: dict, inplace=False):
        funcs = self._funcs if inplace else self._funcs.copy()

        for name, function in funcs.items():
            funcs[name] = funcs[name].subs(mapping)

            if inplace:
                setattr(self, name, funcs[name])

        if not inplace:
            return funcs
