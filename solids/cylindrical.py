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
        # C1 = sp.Integer(0)
        # C2 = (Pi * Ri ** 2 - Po * Ro ** 2) / 2 / (Ro ** 2 - Ri ** 2)
        # C3 = (Ri * Ro) ** 2 * (Po - Pi) / (Ro ** 2 - Ri ** 2)
        # C4, C5, C6 = sp.symbols("C_(4:7)")
        # if axial_symmetric:
        #     C4 = C5 = sp.Integer(0)
        #
        # u_t = (4 * C1 * r * theta + C4 * sp.cos(theta) - C5 * sp.sin(theta) + C6 * r) / E
        # u_r = (C1 * r * ((1 - nu) * (2 * sp.log(r) - 1) - 2 * nu)
        #        + 2 * C2 * (1 - nu) * r
        #        - C3 * (1 + nu) / r) / E + C4 * sp.sin(theta) + C5 * sp.cos(theta)

        # sig_rr = ((Pi * Ri ** 2 - Po * Ro ** 2) / (Ro ** 2 - Ri ** 2)
        #           + (Ri * Ro) ** 2 * (Po - Pi) / r ** 2 / (Ro ** 2 - Ri ** 2))
        # sig_tt = ((Pi * Ri ** 2 - Po * Ro ** 2) / (Ro ** 2 - Ri ** 2)
        #           - (Ri * Ro) ** 2 * (Po - Pi) / r ** 2 / (Ro ** 2 - Ri ** 2))
        sig_rr = -Pi * (1 - (Ro / r) ** 2) / (1 - (Ro / Ri) ** 2) - Po * (1 - (Ri/r)**2) / (1 - (Ri/Ro)**2)
        sig_tt = -Pi * (1 + (Ro / r) ** 2) / (1 - (Ro / Ri) ** 2) - Po * (1 + (Ri/r)**2) / (1 - (Ri/Ro)**2)
        sig_tot = sig_rr + sig_tt
        if plane == 'stress':
            sig_zz = sp.Integer(0)
        else:
            sig_zz = nu * sig_tot

        eps_rr = 1/E * (sig_rr - nu * (sig_tt + sig_zz))
        eps_tt = 1/E * (sig_tt - nu * (sig_rr + sig_zz))
        if plane == 'strain':
            eps_zz = sp.Integer(0)
        else:
            eps_zz = 1/E * (sig_zz - nu * (sig_rr + sig_tt))

        funcs = {'eps_rr': eps_rr, 'eps_tt': eps_tt, 'eps_zz': eps_zz,
                 'sig_rr': sig_rr, 'sig_tt': sig_tt, 'sig_tot': sig_tot, 'sig_zz': sig_zz}
        self._funcs = {k: v.simplify() for (k, v) in funcs.items()}

        for name, func in self._funcs.items():
            setattr(self, name, func)

        if display:
            self.display_funcs()

    def display_funcs(self):
        """
        Display all functions in Jupyter notebook.

        """
        func_names = ['eps_rr', 'eps_tt', 'eps_zz', 'sig_rr', 'sig_tt', 'sig_tot', 'sig_zz']
        latex_names = [r'\varepsilon_{rr}=', r'\varepsilon_{\theta\theta}=', r'\varepsilon_{zz}=', r'\sigma_{rr}=',
                       r'\sigma_{\theta\theta}=', r'\sigma_{tot}=', r'\sigma_{zz}=']
        assert len(func_names) == len(latex_names)
        for name, prefix in zip(func_names, latex_names):
            show(self._funcs[name], prefix)

    def subs(self, mapping: dict, inplace=False):
        funcs = self._funcs if inplace else self._funcs.copy()

        for name, function in funcs.items():
            funcs[name] = funcs[name].subs(mapping)

            if inplace:
                setattr(self, name, funcs[name])

        if not inplace:
            return funcs
