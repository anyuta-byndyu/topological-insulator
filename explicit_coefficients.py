"""Явные формулы для r, t, a_j, b_j (j = 1..7). Обозначения:
kappa = pi k R, thL = pi*Ф/Ф0, thS = pi*Ф/(4Ф0), s = sin kappa, sig = sin(kappa/2), gam = cos(kappa/2), c = cos thS
Dt = i s - 4cos kappa - 2 + 2c^2,  Qt = 2(cos thL + c^2),  Delta = Dt^2 - Qt^2 = (Dt - Qt)(Dt + Qt)
"""
import sympy as sp
k, R, F = sp.symbols('k R Phi_over_Phi0', positive=True)
I = sp.I
ka = sp.pi * k * R
thL, thS = sp.pi * F, sp.pi * F / 4
s, gam, c = sp.sin(ka), sp.cos(ka / 2), sp.cos(thS)
Dt = I * s - 4 * sp.cos(ka) - 2 + 2 * c**2
Qt = 2 * (sp.cos(thL) + c**2)
Delta = Dt**2 - Qt**2
E = sp.exp
coef = {
    'r':  (2 * I * s * Dt - Delta) / Delta,
    't':  -2 * I * s * Qt / Delta,
    'a1': (Dt * E(-I*thL) + Qt * E(-I*ka)) / Delta,
    'b1': -(Dt * E(-I*thL) + Qt * E(I*ka)) / Delta,
    'a2': -(Qt * E(-I*thL) + Dt * E(-I*ka)) / Delta,
    'b2': (Dt * E(I*ka) + Qt * E(-I*thL)) / Delta,
    'a3': (2*gam*Dt*E(-I*thS) - c*(Dt - Qt)*E(-I*ka/2)) / Delta,
    'b3': (c*(Dt - Qt)*E(I*ka/2) - 2*gam*Dt*E(-I*thS)) / Delta,
    'a4': (c*(Dt - Qt)*E(-I*thS) - 2*gam*Dt*E(-I*ka/2)) / Delta,
    'b4': (2*gam*Dt*E(I*ka/2) - c*(Dt - Qt)*E(-I*thS)) / Delta,
    'a5': (c*(Dt - Qt)*E(-I*thS) + 2*gam*Qt*E(-I*ka/2)) / Delta,
    'b5': -(2*gam*Qt*E(I*ka/2) + c*(Dt - Qt)*E(-I*thS)) / Delta,
    'a6': -(2*gam*Qt*E(-I*thS) + c*(Dt - Qt)*E(-I*ka/2)) / Delta,
    'b6': (c*(Dt - Qt)*E(I*ka/2) + 2*gam*Qt*E(-I*thS)) / Delta,
}
coef['a7'], coef['b7'] = coef['t'], sp.Integer(0)

if __name__ == '__main__':
    import numpy as np, graph_full as g
    f = {nm: sp.lambdify((k, R, F), ex, 'numpy') for nm, ex in coef.items()}
    rng = np.random.default_rng(11); err = 0
    for _ in range(500):
        kv, Rv, Fv = rng.uniform(.05, 4), rng.uniform(.5, 2), rng.uniform(0, 4)
        if abs(kv*Rv - round(kv*Rv)) < 1e-2: continue
        for nm in ['r', 't'] + [f'{x}{j}' for j in range(1, 7) for x in 'ab']:
            err = max(err, abs(f[nm](kv, Rv, Fv) - g.f_coef[nm](kv, Rv, g.n_from_flux(Fv, Rv))))
    print('max |явная формула - проверенное решение| =', err)
