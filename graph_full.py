"""
Квантовый граф: большая окружность R (рёбра 1, 2), две малые R/2 (рёбра 3,4 и 5,6),
подводящее ребро 0 (x in (-inf, 0], x=0 в вершине 1) и отводящее 7 (x in [0, +inf), x=0 в вершине 3).

Функции на рёбрах:
    phi_0 = e^{ikx} + r e^{-ikx}
    phi_7 = t e^{ikx}
    phi_j = a_j e^{i(k + n R_j) x} + b_j e^{i(-k + n R_j) x},   n = B*pi/Ф0,  j = 1..6
    R_j = R (рёбра 1, 2),  R_j = R/2 (рёбра 3..6);  длина ребра L_j = pi R_j.
Обозначения: X_j = e^{i(k + nR_j) L_j},  Y_j = e^{i(-k + nR_j) L_j}
    phi_j(0) = a_j + b_j,      phi_j(L_j) = a_j X_j + b_j Y_j

Вершинные условия: непрерывность + Кирхгоф для D = d/dx - i n R_j:
    sum_{исходящие} D phi_j(0) - sum_{входящие} D phi_j(L_j) = 0
    D phi_j(0) = ik (a_j - b_j),   D phi_j(L_j) = ik (a_j X_j - b_j Y_j)

P1 = int|phi_1|^2 + int|phi_2|^2           (большая окружность)
P2 = int|phi_3|^2 + ... + int|phi_6|^2     (малые окружности)
"""
import numpy as np
import sympy as sp

IN = {1: [0, 3, 1], 2: [4, 5], 3: [6, 2]}      # входящие рёбра
OUT = {1: [4, 2], 2: [3, 6], 3: [5, 7, 1]}     # исходящие рёбра
EDGES = {}                                     # j -> (начало, конец)
for v in (1, 2, 3):
    for j in OUT[v]: EDGES.setdefault(j, [None, None])[0] = v
    for j in IN[v]:  EDGES.setdefault(j, [None, None])[1] = v
RING = {j: tuple(EDGES[j]) for j in range(1, 7)}  # {1:(3,1), 2:(1,3), 3:(2,1), 4:(1,2), 5:(3,2), 6:(2,3)}
BIG, SMALL = (1, 2), (3, 4, 5, 6)

# ===================================================== символы
k, R, n = sp.symbols('k R n', positive=True)
XL, YL, XS, YS = sp.symbols('X_L Y_L X_S Y_S')
XY = {j: (XL, YL) if j in BIG else (XS, YS) for j in RING}
Rj = {j: R if j in BIG else R / 2 for j in RING}
Lj = {j: sp.pi * Rj[j] for j in RING}



# ===================================================== 1. решение
def solve_symbolic():
    """
    Неизвестные — значения в вершинах U1 = phi(v1) = 1 + r, U2, U3 = phi(v3) = t.
    Ребро u -> w однозначно определено концами (два граничных условия на ОДУ 2-го порядка):
        a = (w - u Y)/(X - Y),   b = (u X - w)/(X - Y)
    Тогда непрерывность выполнена автоматически, а Кирхгоф (делённый на ik):
        исходящее:  D phi(0)/ik = (2w - u(X+Y)) / (X - Y)
        входящее:   D phi(L)/ik = (w(X+Y) - 2uXY) / (X - Y)
        ребро 0 (входит в v1):  D phi_0(0)/ik = 1 - r = 2 - U1
        ребро 7 (выходит из v3): D phi_7(0)/ik = t = U3
    """
    U = sp.symbols('U1:4')
    ab = {}
    for j, (s, e) in RING.items():
        X, Y = XY[j]; u, w = U[s - 1], U[e - 1]
        ab[j] = ((w - u * Y) / (X - Y), (u * X - w) / (X - Y))
    D0 = {j: ab[j][0] - ab[j][1] for j in RING}
    DL = {j: ab[j][0] * XY[j][0] - ab[j][1] * XY[j][1] for j in RING}
    kir = []
    for v in (1, 2, 3):
        eq = 0
        for j in OUT[v]:
            eq += U[2] if j == 7 else D0[j]
        for j in IN[v]:
            eq -= (2 - U[0]) if j == 0 else DL[j]
        kir.append(sp.expand(sp.cancel(eq * (XL - YL) * (XS - YS))))
    A, rhs = sp.linear_eq_to_matrix(kir, U)
    det = sp.factor(A.det(method='berkowitz'))
    Usol = {}
    for i in range(3):
        Ai = A.copy(); Ai[:, i] = rhs
        Usol[U[i]] = sp.factor(sp.cancel(Ai.det(method='berkowitz') / det))
    out = {'r': sp.factor(sp.cancel(Usol[U[0]] - 1)), 't': Usol[U[2]],
           'U1': Usol[U[0]], 'U2': Usol[U[1]], 'U3': Usol[U[2]]}
    for j in RING:
        out[f'a{j}'] = sp.factor(sp.cancel(ab[j][0].subs(Usol)))
        out[f'b{j}'] = sp.factor(sp.cancel(ab[j][1].subs(Usol)))
    return out

def P_edge(a, b, X, Y, L):
    return (sp.Abs(a) ** 2 + sp.Abs(b) ** 2) * L + sp.re(a * sp.conjugate(b) * (X / Y - 1) / (sp.I * k))


SOL = solve_symbolic()
P = {j: P_edge(SOL[f'a{j}'], SOL[f'b{j}'], *XY[j], Lj[j]) for j in RING}
P1 = P[1] + P[2]
P2 = P[3] + P[4] + P[5] + P[6]
_xy_args = (k, R, n, XL, YL, XS, YS)


def _xy(kv, Rv, nv):
    return (np.exp(1j * (kv + nv * Rv) * np.pi * Rv), np.exp(1j * (-kv + nv * Rv) * np.pi * Rv),
            np.exp(1j * (kv + nv * Rv / 2) * np.pi * Rv / 2), np.exp(1j * (-kv + nv * Rv / 2) * np.pi * Rv / 2))


def _make(expr, real=False):
    f = sp.lambdify(_xy_args, expr, 'numpy')
    if real:
        return lambda kv, Rv, nv: np.real(f(kv, Rv, nv, *_xy(kv, Rv, nv)))
    return lambda kv, Rv, nv: f(kv, Rv, nv, *_xy(kv, Rv, nv))


f_coef = {name: _make(ex) for name, ex in SOL.items()}       # r, t, a1..b6, U1..U3
f_P = {j: _make(P[j], real=True) for j in RING}
f_P1, f_P2 = _make(P1, real=True), _make(P2, real=True)
f_ratio = lambda kv, Rv, nv: f_P1(kv, Rv, nv) / f_P2(kv, Rv, nv)


def n_from_flux(flux, Rv):
    return flux / Rv ** 2

def direct_14x14(kv, Rv, nv):
    X = {j: np.exp(1j * (kv + nv * float(Rj[j].subs(R, Rv))) * np.pi * float(Rj[j].subs(R, Rv))) for j in RING}
    Y = {j: np.exp(1j * (-kv + nv * float(Rj[j].subs(R, Rv))) * np.pi * float(Rj[j].subs(R, Rv))) for j in RING}
    def e(): return np.zeros(14, complex)
    def val(j, end):
        v = e()
        if j == 0: v[0] = 1; return v, 1.0          # phi_0(0) = 1 + r
        if j == 7: v[1] = 1; return v, 0.0          # phi_7(0) = t
        v[2 * j], v[2 * j + 1] = (X[j], Y[j]) if end else (1, 1); return v, 0.0
    def der(j, end):                                 # D phi / ik
        v = e()
        if j == 0: v[0] = -1; return v, 1.0          # 1 - r  ->  (-1)*r + 1
        if j == 7: v[1] = 1; return v, 0.0
        v[2 * j], v[2 * j + 1] = (X[j], -Y[j]) if end else (1, -1); return v, 0.0
    rows, rhs = [], []
    for vtx in (1, 2, 3):
        vals = [val(j, False) for j in OUT[vtx]] + [val(j, True) for j in IN[vtx]]
        for (v, c) in vals[1:]:
            rows.append(v - vals[0][0]); rhs.append(vals[0][1] - c)
        kr, kc = e(), 0.0
        for j in OUT[vtx]: d, c = der(j, False); kr += d; kc += c
        for j in IN[vtx]:  d, c = der(j, True);  kr -= d; kc -= c
        rows.append(kr); rhs.append(-kc)
    return np.linalg.solve(np.array(rows), np.array(rhs, complex))


def run_checks(trials=300, seed=7):
    rng = np.random.default_rng(seed)
    names = ['r', 't'] + [f'{c}{j}' for j in range(1, 7) for c in 'ab']
    w = dict(unitarity=0., vs_14x14=0., integrals=0., residual_sym=0.)
    for _ in range(trials):
        kv, Rv, nv = rng.uniform(0.2, 6), rng.uniform(0.5, 2), rng.uniform(-1.5, 1.5)
        c = {nm: f_coef[nm](kv, Rv, nv) for nm in names}
        w['unitarity'] = max(w['unitarity'], abs(abs(c['r']) ** 2 + abs(c['t']) ** 2 - 1))
        d = direct_14x14(kv, Rv, nv)
        w['vs_14x14'] = max(w['vs_14x14'], max(abs(c[nm] - d[i]) for i, nm in enumerate(names)))
        for j in RING:
            Rv_j = Rv if j in BIG else Rv / 2; L = np.pi * Rv_j
            x = np.linspace(0, L, 20001)
            phi = c[f'a{j}'] * np.exp(1j * (kv + nv * Rv_j) * x) + c[f'b{j}'] * np.exp(1j * (-kv + nv * Rv_j) * x)
            q = np.trapezoid(np.abs(phi) ** 2, x)
            w['integrals'] = max(w['integrals'], abs(q - f_P[j](kv, Rv, nv)) / q)
    U = {i: SOL[f'U{i}'] for i in (1, 2, 3)}
    for _ in range(10):
        vals = {k: rng.uniform(.3, 5), XL: complex(*rng.normal(size=2)), YL: complex(*rng.normal(size=2)),
                XS: complex(*rng.normal(size=2)), YS: complex(*rng.normal(size=2))}
        ev = lambda ex: complex(ex.subs(vals))
        for j, (s, e_) in RING.items():
            X, Y = XY[j]
            w['residual_sym'] = max(w['residual_sym'],
                                    abs(ev(SOL[f'a{j}'] + SOL[f'b{j}'] - U[s])),
                                    abs(ev(SOL[f'a{j}'] * X + SOL[f'b{j}'] * Y - U[e_])))
    return w


def plot(R_val=1.0, fluxes=(0.0, 0.25, 0.5, 1.0), k_max=8.0, fname='P1_over_P2.png'):
    import matplotlib.pyplot as plt
    kk = np.linspace(0.02, k_max, 40001)
    bad = np.abs(kk * R_val - np.round(kk * R_val)) < 2e-3      # kR in Z: формулы 0/0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fl in fluxes:
        y = f_ratio(kk, R_val, n_from_flux(fl, R_val)); y[bad] = np.nan
        ax.plot(kk, y, lw=1.2, label=f'Ф/Ф₀ = {fl}')
    ax.axhline(1, color='gray', lw=0.7, ls='--')
    ax.set_yscale('log'); ax.set_xlabel('k')
    ax.set_ylabel('P₁ / P₂   (большая окружность / малые)')
    ax.set_title(f'P₁/P₂ от k,  R = {R_val}')
    ax.legend(); ax.grid(alpha=0.3, which='both'); fig.tight_layout(); fig.savefig(fname, dpi=150)
    return fname


if __name__ == '__main__':
    print('Рёбра (начало, конец):', RING)
    for nm in ['r', 't', 'U1', 'U2', 'U3'] + [f'{c}{j}' for j in range(1, 7) for c in 'ab']:
        print(f'\n{nm} =', SOL[nm])
    print('\nПроверки, максимальная ошибка:', run_checks())
    print('B=0, k=1.3, R=1:  P1/P2 =', f_ratio(1.3, 1.0, 0.0))
    print('График:', plot())
