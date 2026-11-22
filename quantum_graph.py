"""
Квантовый граф: большая окружность радиуса R (рёбра 1, 2) + две малые окружности
радиуса R/2 (рёбра 3,4 и 5,6), подводящее ребро 0 и отводящее ребро 7.
Перпендикулярное магнитное поле B.

Ориентация рёбер (по рисунку):
    0: (-inf) -> 1        7: 3 -> (+inf)
    1: 3 -> 1  (большая, правая)     2: 1 -> 3  (большая, левая)
    3: 2 -> 1  (верхняя малая, пр.)  4: 1 -> 2  (верхняя малая, лев.)
    5: 3 -> 2  (нижняя малая, пр.)   6: 2 -> 3  (нижняя малая, лев.)
Все рёбра колец идут против часовой стрелки => вектор-потенциал на всех
одного знака: A_L = n*R (большие), A_S = n*R/2 (малые), n = pi*B/Phi_0.

На ребре j:  phi_j(x) = a_j e^{i p_j x} + b_j e^{i m_j x},
             p = k + A,  m = -k + A   (как в исходных обозначениях).
Лиды: phi_0 = e^{ikx} + r e^{-ikx},  phi_7 = T e^{ikx}.

Условия в вершинах:
  * непрерывность: значения всех рёбер в вершине совпадают;
  * Кирхгоф с КОВАРИАНТНОЙ производной D = d/dx - iA:
        sum_{исходящие} D phi_j(0) - sum_{входящие} D phi_j(L_j) = 0.
    (С обычной производной ток не сохраняется: |r|^2+|T|^2 != 1.)
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------- символы
k, R, n = sp.symbols('k R n', positive=True)
I = sp.I

L_L, L_S = sp.pi * R, sp.pi * R / 2           # длины полуокружностей
A_L, A_S = n * R, n * R / 2                   # вектор-потенциал на рёбрах
th_L, th_S = A_L * L_L, A_S * L_S             # магнитные фазы: n*pi*R^2, n*pi*R^2/4
p_L, m_L = k + A_L, -k + A_L
p_S, m_S = k + A_S, -k + A_S

# Ребро u -> w длины L с фазой th:
#   D phi(0) = k/sin(kL) * (-u cos kL + w e^{-i th})
#   D phi(L) = k/sin(kL) * (-u e^{i th} + w cos kL)
# Кирхгоф в вершине 2 даёт  Phi2 = cos(th_S) (Phi1 + Phi3) / (2 cos kL_S),
# а в вершинах 1 и 3 (после деления на k):
#   D*Phi1 + Q*Phi3 = 2i ,   Q*Phi1 + D*Phi3 = 0
Dc = (I - 2 * sp.cot(k * L_L) - 2 * sp.cot(k * L_S)
      + sp.cos(th_S) ** 2 / (sp.sin(k * L_S) * sp.cos(k * L_S)))
Qc = 2 * sp.cos(th_L) / sp.sin(k * L_L) + sp.cos(th_S) ** 2 / (sp.sin(k * L_S) * sp.cos(k * L_S))

Phi1 = 2 * I * Dc / (Dc ** 2 - Qc ** 2)
Phi3 = -2 * I * Qc / (Dc ** 2 - Qc ** 2)
Phi2 = sp.cos(th_S) * (Phi1 + Phi3) / (2 * sp.cos(k * L_S))

r = Phi1 - 1
T = Phi3


def ab(u, w, L, th):
    """Коэффициенты на ребре u -> w: phi(0)=u, phi(L)=w."""
    den = 2 * I * sp.sin(k * L)
    a = (w * sp.exp(-I * th) - u * sp.exp(-I * k * L)) / den
    b = (u * sp.exp(I * k * L) - w * sp.exp(-I * th)) / den
    return a, b


edges = {  # j: (начало, конец, длина, фаза, p, m)
    1: (Phi3, Phi1, L_L, th_L, p_L, m_L),
    2: (Phi1, Phi3, L_L, th_L, p_L, m_L),
    3: (Phi2, Phi1, L_S, th_S, p_S, m_S),
    4: (Phi1, Phi2, L_S, th_S, p_S, m_S),
    5: (Phi3, Phi2, L_S, th_S, p_S, m_S),
    6: (Phi2, Phi3, L_S, th_S, p_S, m_S),
}
coef = {j: ab(u, w, L, th) for j, (u, w, L, th, _, _) in edges.items()}


# ------------------------------------------- шаг 3: интегралы |phi_j|^2
# |a e^{ipx} + b e^{imx}|^2 = |a|^2 + |b|^2 + 2 Re(a conj(b) e^{2ikx})   (A сокращается)
# P_j = (|a|^2+|b|^2) L + Re( a conj(b) (e^{2ikL} - 1) / (i k) )
def P_edge(a, b, L):
    return ((sp.Abs(a) ** 2 + sp.Abs(b) ** 2) * L
            + sp.re(a * sp.conjugate(b) * (sp.exp(2 * I * k * L) - 1) / (I * k)))


P = {j: P_edge(*coef[j], edges[j][2]) for j in edges}
P_inner = P[3] + P[4] + P[5] + P[6]   # малые окружности
P_outer = P[1] + P[2]                 # большая окружность
ratio = P_outer / P_inner

_args = (k, R, n)
f_r = sp.lambdify(_args, r, 'numpy')
f_T = sp.lambdify(_args, T, 'numpy')
f_Phi = sp.lambdify(_args, [Phi1, Phi2, Phi3], 'numpy')
f_ab = {j: sp.lambdify(_args, list(coef[j]), 'numpy') for j in coef}
f_P = {j: sp.lambdify(_args, P[j], 'numpy') for j in P}
_f_ratio_c = sp.lambdify(_args, ratio, 'numpy')
f_ratio = lambda kv, Rv, nv: np.real(_f_ratio_c(kv, Rv, nv))


def n_from_flux(flux_over_Phi0, R_val):
    """Поток через большую окружность Ф/Ф0 = n R^2  ->  n."""
    return flux_over_Phi0 / R_val ** 2


def check_all(trials=200, seed=1):
    rng = np.random.default_rng(seed)
    worst = dict(unit=0, direct=0, quad=0, kirch=0)
    for _ in range(trials):
        kv, Rv, nv = rng.uniform(0.2, 6), rng.uniform(0.5, 2), rng.uniform(-1, 1)
        # (1) сохранение тока
        rv, Tv = f_r(kv, Rv, nv), f_T(kv, Rv, nv)
        worst['unit'] = max(worst['unit'], abs(abs(rv) ** 2 + abs(Tv) ** 2 - 1))
        # (2) независимое прямое решение 14x14 по a_j, b_j
        sol = direct_solve(kv, Rv, nv)
        mine = [rv, Tv] + sum((list(f_ab[j](kv, Rv, nv)) for j in range(1, 7)), [])
        worst['direct'] = max(worst['direct'], np.max(np.abs(np.array(mine) - sol)))
        # (3) интегралы: квадратура vs формула
        for j in range(1, 7):
            a, b = f_ab[j](kv, Rv, nv)
            L = np.pi * Rv * (1 if j <= 2 else 0.5)
            pj = kv + nv * Rv * (1 if j <= 2 else 0.5)
            mj = -kv + nv * Rv * (1 if j <= 2 else 0.5)
            x = np.linspace(0, L, 20001)
            q = np.trapezoid(np.abs(a * np.exp(1j * pj * x) + b * np.exp(1j * mj * x)) ** 2, x)
            worst['quad'] = max(worst['quad'], abs(q - f_P[j](kv, Rv, nv)) / q)
    res = symbolic_residuals()
    worst['kirch'] = max(abs(complex(sp.N(e.subs({k: 1.7, R: 1.1, n: 0.37}), 30))) for e in res)
    return worst


def direct_solve(kv, Rv, nv):
    LL, LS = np.pi * Rv, np.pi * Rv / 2
    pars = {j: ((LL, kv + nv * Rv, -kv + nv * Rv, nv * Rv) if j <= 2
                else (LS, kv + nv * Rv / 2, -kv + nv * Rv / 2, nv * Rv / 2)) for j in range(1, 7)}
    start = {1: 3, 2: 1, 3: 2, 4: 1, 5: 3, 6: 2}
    end = {1: 1, 2: 3, 3: 1, 4: 2, 5: 2, 6: 3}
    idx = lambda j: (2 + 2 * (j - 1), 3 + 2 * (j - 1))
    def val(j, at_end):          # строка значения ребра j в вершине
        row = np.zeros(14, complex); L, p, m, A = pars[j]; ia, ib = idx(j)
        row[ia], row[ib] = (np.exp(1j * p * L), np.exp(1j * m * L)) if at_end else (1, 1)
        return row
    def cder(j, at_end):         # ковариантная производная
        row = np.zeros(14, complex); L, p, m, A = pars[j]; ia, ib = idx(j)
        e = (np.exp(1j * p * L), np.exp(1j * m * L)) if at_end else (1, 1)
        row[ia], row[ib] = 1j * kv * e[0], -1j * kv * e[1]
        return row
    M, rhs = [], []
    # значения на рёбрах в вершине v
    for v in (1, 2, 3):
        rows = [val(j, False) for j in start if start[j] == v] + [val(j, True) for j in end if end[j] == v]
        if v == 1:   # phi_0(0) = 1 + r
            lead = np.zeros(14, complex); lead[0] = 1
            for rw in rows: M.append(rw - lead); rhs.append(1)
        elif v == 3:  # phi_7(0) = T
            lead = np.zeros(14, complex); lead[1] = 1
            for rw in rows: M.append(rw - lead); rhs.append(0)
        else:
            for rw in rows[1:]: M.append(rw - rows[0]); rhs.append(0)
        kr = sum(cder(j, False) for j in start if start[j] == v) - sum(cder(j, True) for j in end if end[j] == v)
        if v == 1: kr[0] += 1j * kv; M.append(kr); rhs.append(1j * kv)      # -phi_0'(0) = -ik(1-r)
        elif v == 3: kr[1] += 1j * kv; M.append(kr); rhs.append(0)          # +phi_7'(0) = ikT
        else: M.append(kr); rhs.append(0)
    return np.linalg.solve(np.array(M), np.array(rhs, complex))


def symbolic_residuals():
    """Подставляет найденные коэффициенты в непрерывность и Кирхгоф."""
    def v(j, end_):
        a, b = coef[j]; L, p, m = edges[j][2], edges[j][4], edges[j][5]
        return a * sp.exp(I * p * L) + b * sp.exp(I * m * L) if end_ else a + b
    def d(j, end_):
        a, b = coef[j]; L, p, m = edges[j][2], edges[j][4], edges[j][5]
        return I * k * (a * sp.exp(I * p * L) - b * sp.exp(I * m * L)) if end_ else I * k * (a - b)
    return [
        v(2, 0) - (1 + r), v(4, 0) - (1 + r), v(1, 1) - (1 + r), v(3, 1) - (1 + r),
        v(3, 0) - v(6, 0), v(4, 1) - v(3, 0), v(5, 1) - v(3, 0),
        v(1, 0) - T, v(5, 0) - T, v(2, 1) - T, v(6, 1) - T,
        d(2, 0) + d(4, 0) - d(1, 1) - d(3, 1) - I * k * (1 - r),
        d(3, 0) + d(6, 0) - d(4, 1) - d(5, 1),
        d(1, 0) + d(5, 0) + I * k * T - d(2, 1) - d(6, 1),
    ]


def plot(R_val=1.0, fluxes=(0.0, 0.25, 0.5, 1.0), k_max=8.0, fname='ratio_vs_k(1).png'):
    import matplotlib.pyplot as plt
    kk = np.linspace(1e-3, k_max, 40001) + 1.234e-6   # сдвиг, чтобы не попасть ровно в sin(kL)=0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # в точках kR ∈ Z (sin или cos от k*pi*R/2 = 0) формулы дают 0/0 -> выкидываем окрестность
    bad = np.abs(kk * R_val - np.round(kk * R_val)) < 2e-3
    for fl in fluxes:
        y = f_ratio(kk, R_val, n_from_flux(fl, R_val))
        y[bad] = np.nan
        ax.plot(kk, y, lw=1.1, label=f'Ф/Ф₀ = {fl}')
    ax.axhline(1, color='gray', lw=0.7, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel('P₁ / P₂  (Большая окружности / малая)')
    ax.set_title(f'Отношение вероятностей, R = {R_val}')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(fname, dpi=150)
    return fname


if __name__ == '__main__':
    print('r    =', sp.simplify(r))
    print('T    =', sp.simplify(T))
    for j in range(1, 7):
        print(f'a{j} =', coef[j][0]); print(f'b{j} =', coef[j][1])
    print('\nПроверки (максимальная ошибка):', check_all())
    print('График:', plot())
