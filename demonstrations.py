"""
Численные демонстрации к объяснению:
  1) почему a1 != a2 (и т.д.), хотя длины и радиусы равны;
  2) что будет, если навязать a1 = a2, b1 = b2, ...;
  3) зачем ковариантная производная: калибровочная инвариантность и сохранение тока.
"""
import numpy as np

EDGES = {1: (3, 1), 2: (1, 3), 3: (2, 1), 4: (1, 2), 5: (3, 2), 6: (2, 3)}  # j: (начало, конец)


def build(k, R, A, covariant=True):
    """
    Матрица 14x14 для неизвестных (r, T, a1, b1, ..., a6, b6).
    A: dict {j: вектор-потенциал на ребре j} — можно задавать произвольно (калибровка).
    phi_j = a e^{i(k+A)x} + b e^{i(-k+A)x}.
    covariant=True : Кирхгоф для D phi = phi' - iA phi  (множители ik, -ik)
    covariant=False: Кирхгоф для обычной phi'          (множители ip, im)
    """
    L = {j: np.pi * R if j <= 2 else np.pi * R / 2 for j in EDGES}
    col = lambda j: (2 * j, 2 * j + 1)
    def row(): return np.zeros(14, complex)

    def value(j, at_end):
        rw = row(); ia, ib = col(j)
        p, m = k + A[j], -k + A[j]
        rw[ia], rw[ib] = (np.exp(1j * p * L[j]), np.exp(1j * m * L[j])) if at_end else (1, 1)
        return rw

    def deriv(j, at_end):
        rw = row(); ia, ib = col(j)
        p, m = k + A[j], -k + A[j]
        cp, cm = (1j * k, -1j * k) if covariant else (1j * p, 1j * m)
        e = (np.exp(1j * p * L[j]), np.exp(1j * m * L[j])) if at_end else (1, 1)
        rw[ia], rw[ib] = cp * e[0], cm * e[1]
        return rw

    rows, rhs = [], []
    for v in (1, 2, 3):
        vals = [value(j, False) for j, (s, e) in EDGES.items() if s == v] + \
               [value(j, True) for j, (s, e) in EDGES.items() if e == v]
        kir = sum(deriv(j, False) for j, (s, e) in EDGES.items() if s == v) - \
              sum(deriv(j, True) for j, (s, e) in EDGES.items() if e == v)
        if v == 1:                       # phi_0(0) = 1 + r,  -phi_0'(0) = -ik(1-r)
            for vr in vals: vr = vr.copy(); vr[0] -= 1; rows.append(vr); rhs.append(1)
            kir[0] += 1j * k; rows.append(kir); rhs.append(1j * k)
        elif v == 3:                     # phi_7(0) = T,  +phi_7'(0) = ikT
            for vr in vals: vr = vr.copy(); vr[1] -= 1; rows.append(vr); rhs.append(0)
            kir[1] += 1j * k; rows.append(kir); rhs.append(0)
        else:
            for vr in vals[1:]: rows.append(vr - vals[0]); rhs.append(0)
            rows.append(kir); rhs.append(0)
    # колонки: 0 = r, 1 = T, 2j и 2j+1 = a_j, b_j
    return np.array(rows), np.array(rhs, complex)


def solve(k, R, A, covariant=True):
    M, V = build(k, R, A, covariant)
    s = np.linalg.solve(M, V)
    return s


def symmetric_gauge(R, n):
    return {1: n * R, 2: n * R, 3: n * R / 2, 4: n * R / 2, 5: n * R / 2, 6: n * R / 2}


def other_gauge(R, n, chi2=1.7):
    """Калибровочное преобразование phi -> e^{i chi} phi, A -> A + chi'.
    chi = 0 в вершинах 1 и 3 (и на подводящих рёбрах), chi = chi2 в вершине 2,
    линейно по рёбрам. Потоки через ВСЕ циклы не меняются -> физика та же."""
    A = symmetric_gauge(R, n); LS = np.pi * R / 2
    A[3] -= chi2 / LS   # 2 -> 1
    A[4] += chi2 / LS   # 1 -> 2
    A[5] += chi2 / LS   # 3 -> 2
    A[6] -= chi2 / LS   # 2 -> 3
    return A


def names():
    return ['r', 'T'] + [f'{c}{j}' for j in range(1, 7) for c in 'ab']


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    k, R = 1.3, 1.0

    print('=' * 70, '\n1) Даже при B = 0 коэффициенты половин окружности НЕ равны')
    s = dict(zip(names(), solve(k, R, symmetric_gauge(R, 0.0))))
    for j1, j2 in [(1, 2), (3, 4), (5, 6)]:
        print(f'  a{j1} = {s[f"a{j1}"]: .4f}   a{j2} = {s[f"a{j2}"]: .4f}   |  '
              f'b{j1} = {s[f"b{j1}"]: .4f}   b{j2} = {s[f"b{j2}"]: .4f}')
    X = np.exp(1j * k * np.pi * R); Y = 1 / X
    print('  Зеркальная симметрия при B=0 даёт ДРУГОЕ соотношение: a2 = b1*Y_L, b2 = a1*X_L')
    print(f'     a2 - b1*Y_L = {abs(s["a2"] - s["b1"] * Y):.1e},   b2 - a1*X_L = {abs(s["b2"] - s["a1"] * X):.1e}')

    n = 0.4
    s = dict(zip(names(), solve(k, R, symmetric_gauge(R, n))))
    XL, YL = np.exp(1j * (k + n * R) * np.pi * R), np.exp(1j * (-k + n * R) * np.pi * R)
    print(f'  При B != 0 (n={n}) и это соотношение ломается: |a2 - b1*Y_L| = {abs(s["a2"] - s["b1"] * YL):.3f}')

    print('=' * 70, '\n2) Навязываем a1=a2, b1=b2, a3=a4, b3=b4, a5=a6, b5=b6 (8 неизвестных, 14 уравнений)')
    for nn in (0.0, 0.4):
        M, V = build(k, R, symmetric_gauge(R, nn))
        P = np.zeros((14, 8))
        P[0, 0] = P[1, 1] = 1
        for pair, (ca, cb) in enumerate([(2, 4), (6, 8), (10, 12)]):   # a1/a2, a3/a4, a5/a6
            P[ca, 2 + 2 * pair] = P[cb, 2 + 2 * pair] = 1                # a
            P[ca + 1, 3 + 2 * pair] = P[cb + 1, 3 + 2 * pair] = 1        # b
        x, res, rank, _ = np.linalg.lstsq(M @ P, V, rcond=None)
        print(f'  n={nn}: ранг расширенной системы = {np.linalg.matrix_rank(np.c_[M @ P, V])} > '
              f'ранг матрицы = {rank}  ->  несовместна, невязка = {np.linalg.norm(M @ P @ x - V):.3f}')

    print('=' * 70, '\n3) Ковариантная vs обычная производная, n =', n)
    for cov in (True, False):
        s1 = solve(k, R, symmetric_gauge(R, n), cov)
        s2 = solve(k, R, other_gauge(R, n), cov)
        tag = 'D = d/dx - iA' if cov else 'обычная d/dx '
        print(f'  [{tag}]  калибровка 1: |T|^2 = {abs(s1[1])**2:.6f}, |r|^2+|T|^2 = {abs(s1[0])**2 + abs(s1[1])**2:.6f}'
              f'   |  калибровка 2: |T|^2 = {abs(s2[1])**2:.6f}, |r|^2+|T|^2 = {abs(s2[0])**2 + abs(s2[1])**2:.6f}')
    print('  Калибровка 1 (одинаковый A на всех рёбрах кольца) — особая: в каждой вершине\n'
          '  sum_out A - sum_in A = 0, поэтому там обе производные дают одно и то же.')
