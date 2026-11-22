"""Exact magnetic scattering coefficients for 3 and 4 tangent inner rings."""
import argparse

import sympy as sp


k, R = sp.symbols("k R", positive=True)
f = sp.Symbol("f", real=True)
A, B, G, QL = sp.symbols("A B G Q_L", real=True)


def symbolic_solution(count):
    """Return the full vertex matrix, vertices and all edge coefficients.

    A, B, G, Q_L are real entries of the vertex matrix after dividing
    the covariant Kirchhoff equations by k. See physical_entries().
    The solution applies away from zeros of sin(kL) and det(H).
    """
    if count not in (3, 4):
        raise ValueError("Use 3 or 4 inner rings.")

    middle = count - 1
    H = sp.diag(*([B] * middle))
    for i in range(middle - 1):
        H[i, i + 1] = H[i + 1, i] = G

    Hinv = H.inv().applyfunc(sp.factor)
    d = sp.factor(A - G**2 * Hinv[0, 0])
    h = sp.factor(-G**2 * Hinv[0, middle - 1])
    D, Q = d + sp.I, QL + h
    determinant = D**2 - Q**2
    left = 2 * sp.I * D / determinant
    right = -2 * sp.I * Q / determinant
    coupling = sp.zeros(middle, 2)
    coupling[0, 0] = coupling[-1, 1] = G
    interior = -Hinv * coupling * sp.Matrix([left, right])
    vertices = sp.Matrix([left, *interior, right])

    M = sp.zeros(count + 1)
    M[0, 0] = M[count, count] = A + sp.I
    M[0, count] = M[count, 0] = QL
    M[1:count, 1:count] = H
    M[1:count, 0] = coupling[:, 0]
    M[1:count, count] = coupling[:, 1]
    M[0, 1:count] = coupling[:, 0].T
    M[count, 1:count] = coupling[:, 1].T

    coefficients = {"r": left - 1, "t": right}
    pairs = [(count, 0), (0, count)]
    for i in range(count):
        pairs.extend(((i + 1, i), (i, i + 1)))
    for j, (start, end) in enumerate(pairs, 1):
        L = sp.pi * R / (1 if j <= 2 else count)
        theta = sp.pi * f / (1 if j <= 2 else count**2)
        X = sp.exp(sp.I * (theta + k * L))
        Y = sp.exp(sp.I * (theta - k * L))
        u, v = vertices[start], vertices[end]
        coefficients[f"a{j}"] = (v - Y*u) / (X - Y)
        coefficients[f"b{j}"] = (X*u - v) / (X - Y)
    return M, vertices, coefficients, d, h


def physical_entries(count):
    """Substitution into the exact expressions, with f = Phi_outer/Phi_0."""
    K = sp.pi * k * R
    s = K / count
    return {A: -2*sp.cot(K) - 2*sp.cot(s),
            B: -4*sp.cot(s),
            G: 2*sp.cos(sp.pi*f/count**2)/sp.sin(s),
            QL: 2*sp.cos(sp.pi*f)/sp.sin(K)}


def verify():
    from ring_scattering import solve
    import numpy as np

    rng = np.random.default_rng(19)
    for n in (3, 4):
        mat, v, coefficients, _, _ = symbolic_solution(n)
        rhs = sp.zeros(n + 1, 1)
        rhs[0] = 2*sp.I
        residual = (mat*v-rhs).applyfunc(sp.cancel)
        assert residual == sp.zeros(n + 1, 1)
        physical = [value.subs(physical_entries(n)) for value in coefficients.values()]
        evaluate = sp.lambdify((k, R, f), physical, "numpy", cse=True)
        errors = []
        for _ in range(100):
            kv, Rv, fv = rng.uniform(.07, 7), rng.uniform(.65, 1.7), rng.uniform(-2, 2)
            r, t, ab, *_ = solve(n, kv, fv, Rv)
            exact = np.array(evaluate(kv, Rv, fv))
            errors.append(np.max(np.abs(exact-np.r_[r, t, ab.ravel()])))
        worst = max(errors)
        assert worst < 1e-8
        print(f"N={n}: exact symbolic vertex residual zero; "
              f"max abs difference vs full edge matching: {worst:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=[3, 4], nargs="+", default=[3, 4])
    parser.add_argument("--physical", action="store_true",
                        help="Substitute k, R and f in every printed coefficient.")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    for count in args.count:
        M, U, coeff, d, h = symbolic_solution(count)
        substitutions = physical_entries(count)
        print(f"\nN = {count}\nM = {M}\nd = {d}\nh = {h}")
        print("Physical entries:")
        for key, value in substitutions.items():
            print(f"{key} = {value}")
        for name, expression in {**dict(zip((f"U{i}" for i in range(count+1)), U)),
                                 **coeff}.items():
            if args.physical:
                expression = expression.subs(substitutions)
            print(f"{name} = {expression}")


if __name__ == "__main__":
    main()
