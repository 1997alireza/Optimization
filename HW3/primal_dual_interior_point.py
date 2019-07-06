import numpy as np
from numpy import matmul
from numpy.linalg import inv as matinv, norm

TOLERANCE, EASY_TOLERANCE = .001, .01


def backtracking_line_search_for_interior_point(f_provider, P, q, x, landa, delta_x, delta_landa, t):
    alpha, m, c = 1, .5, .5  # parameters

    r_dual = lambda x: residual_dual(f_provider, P, q, x, landa)
    r_cent = lambda x: residual_cent(P, q, x, landa, t)
    norm_residual = lambda x: norm(np.append(r_dual(x), r_cent(x), axis=0))

    while True:
        alpha = alpha * m
        new_x = x + alpha * delta_x
        new_landa = landa + alpha * delta_landa
        consts = np.min(new_landa) > 0 > np.max(matmul(P, new_x) - q) and \
                 norm_residual(new_x) <= (1 - c * alpha) * norm_residual(x)

        if alpha < TOLERANCE or consts:
            return alpha


def residual_dual(f_provider, P, q, x, landa):
    fc_g = P
    return f_provider.grad(x) + matmul(fc_g.T, landa)


def residual_cent(P, q, x, landa, t):
    fc_f = matmul(P, x) - q
    m = np.shape(P)[0]
    return - matmul(np.diag(np.reshape(landa, -1)), fc_f) - np.ones([m, 1]) / t


def primal_dual_interior_point(f_provider, P, q, mu=10):
    """
    solving min f(x), subject to P.x <= q
    q should be positive
    """
    m, n = np.shape(P)
    x = np.zeros([n, 1])
    landa = np.random.rand(m, 1) + .1  # must be positive

    r_pri = 0  # is always zero because the problem has not equality constraint

    surrogate_duality_gaps = []
    r_feas = []  # = (|r_pri| ** 2 + |r_dual| ** 2) ** 0.5 ==() |r_dual|, (in this problem r_pri is zero)
    iterations = []

    for _i in range(10000):
        fc_f = matmul(P, x) - q
        fc_g = P
        n_hat = matmul(fc_f.T, landa)[0, 0]
        t = mu * m / n_hat

        matrix_factor = np.zeros([m + n, m + n])

        m_1_1 = f_provider.hessian(x)
        m_1_2 = fc_g.T
        m_2_1 = - matmul(np.diag(np.reshape(landa, -1)), fc_g)
        m_2_2 = np.diag(np.reshape(fc_f, -1))

        for i in range(n):
            for j in range(n):
                matrix_factor[i, j] = m_1_1[i, j]
        for i in range(n):
            for j in range(m):
                matrix_factor[i, j + n] = m_1_2[i, j]
        for i in range(m):
            for j in range(n):
                matrix_factor[i + n, j] = m_2_1[i, j]
        for i in range(m):
            for j in range(m):
                matrix_factor[i + n, j + n] = m_2_2[i, j]

        r_dual = residual_dual(f_provider, P, q, x, landa)
        r_cent = residual_cent(P, q, x, landa, t)

        matrix_residual = -np.append(r_dual, r_cent, axis=0)

        matrix_delta = matmul(matinv(matrix_factor), matrix_residual)

        delta_x = matrix_delta[:n]
        delta_landa = matrix_delta[n:]

        s = backtracking_line_search_for_interior_point(f_provider, P, q, x, landa, delta_x, delta_landa, t)

        x += s * delta_x
        landa += s * delta_landa

        r_dual_norm = norm(r_dual)

        iterations.append(_i)
        surrogate_duality_gaps.append(t)
        r_feas.append(r_dual_norm)

        if r_dual_norm <= TOLERANCE and n_hat <= TOLERANCE:
            return x, iterations, surrogate_duality_gaps, r_feas

    return x, iterations, surrogate_duality_gaps, r_feas

