import numpy as np
from numpy import matmul
from numpy.linalg import inv as matinv, norm
from math import log
from HW3.function_provider import FunctionProvider


TOLERANCE, EASY_TOLERANCE = .0000001, .01


# def backtracking_line_search_for_unconstrained_newton(f_provider, x, p):
#     t0, alpha, beta = 1, .25, .5
#     t = t0
#
#     while norm(f_provider.grad(x + t * p)) > (1 - alpha * t) * norm(f_provider.grad(x)):
#         t *= beta
#
#     return t


def first_wolf_condition(f_provider, x, p, alpha, c):
    """
    sufficient decrease condition
    """
    new_x = x + alpha * p
    return f_provider.f(new_x) <= f_provider.f(x) + c * alpha * np.dot(p.T, f_provider.grad(x))


def backtracking_line_length(f_provider, x, p):
    alpha, m, c = 1, .5, .5  # parameters
    while not first_wolf_condition(f_provider, x, p, alpha, c):
        alpha = alpha * m
        if alpha < TOLERANCE:
            return alpha
    return alpha


def compute_newton_step_unconstrained_problem(f_provider, x):
    return -matmul(matinv(f_provider.hessian(x)), f_provider.grad(x))


def newton_method_unconstrained_problem(f_provider, x0):
    """
    solving min f(x) using second-order approximation (newton method)
    """
    x = np.reshape(x0, [-1, 1])

    for i in range(500):
        delta_x = compute_newton_step_unconstrained_problem(f_provider, x)
        h = f_provider.hessian(x)
        landa_pow2 = abs(matmul(matmul(delta_x.T, h), delta_x)[0][0])

        if landa_pow2/2 <= 10:
            return x, i

        t = backtracking_line_length(f_provider, x, delta_x)

        x += t * delta_x

    return x, i


def barrier_method_phase_one(f_provider, P, q, mu):
    m, n = np.shape(P)

    P_new = np.zeros([m, n + 1])
    for i in range(m):
        for j in range(n):
            P_new[i, j] = P[i, j]
        P_new[i, n] = -1

    f_new = lambda xs: f_provider.f(xs[:-1])
    g_new = lambda xs: np.reshape(np.array([*f_provider.grad(xs[:-1]), np.zeros([1, 1])]), [-1, 1])
    h_new = lambda xs: np.array([*(np.append(f_provider.hessian(xs[:-1]), np.zeros([n, 1]), axis=1)),
                                  np.zeros(n + 1)])
    f_provider_new = FunctionProvider(f_new, g_new, h_new)

    initial_x = np.random.rand(np.shape(P)[1], 1)
    cons_value_on_initial = matmul(P, initial_x) - q

    s = abs(np.max(cons_value_on_initial)) * 1.5
    # to ensure that (P.x - q)i < s is established, so the initial point for phase one barrier problem is strictly feasible

    xs = np.reshape(np.append(initial_x, s), [-1, 1])

    return barrier_method(f_provider_new, P_new, q, xs, mu)


def barrier_method(f_provider, P, q, x=None, mu=2.):
    """
    solving min f(x), subject to P.x <= q
    """
    q = np.reshape(q, (-1, 1))
    m = np.shape(q)[0]
    assert np.shape(P)[0] == m

    def phi_f(x):
        r = 0
        for i in range(m):
            pi = np.reshape(P[i], (1, -1))
            qi = np.reshape(q[i], (1, 1))
            fi = matmul(pi, x) - qi
            if fi >= 0:
                r -= - 1 / TOLERANCE
            else:
                r -= log(-fi)

        return r

    def phi_g(x):
        r = 0
        for i in range(m):
            pi = np.reshape(P[i], (1, -1))
            qi = np.reshape(q[i], (1, 1))
            fi = matmul(pi, x) - qi
            fi_grad = pi.T
            r += fi_grad / (-fi)

        return r

    def phi_h(x):
        r = 0
        for i in range(m):
            pi = np.reshape(P[i], (1, -1))
            qi = np.reshape(q[i], (1, 1))
            fi = matmul(pi, x) - qi
            fi_grad = pi.T
            r += matmul(fi_grad, fi_grad.T) / (fi*fi)

        return r

    phi = FunctionProvider(phi_f, phi_g, phi_h)

    if x is None:
        # xs = barrier_method_phase_one(f_provider, P, q, mu)
        #
        # s = xs[-1][0]
        # print(s)
        # assert s <= 0, 'Problem is infeasible'
        # assert s < 0, 'Problem is feasible but not strictly, so you can\'t solve it using barrier method'
        #
        # x = xs[:-1]

        x = np.zeros([np.shape(P)[1], 1])

    newton_iters = []
    gaps = []
    t = 1
    while True:
        x, newton_iter_num = newton_method_unconstrained_problem(t * f_provider + phi, x)
        duality_gap = m/t
        newton_iters.append(newton_iter_num)
        gaps.append(duality_gap)
        if duality_gap < EASY_TOLERANCE:
            return x, newton_iters, gaps

        t *= mu
