import numpy as np
from HW2.tools import norm_p
import matplotlib.pyplot as plt


def approximate_model_generator(x, b_matrix):
    return lambda p: f_provider.f(*x) + np.dot(f_provider.grad(*x), p) + 0.5 * np.dot(np.dot(p, b_matrix), p)


def cauchy_point(x, b_matrix, delta):
    g_matrix = f_provider.grad(*x)
    gT_b_g = np.dot(np.dot(g_matrix, b_matrix), g_matrix)
    gT_g = np.dot(g_matrix, g_matrix)
    g_norm = norm_p(g_matrix)

    if gT_b_g > 0 and abs(gT_g / gT_b_g) * g_norm < delta:
        alpha = gT_g / gT_b_g
    else:
        alpha = delta / g_norm

    return -alpha * g_matrix


def dog_leg(x, b_matrix, delta):
    g_matrix = f_provider.grad(*x)
    gT_b_g = np.dot(np.dot(g_matrix, b_matrix), g_matrix)
    gT_g = np.dot(g_matrix, g_matrix)
    grad_path_best_alpha = gT_g / gT_b_g
    p_u = -grad_path_best_alpha * g_matrix
    p_b = -np.dot(np.linalg.inv(b_matrix), g_matrix)
    tau = delta / norm_p(p_u)
    if tau <= 1:
        return tau * p_u
    elif tau <= 2:
        return p_u + (tau-1) * (p_b - p_u)
    else:
        p_b_norm = norm_p(p_b)
        if p_b_norm <= delta:
            return p_b
        else:
            return p_b * delta / p_b_norm


MAX_STEP, RHO_TOLERANCE, DELTA_HAT, ETA, EQUALITY_TOLERANCE = 10000, 1/.000001, 1.5, 1 / 5, .000001
def trust_region(x, function_provider, subproblem_solver=cauchy_point, plot=False):
    global f_provider
    f_provider = function_provider

    delta = DELTA_HAT/2
    step = 0
    rho = RHO_TOLERANCE
    initial_x = x
    plot_y = []
    while step < MAX_STEP and abs(rho) <= RHO_TOLERANCE:
        # b_matrix = np.zeros([np.shape(x)[0], np.shape(x)[0]])
        b_matrix = f_provider.hessian(*x)  # use hessian matrix as Bk
        p = subproblem_solver(x, b_matrix, delta)
        approximate_model = approximate_model_generator(x, b_matrix)
        delta_f = f_provider.f(*x) - f_provider.f(*(x + p))
        delta_m = approximate_model(np.zeros(np.shape(x)[0])) - approximate_model(p)
        rho = delta_f / delta_m
        if rho < .25:
            delta = .25 * delta
        else:
            if rho > .75 and abs(norm_p(p) - delta) < EQUALITY_TOLERANCE:
                delta = min(2 * delta, DELTA_HAT)

        if rho > ETA:
            x = x + p

        step += 1
        # print('{}. new x: {}, rho={}'.format(step, x, rho))
        plot_y.append(rho)

    if plot:
        plt.title('Rho on each step for point ({}, {})'.format(*initial_x))
        plt.ylabel('Rho')
        plt.xlabel('step')
        plt.plot(plot_y)
        plt.show()
    print('number of steps: ', step)
    return x
