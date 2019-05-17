import numpy as np
from HW2.tools import norm_p


def steepest_descent_direction(x):
    return -1 * f_provider.grad(*x)


def newton_direction(x):
    h_i = np.linalg.inv(f_provider.hessian(*x))
    g = f_provider.grad(*x)
    return -1 * np.dot(h_i, g)


last_BFGS_H, last_step_length, last_x = [None] * 3
def BFGS_direction(x):
    global last_BFGS_H, last_step_length, last_x

    if last_BFGS_H is None:
        H = np.linalg.inv(f_provider.hessian(*x))
    else:
        last_p = -1 * np.dot(last_BFGS_H, f_provider.grad(*x))
        x_kPlus1 = last_x
        s = last_step_length * last_p
        x_k = x_kPlus1 - s
        y = f_provider.grad(*x_kPlus1) - f_provider.grad(*x_k)
        # print("-----------", s,"============", y)
        m = 1 / np.dot(s, y)
        s = s[:, None]  # shape: (size of x,) -> (size of x,1)
        y = y[:, None]
        s_yT = np.dot(s, y.T)
        y_sT = np.dot(y, s.T)
        s_sT = np.dot(s, s.T)
        mL = np.identity(np.shape(x)[0]) - m * s_yT
        mR = np.identity(np.shape(x)[0]) - m * y_sT
        rL = np.dot(np.dot(mL, last_BFGS_H), mR)
        rR = m * s_sT
        H = rL + rR

    last_BFGS_H = H
    return -1 * np.dot(H, f_provider.grad(*x))


def first_wolf_condition(x, p, alpha, c):
    """
    sufficient decrease condition
    """
    new_x = x + alpha * p
    return f_provider.f(*new_x) <= f_provider.f(*x) + c * alpha * np.dot(p, f_provider.grad(*x))


def backtracking_line_length(x, p):
    alpha, m, c = 1, .5, .5  # parameters
    while not first_wolf_condition(x, p, alpha, c):
        alpha = alpha * m
        if alpha < TOLERANCE:
            return alpha
    return alpha


TOLERANCE, MAX_STEP = .001, 10000
def line_search(x, function_provider, descent_direction_algorithm=steepest_descent_direction):
    global f_provider, last_BFGS_H, last_x, last_step_length
    f_provider = function_provider
    last_BFGS_H = None
    step = 0
    while norm_p(f_provider.grad(*x)) >= TOLERANCE and step < MAX_STEP:
        p = descent_direction_algorithm(x)
        alpha = backtracking_line_length(x, p)
        x = x + alpha * p
        last_x, last_step_length = x, alpha
        step += 1
        # print('{}. new x: {} using alpha={}'.format(step, x, alpha))
    print('number of steps: ', step)
    return x
