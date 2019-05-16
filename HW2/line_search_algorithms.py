import numpy as np
from HW2.tools import norm_p, RosenbrockProvider as F_Provider


def steepest_descent_direction(x):
    return -1 * F_Provider.grad(*x)


def newton_direction(x):
    h_i = np.linalg.inv(F_Provider.hessian(*x))
    g = F_Provider.grad(*x)
    return -1 * np.dot(h_i, g)


last_BFGS_H, last_step_length, last_x = [None] * 3
def BFGS_direction(x):
    global last_BFGS_H, last_step_length, last_x

    if last_BFGS_H is None:
        H = np.linalg.inv(F_Provider.hessian(*x))
    else:
        last_p = -1 * np.dot(last_BFGS_H, F_Provider.grad(*x))
        x_kPlus1 = last_x
        s = last_step_length * last_p
        x_k = x_kPlus1 - s
        y = F_Provider.grad(*x_kPlus1) - F_Provider.grad(*x_k)
        m = 1 / np.dot(s, y)
        s = s[:, None]  # shape: (2,) -> (2,1)
        y = y[:, None]
        s_yT = np.dot(s, y.T)
        y_sT = np.dot(y, s.T)
        s_sT = np.dot(s, s.T)
        mL = np.identity(2) - m * s_yT
        mR = np.identity(2) - m * y_sT
        rL = np.dot(np.dot(mL, last_BFGS_H), mR)
        rR = m * s_sT
        H = rL + rR

    last_BFGS_H = H
    return -1 * np.dot(H, F_Provider.grad(*x))


def first_wolf_condition(x, p, alpha, c):
    """
    sufficient decrease condition
    """
    new_x = x + alpha * p
    return F_Provider.f(*new_x) <= F_Provider.f(*x) + c * alpha * np.dot(p, F_Provider.grad(*x))


def backtracking_line_length(x, p):
    alpha, m, c = 1, .5, .5  # parameters
    while not first_wolf_condition(x, p, alpha, c):
        alpha = alpha * m
    return alpha


TOLERANCE, MAX_STEP = .001, 1000
def line_search(x, descent_direction_algorithm=steepest_descent_direction):
    global last_BFGS_H, last_x, last_step_length
    last_BFGS_H = None
    step = 0
    while norm_p(F_Provider.grad(*x)) >= TOLERANCE and step < MAX_STEP:
        p = descent_direction_algorithm(x)
        alpha = backtracking_line_length(x, p)
        x = x + alpha * p
        last_x, last_step_length = x, alpha
        step += 1
        # print('{}. new x: {} using alpha={}'.format(step, x, alpha))
    print('number of steps: ', step)
    return x
